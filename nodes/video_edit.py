import math
import torch
import numpy as np
import os
import sys
import subprocess
import folder_paths
from .video_utils.utils import ffmpeg_path
from comfy.utils import ProgressBar
import uuid
import tempfile

CATEGORY_NAME = "WJNode/video/merge"

#测试视频衔接
class Video_OverlappingSeparation_test:
    DESCRIPTION = """
    Separate video frames into two segments with a specified number of overlapping frames at the beginning and end, 
        for testing video coherence
    Overlapping frames and total frames are recommended to be even numbers. 
        If the total number of frames is less than the overlapping frames, divide them evenly
    将视频帧分离为开始段和结尾段有指定重叠帧数的两段，用于测试视频衔接
    重叠帧和总帧数建议为偶数,若总帧数比重叠帧少,将平均分割
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "OverlappingFrame": ("INT", {"default": 8, "min": 2, "max": 4096,"step":2}),
            },
            "optional": {
                "video": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","IMAGE","MASK","MASK","INT")
    RETURN_NAMES = ("video1","video2","mask1","mask2","OverlappingFrame")
    FUNCTION = "fade"
    def fade(self, OverlappingFrame, video = None, mask = None):
        #初始化数值
        video1, video2, mask1, mask2 = None,None,None,None
        if OverlappingFrame % 2 == 1: OverlappingFrame -= 1
        if OverlappingFrame <=1 : OverlappingFrame = 2

        #处理data
        if video is not None:
            video1, video2 = self.truncation(video,OverlappingFrame)
        if mask is not None:
            mask1, mask2 = self.truncation(mask,OverlappingFrame)
        return (video1, video2, mask1, mask2, OverlappingFrame)

    #分割
    def truncation(self,data,frame):
        n = len(data)
        d1, d2 = data[:int(n/2)], data[int(n/2):]
        if frame >= n:
            print("Error: The total number of frames is less than the overlapping frames, and this processing will be skipped")
            print("错误：总帧数比重叠帧少,将跳过此次处理")
            #raise ValueError("错误：总帧数至少要比重叠帧的2倍多2帧")
        else:
            d1, d2 = data[:int((n+frame)/2)], data[int((n-frame)/2):]
        return (d1,d2)

#视频渐入渐出
class Video_fade:
    DESCRIPTION = """
    Support video fade in and fade out
        mask: Local gradient is currently under development
        Exponential: Index gradient development in progress
    支持视频渐入和渐出
        mask:局部渐变正在开发中
        Exponential:指数渐变开发中
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video1": ("IMAGE",),
                "video2": ("IMAGE",),
                "OverlappingFrame": ("INT", {"default": 8, "min": 2, "max": 4096}),
                "method":(["Linear","Cosine","Exponential"],{"default":"Cosine"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","INT")
    RETURN_NAMES = ("video","frames")
    FUNCTION = "fade"
    def fade(self, video1, video2, OverlappingFrame, method="Exponential", mask=None):
        frame = video1.shape[0]+video2.shape[0]-OverlappingFrame
        video = None

        # 如果输入视频有alpha通道，则去除alpha通道
        if video1.shape[3] == 4:
            video1 = video1[...,:3]
        if video2.shape[3] == 4:
            video2 = video2[...,:3]

        # 如果OverlappingFrame为0，则直接合并视频
        if OverlappingFrame == 2:
            video = torch.cat((video1, video2), dim=0)
        elif OverlappingFrame == 3:
            video_temp = video1[-1]*0.5+video2[0]*0.5
            video = torch.cat((video1[:-1], video_temp.unsqueeze(0), video2[1:]), dim=0)
        else:
            Gradient = []
            if method == "Linear":
                frame = OverlappingFrame + 1
                Gradient = [i/frame for i in range(frame)][1:][::-1]
            elif method == "Cosine":
                f = OverlappingFrame + 1
                Gradient = [0.5+0.5*math.cos((i+1)*math.pi/f) for i in range(f)]
            elif method == "Exponential":
                frame = int(OverlappingFrame/2)
                if OverlappingFrame % 2 == 0:
                    Gradient = [math.exp(-(i)/OverlappingFrame) for i in range(frame)]
                    Gradient = Gradient + list(1-np.array(Gradient[::-1]))
                else:
                    Gradient = [math.exp(-(i+1)/OverlappingFrame) for i in range(frame+1)]
                    Gradient = Gradient[0:-1] + [0.5] + list(1-np.array(Gradient[::-1][1:]))
            else:
                raise TypeError("Error: Selected class not specified in the list\n错误：选择的不是列表内指定的类")
            video_temp = torch.zeros((0,*video1.shape[1:]), device=video1.device)
            for i in range(OverlappingFrame):
                video_temp_0 = video1[i-OverlappingFrame] * Gradient[i] + video2[i] * (1-Gradient[i])
                video_temp = torch.cat((video_temp, video_temp_0.unsqueeze(0)), dim=0)
            video = torch.cat((video1[:-OverlappingFrame], video_temp, video2[OverlappingFrame:]), dim=0)
        return (video, frame)

CATEGORY_NAME = "WJNode/video/video_file"

#保存视频 搬运的VHS的代码
class SaveMP4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "MP4Video"}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 400.0, "step": 0.1}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME
    FUNCTION = "save_mp4"

    def save_mp4(
        self,
        images,
        filename_prefix,
        frame_rate,
        crf,
        audio=None,
    ):
        save_output = True
        if images is None or len(images) == 0:
            return ("",)
            
        # 获取输出目录
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        (full_output_folder, filename, _, _, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        
        # 创建进度条
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        
        # 获取计数器
        counter = 1
        existing_files = [f for f in os.listdir(full_output_folder) if f.startswith(filename) and f.endswith(".mp4")]
        if existing_files:
            # 提取现有文件的计数器
            counters = []
            for f in existing_files:
                try:
                    counter_str = f.replace(filename, "").replace(".mp4", "").strip("_")
                    if counter_str.isdigit():
                        counters.append(int(counter_str))
                except:
                    pass
            if counters:
                counter = max(counters) + 1
        
        # 准备输出MP4文件
        output_file = f"{filename}_{counter:05}.mp4"
        output_path = os.path.join(full_output_folder, output_file)
        
        # 检查ffmpeg是否可用
        if ffmpeg_path is None:
            raise ProcessLookupError("ffmpeg is required for video outputs and could not be found.")
        
        # 准备ffmpeg参数
        first_image = images[0]
        
        # 确保视频尺寸是偶数，如果不是则调整
        height, width = first_image.shape[0], first_image.shape[1]
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        
        # 如果需要调整大小，裁剪所有图像
        if width != first_image.shape[1] or height != first_image.shape[0]:
            print(f"注意：调整视频尺寸从 {first_image.shape[1]}x{first_image.shape[0]} 到 {width}x{height} 以符合H.264要求")
            adjusted_images = []
            for img in images:
                adjusted_images.append(img[:height, :width])
            images = adjusted_images
        
        dimensions = f"{width}x{height}"
        has_alpha = first_image.shape[-1] == 4
        i_pix_fmt = 'rgba' if has_alpha else 'rgb24'
        
        # 转换图像为字节
        images_bytes = [self.tensor_to_bytes(img).tobytes() for img in images]
        
        # 如果有音频输入，使用一个临时文件
        temp_video_path = None
        if audio is not None:
            import tempfile
            import uuid
            
            # 创建唯一的临时文件名，避免冲突
            unique_id = uuid.uuid4().hex
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"ffmpeg_temp_{unique_id}.mp4")
            
            # 确保文件不存在
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    # 如果删除失败，再生成一个随机文件名
                    unique_id = uuid.uuid4().hex
                    temp_video_path = os.path.join(temp_dir, f"ffmpeg_temp_{unique_id}.mp4")
            
            video_output = temp_video_path
        else:
            video_output = output_path
        
        # 设置ffmpeg命令
        args = [
            ffmpeg_path, "-v", "error", 
            "-f", "rawvideo", 
            "-pix_fmt", i_pix_fmt,
            "-s", dimensions, 
            "-r", str(frame_rate), 
            "-i", "-", 
            "-c:v", "libx264", 
            "-preset", "medium", 
            "-crf", str(crf), 
            "-pix_fmt", "yuv420p",
            "-y",  # 强制覆盖输出文件
            video_output
        ]
        
        # 执行ffmpeg命令
        env = os.environ.copy()
        with subprocess.Popen(args, stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env) as proc:
            try:
                for i, image in enumerate(images_bytes):
                    proc.stdin.write(image)
                    pbar.update(1)
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
                
                # 等待进程完成
                proc.wait()
                if proc.returncode != 0:
                    raise Exception(f"FFmpeg返回错误代码 {proc.returncode}")
                    
            except BrokenPipeError:
                res = proc.stderr.read()
                raise Exception("FFmpeg子进程错误:\n" + res.decode("utf8", errors="ignore"))
            except Exception as e:
                if hasattr(proc, 'stderr') and proc.stderr:
                    error_msg = proc.stderr.read().decode("utf8", errors="ignore")
                    raise Exception(f"FFmpeg错误: {str(e)}\n{error_msg}")
                else:
                    raise e
            finally:
                # 重置进度条
                pbar.update(0)
                
                # 确保子进程结束
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except:
                        pass
        
        if len(res) > 0:
            print(res.decode("utf8", errors="ignore"), end="", file=sys.stderr)
        
        # 如果有音频，添加到视频中
        if audio is not None and os.path.exists(temp_video_path):
            try:
                # 设置音频参数
                channels = audio['waveform'].size(1)
                
                # 执行音频合并命令
                mux_args = [
                    ffmpeg_path, "-v", "error", 
                    "-i", temp_video_path,
                    "-ar", str(audio['sample_rate']), 
                    "-ac", str(channels),
                    "-f", "f32le", 
                    "-i", "-", 
                    "-c:v", "copy", 
                    "-c:a", "aac", 
                    "-b:a", "192k", 
                    "-shortest", 
                    "-y",  # 强制覆盖输出文件
                    output_path
                ]
                
                audio_data = audio['waveform'].squeeze(0).transpose(0,1).numpy().tobytes()
                
                try:
                    res = subprocess.run(mux_args, input=audio_data, env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("FFmpeg子进程错误:\n" + e.stderr.decode("utf8", errors="ignore"))
                
                if res.stderr:
                    print(res.stderr.decode("utf8", errors="ignore"), end="", file=sys.stderr)
                
                print(f"成功保存带音频的视频到: {output_path}")
                
            except Exception as e:
                print(f"添加音频到视频失败: {str(e)}")
                # 如果音频添加失败，使用无音频视频
                if temp_video_path and os.path.exists(temp_video_path):
                    import shutil
                    try:
                        shutil.copy(temp_video_path, output_path)
                        print(f"已保存无音频视频到: {output_path}")
                    except Exception as copy_err:
                        print(f"复制视频文件失败: {str(copy_err)}")
            finally:
                # 清理临时文件
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception as del_err:
                        print(f"删除临时文件失败: {str(del_err)}")
        else:
            print(f"成功保存视频到: {output_path}")
        
        # 返回最终输出路径
        return (output_path,)
    
    def tensor_to_bytes(self, tensor, bits=8):
        tensor = tensor.cpu().numpy() * (2**bits-1)
        return np.clip(tensor, 0, (2**bits-1)).astype(np.uint8)

#批量保存视频
class SaveMP4_batch:
    DESCRIPTION = """
    Save Video_Batch output from Cutting_video as MP4 files.
    Features:
    1. Support H.264 codec with yuv420p pixel format
    2. Adjustable video quality via CRF parameter
    3. Batch processing support
    4. Optional audio support
    将Cutting_Video的Video_Batch输出保存为MP4文件。
    说明：用于批量保存视频
    1.将Video_Batch保存为H.264/yuv420p像素格式
    2.通过fps/CRF参数调节视频帧数和质量
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Video_Batch": ("Video_Batch",),
                "filename_prefix": ("STRING", {"default": "MP4Video"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 400.0, "step": 0.1}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Video_filenames",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    
    def save_video(self, Video_Batch, filename_prefix, fps=25.0, crf=23):
        # 检查ffmpeg是否可用
        if ffmpeg_path is None:
            raise ProcessLookupError("ffmpeg is required for video outputs and could not be found.")
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        (full_output_folder, filename, _, _, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        
        # 获取视频和音频数据
        videos = Video_Batch.get("video", [])
        audios = Video_Batch.get("audio", [None] * len(videos))
        
        # 确保视频列表非空
        if not videos or len(videos) == 0:
            print("警告：视频批次中没有视频数据。")
            return ("没有视频数据可保存",)
        
        # 获取基础计数器
        base_counter = 1
        existing_files = [f for f in os.listdir(full_output_folder) if f.startswith(filename) and f.endswith(".mp4")]
        if existing_files:
            counters = []
            for f in existing_files:
                try:
                    counter_str = f.replace(filename, "").replace(".mp4", "").strip("_")
                    if counter_str.isdigit():
                        counters.append(int(counter_str))
                except:
                    pass
            if counters:
                base_counter = max(counters) + 1
        
        # 保存所有输出文件的路径
        output_paths = []
        
        # 处理每个视频
        for idx, video in enumerate(videos):
            if video is None or len(video) == 0:
                print(f"警告：批次中第 {idx+1} 个视频为空，已跳过。")
                continue
            
            # 获取与此视频关联的音频
            audio = None if idx >= len(audios) else audios[idx]
            
            # 创建计数器
            counter = base_counter + idx
            
            # 准备输出MP4文件
            output_file = f"{filename}_{counter:05}.mp4"
            output_path = os.path.join(full_output_folder, output_file)
            
            # 创建进度条
            num_frames = len(video)
            pbar = ProgressBar(num_frames)
            print(f"保存第 {idx+1}/{len(videos)} 个视频，帧数：{num_frames}")
            
            # 确保视频尺寸是偶数，如果不是则调整
            height, width = video.shape[1], video.shape[2]
            orig_width, orig_height = width, height
            
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
            # 如果需要调整大小，裁剪视频
            if width != orig_width or height != orig_height:
                print(f"注意：调整视频尺寸从 {orig_width}x{orig_height} 到 {width}x{height} 以符合H.264要求")
                adjusted_video = video[:, :height, :width]
                video = adjusted_video
            
            # 准备ffmpeg参数 - 已确保宽高为偶数
            dimensions = f"{width}x{height}"
            i_pix_fmt = 'rgb24'  # 已知通道数是3，固定使用rgb24
            
            # 转换图像为字节
            images_bytes = [self.tensor_to_bytes(img).tobytes() for img in video]
            
            # 如果有音频输入，使用临时文件
            temp_video_path = None
            if audio is not None:
                # 创建唯一的临时文件名
                unique_id = uuid.uuid4().hex
                temp_dir = tempfile.gettempdir()
                temp_video_path = os.path.join(temp_dir, f"ffmpeg_temp_{unique_id}.mp4")
                video_output = temp_video_path
            else:
                video_output = output_path
            
            # 设置ffmpeg命令
            args = [
                ffmpeg_path, "-v", "error", 
                "-f", "rawvideo", 
                "-pix_fmt", i_pix_fmt,
                "-s", dimensions, 
                "-r", str(fps), 
                "-i", "-", 
                "-c:v", "libx264", 
                "-preset", "medium", 
                "-crf", str(crf), 
                "-pix_fmt", "yuv420p",
                "-y",
                video_output
            ]
            
            # 执行ffmpeg命令
            env = os.environ.copy()
            with subprocess.Popen(args, stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env) as proc:
                try:
                    for image in images_bytes:
                        proc.stdin.write(image)
                        pbar.update(1)
                    proc.stdin.flush()
                    proc.stdin.close()
                    res = proc.stderr.read()
                    
                    # 等待进程完成
                    proc.wait()
                    if proc.returncode != 0:
                        raise Exception(f"FFmpeg返回错误代码: {proc.returncode}")
                        
                except BrokenPipeError:
                    res = proc.stderr.read()
                    raise Exception("FFmpeg子进程发生错误:\n" + res.decode("utf8", errors="ignore"))
                except Exception as e:
                    if hasattr(proc, 'stderr') and proc.stderr:
                        error_msg = proc.stderr.read().decode("utf8", errors="ignore")
                        raise Exception(f"FFmpeg错误: {str(e)}\n{error_msg}")
                    else:
                        raise e
                finally:
                    # 重置进度条
                    pbar.update(0)
                    
                    # 确保子进程结束
                    if proc.poll() is None:
                        try:
                            proc.kill()
                        except:
                            pass
            
            if len(res) > 0:
                print(res.decode("utf8", errors="ignore"), end="", file=sys.stderr)
            
            # 如果有音频，添加到视频中
            if audio is not None and os.path.exists(temp_video_path):
                try:
                    # 设置音频参数
                    channels = audio['waveform'].size(1)
                    
                    # 执行音频合并命令
                    mux_args = [
                        ffmpeg_path, "-v", "error", 
                        "-i", temp_video_path,
                        "-ar", str(audio['sample_rate']), 
                        "-ac", str(channels),
                        "-f", "f32le", 
                        "-i", "-", 
                        "-c:v", "copy", 
                        "-c:a", "aac", 
                        "-b:a", "192k", 
                        "-shortest", 
                        "-y",
                        output_path
                    ]
                    
                    audio_data = audio['waveform'].squeeze(0).transpose(0,1).numpy().tobytes()
                    
                    try:
                        res = subprocess.run(mux_args, input=audio_data, env=env, capture_output=True, check=True)
                    except subprocess.CalledProcessError as e:
                        raise Exception("FFmpeg音频处理错误:\n" + e.stderr.decode("utf8", errors="ignore"))
                    
                    if res.stderr:
                        print(res.stderr.decode("utf8", errors="ignore"), end="", file=sys.stderr)
                    
                    print(f"成功保存带音频的视频到: {output_path}")
                    
                except Exception as e:
                    print(f"添加音频到视频失败: {str(e)}")
                    # 如果音频添加失败，使用无音频视频
                    if temp_video_path and os.path.exists(temp_video_path):
                        import shutil
                        try:
                            shutil.copy(temp_video_path, output_path)
                            print(f"已保存无音频视频到: {output_path}")
                        except Exception as copy_err:
                            print(f"复制视频文件失败: {str(copy_err)}")
                finally:
                    # 清理临时文件
                    if temp_video_path and os.path.exists(temp_video_path):
                        try:
                            os.remove(temp_video_path)
                        except:
                            pass
            else:
                print(f"成功保存视频到: {output_path}")
            
            # 记录输出路径
            output_paths.append(output_path)
        
        # 返回所有保存的视频路径
        if len(output_paths) == 1:
            return (output_paths[0],)
        return (";".join(output_paths),)
    
    def tensor_to_bytes(self, tensor, bits=8):
        tensor = tensor.cpu().numpy() * (2**bits-1)
        return np.clip(tensor, 0, (2**bits-1)).astype(np.uint8)

CATEGORY_NAME = "WJNode/video/slicing"

#检测空白遮罩分割视频
class Video_MaskBasedSplit:
    DESCRIPTION = """
    Split video, audio and mask sequences based on empty mask frames.
    When a mask becomes empty (all black), the sequences are split at that point.
    Multiple segments may be created, and segments that are too short can be discarded.
    根据空白遮罩帧分割视频、音频和遮罩序列。
    当遮罩变为空（全黑）时，会在该位置分割序列。
    可能会创建多个片段，过短的片段可以被丢弃。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "empty_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "empty_frames_count": ("INT", {"default": 1, "min": 1, "max": 100}),
                "min_segment_frames": ("INT", {"default": 5, "min": 1, "max": 1000}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }
    CATEGORY = CATEGORY_NAME
    OUTPUT_IS_LIST = (True,True,True)
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO")
    RETURN_NAMES = ("image_segments", "mask_segments", "audio_segments")
    FUNCTION = "split_by_mask"
    
    def split_by_mask(self, images, masks, empty_threshold, empty_frames_count, min_segment_frames, audio=None):
        # 检查输入
        if len(masks.shape) == 2:  # 单帧遮罩，扩展为批次
            masks = masks.unsqueeze(0)
        
        # 初始化结果列表
        image_segments = []
        mask_segments = []
        audio_segments = []
        
        # 检测空白遮罩帧
        empty_mask_indices = []
        consecutive_empty = 0
        
        for i in range(masks.shape[0]):
            # 计算当前遮罩帧的平均值
            mask_mean = masks[i].mean().item()
            
            # 如果平均值低于阈值，认为是空白帧
            if mask_mean <= empty_threshold:
                consecutive_empty += 1
                if consecutive_empty >= empty_frames_count:
                    # 记录分割点（连续空白帧的第一帧）
                    split_index = i - consecutive_empty + 1
                    if split_index not in empty_mask_indices:
                        empty_mask_indices.append(split_index)
            else:
                consecutive_empty = 0
        
        # 如果没有检测到空白帧，直接返回原始序列
        if not empty_mask_indices:
            return ([images], [masks], [audio] if audio is not None else [])
        
        # 添加序列开始和结束索引，便于分割
        split_indices = [0] + empty_mask_indices + [masks.shape[0]]
        split_indices = sorted(list(set(split_indices)))  # 去重并排序
        
        # 根据分割点切分序列
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            
            # 如果片段长度小于最小帧数要求，跳过
            if end_idx - start_idx < min_segment_frames:
                continue
            
            # 分割图像序列
            image_segment = images[start_idx:end_idx]
            image_segments.append(image_segment)
            
            # 分割遮罩序列
            mask_segment = masks[start_idx:end_idx]
            mask_segments.append(mask_segment)
            
            # 如果有音频，也进行分割
            if audio is not None:
                # 假设音频与图像帧一一对应，或者可以按比例分割
                # 这里需要根据实际音频数据格式进行调整
                if hasattr(audio, "shape") and len(audio.shape) > 0:
                    # 如果音频是张量形式
                    audio_segment = audio[start_idx:end_idx]
                    audio_segments.append(audio_segment)
                else:
                    # 如果音频是其他形式，可能需要特殊处理
                    # 这里仅作为占位，实际实现需要根据音频格式调整
                    audio_segments.append(None)
        
        # 如果所有片段都被丢弃（因为太短），返回空列表
        if not image_segments:
            print("Warning: All segments are shorter than min_segment_frames and have been discarded.")
            print("警告：所有片段都短于最小帧数要求，已全部丢弃。")
            return ([], [], [])
        
        return (image_segments, mask_segments, audio_segments)

#检测空白bboxs输出片段数据
class Detecting_videos_mask:
    DESCRIPTION = """
    Detect valid segments in video sequences based on bounding box continuity.
    Features:
    1. Analyze image sequences and corresponding bounding boxes
    2. Identify continuous segments based on frame-to-frame continuity
    3. Filter segments by minimum length requirement
    4. Handle multiple detection regions with confidence-based selection
    5. Output segment data for further processing
    检测视频序列中基于边界框连续性的有效片段
    功能：
    1. 分析图像序列和对应的边界框数据
    2. 基于帧间连续性识别连续片段
    3. 根据最小长度要求过滤片段
    4. 处理多检测区域情况，支持基于置信度的选择
    5. 输出片段数据供后续处理
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_segment_length": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "use_highest_confidence": ("BOOLEAN", {"default": True}),
                "distance_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "bboxes": ("bboxs",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("segments_data", "LIST", "INT")
    RETURN_NAMES = ("segments_data", "segment_bboxes", "segment_count")
    FUNCTION = "detect_video_segments"

    def detect_video_segments(self, images, min_segment_length, use_highest_confidence, distance_threshold, bboxes=None):
        """检测视频序列中的有效片段"""
        # 如果没有提供边界框数据，返回空结果
        if bboxes is None or len(bboxes) == 0:
            return ([], [], 0)
        
        # 确保图像和边界框数量一致
        frame_count = min(len(images), len(bboxes))
        if frame_count == 0:
            return ([], [], 0)
        
        # 处理每一帧的边界框数据
        processed_bboxes = []
        for i in range(frame_count):
            frame_bboxes = bboxes[i]
            
            # 如果当前帧没有边界框，添加None表示
            if not frame_bboxes:
                processed_bboxes.append(None)
                continue
            
            # 如果有多个边界框且use_highest_confidence为True，选择置信度最高的
            if len(frame_bboxes) > 1 and use_highest_confidence:
                highest_conf_bbox = max(frame_bboxes, key=lambda x: x.get('confidence', 0))
                processed_bboxes.append(highest_conf_bbox)
            # 否则使用第一个边界框
            elif len(frame_bboxes) >= 1:
                processed_bboxes.append(frame_bboxes[0])
            else:
                processed_bboxes.append(None)
        
        # 查找连续片段
        segments = []
        segment_bboxes = []  # 存储每个片段对应的边界框数据
        start_idx = None
        
        for i in range(frame_count):
            current_bbox = processed_bboxes[i]
            
            # 如果当前帧有边界框
            if current_bbox is not None:
                # 如果还没有开始一个片段，标记开始
                if start_idx is None:
                    start_idx = i
                # 如果已经在一个片段中，检查与前一帧的连续性
                elif i > 0 and processed_bboxes[i-1] is not None:
                    prev_bbox = processed_bboxes[i-1]['bbox']
                    curr_bbox = current_bbox['bbox']
                    
                    # 计算距离来判断连续性
                    if not self._is_continuous(prev_bbox, curr_bbox, distance_threshold):
                        # 如果片段长度满足要求，记录该片段
                        if i - start_idx >= min_segment_length:
                            segments.append((start_idx, i-1))
                            # 收集该片段的所有边界框
                            segment_bboxes.append(processed_bboxes[start_idx:i])
                        # 重新开始一个新片段
                        start_idx = i
            # 如果当前帧没有边界框但之前有一个进行中的片段
            elif start_idx is not None:
                # 如果片段长度满足要求，记录该片段
                if i - start_idx >= min_segment_length:
                    segments.append((start_idx, i-1))
                    # 收集该片段的所有边界框
                    segment_bboxes.append(processed_bboxes[start_idx:i])
                # 重置开始索引
                start_idx = None
        
        # 处理最后一个可能的片段
        if start_idx is not None and frame_count - start_idx >= min_segment_length:
            segments.append((start_idx, frame_count-1))
            # 收集该片段的所有边界框
            segment_bboxes.append(processed_bboxes[start_idx:frame_count])
        
        return (segments, segment_bboxes, len(segments))
    
    def _is_continuous(self, bbox1, bbox2, distance_threshold):
        """判断两个边界框是否连续（基于中心点距离）"""
        # 提取边界框坐标
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算两个边界框的中心点
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2

        # 计算中心点之间的欧氏距离
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5

        # 计算边界框的平均大小作为参考
        width1 = x2_1 - x1_1
        height1 = y2_1 - y1_1
        width2 = x2_2 - x1_2
        height2 = y2_2 - y1_2
        avg_size = (width1 + height1 + width2 + height2) / 4

        # 使用相对距离：距离与平均大小的比值，与阈值比较
        # 注意：这里distance_threshold参数被解释为距离阈值
        # 距离越小，连续性越好，所以使用小于等于判断
        return distance <= avg_size * distance_threshold

#根据片段数据分割视频
class Cutting_video:
    DESCRIPTION = """
        Cut video sequences based on segment data from Detecting_videos_mask.
        Features:
        1. Extract frames from specified segments in video sequences
        2. Synchronize audio data with extracted frames
        3. Support multiple segment extraction

        根据Detecting_videos_mask的片段数据裁剪视频序列
        功能：
        1. 从视频序列中提取指定片段的帧
        2. 同步提取的帧与音频数据
        3. 支持多片段提取(输出视频批次和边界框批次)
        4. 支持合并片段(输出视频批次和边界框批次)
        5. 根据segment_bboxes数据裁剪crop_size大小的方形视频序列(若不输入bbox则忽略)
        """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "segments_data": ("segments_data",),
                "merge_segments": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_data": ("AUDIO",),
            }
        }
    CATEGORY = CATEGORY_NAME
    #OUTPUT_IS_LIST = (True,True,True)
    RETURN_TYPES = ("Video_Batch",)
    RETURN_NAMES = ("Video_Batch",)
    FUNCTION = "cut_video"

    def cut_video(self, images, segments_data, merge_segments, audio_data=None):
        """根据片段数据segments_data裁剪视频序列和音频数据"""
        if images.shape[-1] == 4: # 如果输入是RGBA图像，则转换为RGB
            images = images[...,0:3]
        image_batch = []
        for i in segments_data:
            image_batch.append(images[...,i[0]:i[-1]])
        if merge_segments:
            image_batch = [torch.cat(image_batch,dim=0)]

        audio_bath = [None,]
        if audio_data is not None:
            audio_bath = []
            sample_rate = audio_data["sample_rate"]
            waveform = audio_data["waveform"]
            n = int(waveform.shape[-1]/images.shape[0])
            for i in segments_data:
                audio_bath.append({"sample_rate":sample_rate,"waveform":waveform[...,i[0]*n:i[-1]*n]})
        else:
            audio_bath = audio_bath * len(segments_data)
        
        return({"video":image_batch,"audio":audio_bath},)


NODE_CLASS_MAPPINGS = {
    "Video_OverlappingSeparation_test": Video_OverlappingSeparation_test,
    "Video_Fade": Video_fade,
    "SaveMP4": SaveMP4,
    "SaveMP4_batch": SaveMP4_batch,
    "Video_MaskBasedSplit": Video_MaskBasedSplit,
    "Detecting_videos_mask": Detecting_videos_mask,
    "Cutting_video": Cutting_video,
}