
import torch
import numpy as np
import logging
import torchaudio.functional as res
import librosa

CATEGORY_NAME = "WJNode/Audio"

#音频采样率调整
class audio_resample:
    DESCRIPTION = """
    Adjust the audio sampling rate, whether to resample 
    (not resampling can adjust the audio speed, but the frequency will change).
        Parameters: 
        -sample: Target sampling rate, range 1 - 4MHz
        -resample: Whether to resample.
    调整音频采样率，是否重采样(不重采样可调整音频速度，但是频率会变),支持批次。
        参数: 
        -sample: 目标采样率,范围 1 - 4MHz
        -resample: 是否重采样。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample": ("INT", {"default": 24000, "min": 1, "max": 4096000}),
                "re_sample": ("BOOLEAN", {"default": True}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "original_sample")
    FUNCTION = "audio_sampling"

    def audio_sampling(self, audio, sample, re_sample):
        self.sample = sample
        self.re_sample = re_sample

        if isinstance(audio, (list, tuple)): #音频组
            audio_list = []
            sample_list = []
            for i in audio:
                audio_test ,sample_test= self.sampling(i)
                audio_list.append(audio_test)   
                sample_list.append(sample_test)
            return (audio_list,sample_list)
        elif isinstance(audio, dict):
            return self.sampling(audio)
        else:
            print("Error: unsupported audio data type!")
            return (audio,None)

    def sampling(self, audio): #单个音频
        audio_sample = audio["sample_rate"]
        if audio_sample != self.sample:
            audio_data = audio["waveform"]
            if isinstance(audio_data, torch.Tensor) and self.re_sample:
                audio_data = res.resample(audio_data, audio_sample, self.sample)
            elif isinstance(audio_data, np.ndarray) and self.re_sample:
                audio_data = librosa.resample(
                    audio_data, orig_sr=audio_sample, target_sr=self.sample)
            else:
                logging.error(
                    "Error-MaskGCT-Audio_resample: unsupported audio data type!", exc_info=True)
            new_audio = {
                "sample_rate": self.sample,
                "waveform": audio_data
            }
        else:
            new_audio = audio
        return (new_audio, audio_sample)

#音频速度简单调整
class audio_scale:
    DESCRIPTION = """
    Audio time duration adjustment
        Parameter: - scale: Audio time duration multiplier, range 0.001-99999
    音频速度简单调整
        参数: -scale: 音频时长倍数,范围 0.001-99999
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "scale": ("FLOAT", {"default": 1, "min": 0.001, "max": 99999, "step": 0.001}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "audio_scale"

    def audio_scale(self, audio, scale,):
        if scale == 1:
            return (audio,)
        else:
            new_audio = {}
            new_audio["sample_rate"] = audio["sample_rate"]
            audio_data = torch.tensor([])
            audio_data = audio["waveform"]
            s=list(audio_data.shape)
            s[1] = 0;s[2] = int(round(s[2]/scale))
            audio_data_new = torch.zeros(*s)
            for i in audio_data[0]:
                new_i = np.array(i)
                new_i = librosa.effects.time_stretch(new_i, rate = scale)
                new_i = torch.tensor(new_i)
                audio_data_new = torch.cat((audio_data_new, new_i.repeat(1,1,1)), dim=1)
            new_audio["waveform"] = audio_data_new
        return (new_audio,)

#音频对应视频时长和帧数计算
class AudioDuration_wan:
    DESCRIPTION = """
    Calculate the frame length of a video/wan video based on time (in seconds)  
    Can calculate duration and video frame rate through audio (ignore custom time if audio is input)
    通过时间(秒)计算视频/wan视频的帧长度
    可通过音频计算时长和视频帧数(如果输入了音频则忽略自定义时间)
    """
    @classmethod
    def INPUT_TYPES(s):
        alignment_list = ["None","4n+1","6n+1","8n+1(wan2.1)","10n+1","15n+1","16n+1"]
        return {
            "required": {
                "time_custom": ("FLOAT", {"default": 0, "min": 0, "max": 1000000, "step": 0.001}),
                "video_fps": ("FLOAT", {"default": 16, "min": 1, "max": 640, "step": 0.001}),
                "alignment_frames": (alignment_list, {"default": "8n+1(wan2.1)"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("FLOAT","INT")  # 返回音频的时长，单位为秒，浮点数格式
    RETURN_NAMES = ("time(s)", "video_frames")
    CATEGORY = CATEGORY_NAME
    FUNCTION = "calculate_duration"

    def calculate_duration(self, time_custom, video_fps, alignment_frames, audio=None):
        segment = 0.0
        duration_seconds = 0.0
        if time_custom == 0 and audio is None: #时长和音频都未输入直接报错
            print("Error: No audio input and custom time is 0!")
            return (segment,0)
        elif time_custom != 0 and audio is None: #直接转换时长为帧数
            duration_seconds = time_custom
        elif time_custom == 0 and audio is not None: #通过音频计算时长
            duration_seconds = self.handle_audio(audio)
        elif time_custom != 0 and audio is not None: 
            #将音频剪切到指定时长后再转换为帧数,暂未开发，使用开始可结束来剪切更有意义
            ...

        frames = int(round(duration_seconds * video_fps))
        if alignment_frames != "None":
            n = int(alignment_frames.split("n+")[0])
            frames = int(round(frames / n)) * n + 1
        return (duration_seconds,frames)
    
    def handle_audio(self, audio):
        waveform = audio['waveform'].squeeze(0).numpy() * 32768  # 转换为numpy并放大
        waveform = waveform.astype(np.int16)
        sample_rate = audio['sample_rate']
        # 使用AudioSegment从numpy数组创建音频
        from pydub import AudioSegment
        segment = AudioSegment(
            data=waveform.tobytes(),
            sample_width=2,  # 16-bit audio
            frame_rate=sample_rate,
            channels=1
        )
        return len(segment) / 1000.0

#裁剪音频
class Audio_Crop_Batch:
    DESCRIPTION = """
    音频裁剪
    输入参数: 
        -audio: 输入音频，支持音频批次
        -start_time: 裁剪开始时间,单位秒,默认0,若为负数则从后面开始计算时长，
        -end_time: 裁剪结束时间,单位秒,若为负数则从后面开始计算时长,
            如果为0则裁剪到音频结束(若此时start_time也为0则不裁剪)，
            如果任意值超出了时间范围，则忽略超出的部分
            如果结束时间小于开始时间，则输出范围内的倒放音频
        -start_batch: 作用于哪些音频批次，开始生效的批次编号(0开始)
        -batch_len: 批次长度,如果为负数则从末尾(-1)计算批次数，如果batch_len
            如果batch_len比start_batch小，则输出范围内的批次，并将批次顺序反转
            如果超出了批次总长，则忽略超出的部分
    输出参数: 
        -select_audio: 输出处理后且在裁剪范围内的音频，支持音频批次
        -start_audio: 输出处理后且在裁剪范围前的音频，支持音频批次
        -end_audio: 输出处理后且在裁剪范围后的音频，支持音频批次
        -start_batch: 输出没有处理的音频批次
        -end_batch: 输出处理后且在裁剪范围后的音频批次编号

    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {"default": 0, "min": -1000000, "max": 1000000, "step": 0.001}),
                "end_time": ("FLOAT", {"default": 0, "min": -1000000, "max": 1000000, "step": 0.001}),
                "start_batch": ("INT", {"default": 0, "min": -1000000, "max": 1000000, "step": 1}),
                "batch_len": ("INT", {"default": -1, "min": -1000000, "max": 1000000, "step": 1}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO","AUDIO","AUDIO","AUDIO","AUDIO")
    RETURN_NAMES = ("select_audio","start_audio","end_audio","start_batch","end_batch")
    FUNCTION = "crop_audio"

    def crop_audio(self, audio, start_time, end_time, start_batch, batch_len):
        # 处理音频批次
        if isinstance(audio, (list, tuple)):
            audio_list = audio
        else:
            audio_list = [audio]
        
        # 计算批次范围
        total_batch = len(audio_list)
        if start_batch < 0:
            start_batch = total_batch + start_batch
            if start_batch < 0:
                start_batch = 0
        
        if batch_len < 0:
            end_batch = total_batch + batch_len + 1
            if end_batch < 0:
                end_batch = 0
        else:
            end_batch = start_batch + batch_len
        
        if end_batch > total_batch:
            end_batch = total_batch
        
        # 初始化输出列表
        select_audio_list = []
        start_audio_list = []
        end_audio_list = []
        start_batch_list = []
        end_batch_list = []
        
        # 处理批次顺序
        batch_range = range(start_batch, end_batch)
        reverse_order = False
        if batch_len < 0 and abs(batch_len) > start_batch:
            reverse_order = True
            batch_range = reversed(batch_range)
        
        # 处理每个批次的音频
        for i, audio_item in enumerate(audio_list):
            if i < start_batch:
                start_batch_list.append(audio_item)
                continue
            elif i >= end_batch:
                end_batch_list.append(audio_item)
                continue
            
            # 获取音频时长
            sample_rate = audio_item["sample_rate"]
            waveform = audio_item["waveform"]
            audio_length = waveform.shape[-1] / sample_rate
            
            # 计算裁剪时间点
            start_sec = start_time
            end_sec = end_time
            
            # 处理负数时间（从末尾计算）
            if start_sec < 0:
                start_sec = audio_length + start_sec
                if start_sec < 0:
                    start_sec = 0
            
            if end_sec <= 0:
                end_sec = audio_length if end_sec == 0 else audio_length + end_sec
                if end_sec < 0:
                    end_sec = 0
            
            # 如果时间超出范围，调整到有效范围内
            if start_sec > audio_length:
                start_sec = audio_length
            if end_sec > audio_length:
                end_sec = audio_length
            
            # 转换时间到采样点
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            
            # 创建裁剪后的音频
            if start_sec == 0 and end_sec == 0:
                # 不裁剪
                select_audio = audio_item
                start_audio = {"sample_rate": sample_rate, "waveform": torch.zeros_like(waveform[:,:,:0])}
                end_audio = {"sample_rate": sample_rate, "waveform": torch.zeros_like(waveform[:,:,:0])}
            else:
                # 裁剪音频
                if start_sample < end_sample:
                    # 正向裁剪
                    select_waveform = waveform[:,:,start_sample:end_sample]
                    start_waveform = waveform[:,:,:start_sample] if start_sample > 0 else torch.zeros_like(waveform[:,:,:0])
                    end_waveform = waveform[:,:,end_sample:] if end_sample < waveform.shape[-1] else torch.zeros_like(waveform[:,:,:0])
                else:
                    # 倒放音频（结束时间小于开始时间）
                    select_waveform = torch.flip(waveform[:,:,end_sample:start_sample], dims=[-1])
                    start_waveform = waveform[:,:,:end_sample] if end_sample > 0 else torch.zeros_like(waveform[:,:,:0])
                    end_waveform = waveform[:,:,start_sample:] if start_sample < waveform.shape[-1] else torch.zeros_like(waveform[:,:,:0])
                
                select_audio = {"sample_rate": sample_rate, "waveform": select_waveform}
                start_audio = {"sample_rate": sample_rate, "waveform": start_waveform}
                end_audio = {"sample_rate": sample_rate, "waveform": end_waveform}
            
            # 添加到输出列表
            select_audio_list.append(select_audio)
            start_audio_list.append(start_audio)
            end_audio_list.append(end_audio)
        
        # 如果需要反转顺序
        if reverse_order:
            select_audio_list.reverse()
            start_audio_list.reverse()
            end_audio_list.reverse()
        
        # 处理输出格式
        if len(select_audio_list) == 1:
            select_audio_out = select_audio_list[0]
        else:
            select_audio_out = select_audio_list
        
        if len(start_audio_list) == 1:
            start_audio_out = start_audio_list[0]
        else:
            start_audio_out = start_audio_list
        
        if len(end_audio_list) == 1:
            end_audio_out = end_audio_list[0]
        else:
            end_audio_out = end_audio_list
        
        if len(start_batch_list) == 1:
            start_batch_out = start_batch_list[0]
        elif len(start_batch_list) == 0:
            start_batch_out = {"sample_rate": sample_rate, "waveform": torch.zeros(1, 1, 0)}
        else:
            start_batch_out = start_batch_list
        
        if len(end_batch_list) == 1:
            end_batch_out = end_batch_list[0]
        elif len(end_batch_list) == 0:
            end_batch_out = {"sample_rate": sample_rate, "waveform": torch.zeros(1, 1, 0)}
        else:
            end_batch_out = end_batch_list
        
        return (select_audio_out, start_audio_out, end_audio_out, start_batch_out, end_batch_out)

#音频批次选择
class Audio_Batch_Edit:
    DESCRIPTION = """
    音频批次编辑节点
    输入参数:
        -audio0: 输入音频，支持音频批次
        -audio1: 输入音频，支持音频批次，和其它audio输入合并，非必须输入
        -audio2: 输入音频，支持音频批次，和其它audio输入合并，非必须输入
        -audio3: 输入音频，支持音频批次，和其它audio输入合并，非必须输入
        -start_batch: audio0-3合并后的音频批次选择的起始编号，从0开始
        -batch_len: 批次长度,如果为负数则从末尾(-1)计算批次数，如果batch_len
            如果batch_len比start_batch小，则输出范围内的批次，并将批次顺序反转
            如果超出了批次总长，则忽略超出的部分
    输出参数: 
        -select_audio_batch: 输出选择的音频批次
        -start_audio_batch: 输出选择的音频批次前面的批次
        -end_audio_batch: 输出选择的音频批次后面的批次
    """
    CATEGORY = CATEGORY_NAME
    INPUT_TYPES = lambda: {
        "required": {
            "audio0": ("AUDIO",),
            "start_batch": ("INT", {"default": 0, "min": -1000000, "max": 1000000}),
            "batch_len": ("INT", {"default": -1, "min": -1000000, "max": 1000000}),
        },
        "optional": {
            "audio1": ("AUDIO",),
            "audio2": ("AUDIO",),
            "audio3": ("AUDIO",),
        }
    }
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("select_audio_batch", "start_audio_batch", "end_audio_batch")
    FUNCTION = "audio_batch_edit"
    def audio_batch_edit(self, audio0, start_batch, batch_len, audio1=None, audio2=None, audio3=None):
        # 合并所有输入的音频批次
        audio_list = []
        
        # 处理audio0
        if isinstance(audio0, (list, tuple)):
            audio_list.extend(audio0)
        else:
            audio_list.append(audio0)
        
        # 处理可选的audio1-3
        for audio in [audio1, audio2, audio3]:
            if audio is not None:
                if isinstance(audio, (list, tuple)):
                    audio_list.extend(audio)
                else:
                    audio_list.append(audio)
        
        # 计算批次范围
        total_batch = len(audio_list)
        if start_batch < 0:
            start_batch = total_batch + start_batch
            if start_batch < 0:
                start_batch = 0
        
        if batch_len < 0:
            end_batch = total_batch + batch_len + 1
            if end_batch < 0:
                end_batch = 0
        else:
            end_batch = start_batch + batch_len
        
        if end_batch > total_batch:
            end_batch = total_batch
        
        # 初始化输出列表
        select_audio_list = []
        start_audio_list = []
        end_audio_list = []
        
        # 处理批次顺序
        reverse_order = False
        if batch_len < 0 and abs(batch_len) > start_batch:
            reverse_order = True
        
        # 分配音频批次
        for i, audio_item in enumerate(audio_list):
            if i < start_batch:
                start_audio_list.append(audio_item)
            elif i < end_batch:
                select_audio_list.append(audio_item)
            else:
                end_audio_list.append(audio_item)
        
        # 如果需要反转顺序
        if reverse_order:
            select_audio_list.reverse()
        
        # 处理输出格式
        if len(select_audio_list) == 1:
            select_audio_out = select_audio_list[0]
        elif len(select_audio_list) == 0:
            # 如果没有选择的音频，创建一个空的音频
            sample_rate = 44100  # 默认采样率
            if len(audio_list) > 0:
                sample_rate = audio_list[0]["sample_rate"]
            select_audio_out = {"sample_rate": sample_rate, "waveform": torch.zeros(1, 1, 0)}
        else:
            select_audio_out = select_audio_list
        
        if len(start_audio_list) == 1:
            start_audio_out = start_audio_list[0]
        elif len(start_audio_list) == 0:
            sample_rate = 44100  # 默认采样率
            if len(audio_list) > 0:
                sample_rate = audio_list[0]["sample_rate"]
            start_audio_out = {"sample_rate": sample_rate, "waveform": torch.zeros(1, 1, 0)}
        else:
            start_audio_out = start_audio_list
        
        if len(end_audio_list) == 1:
            end_audio_out = end_audio_list[0]
        elif len(end_audio_list) == 0:
            sample_rate = 44100  # 默认采样率
            if len(audio_list) > 0:
                sample_rate = audio_list[0]["sample_rate"]
            end_audio_out = {"sample_rate": sample_rate, "waveform": torch.zeros(1, 1, 0)}
        else:
            end_audio_out = end_audio_list
        
        return (select_audio_out, start_audio_out, end_audio_out)

#音频批次合并为通道
class Audio_MergeBatch_To_Channel:
    DESCRIPTION = """
    音频批次合并为单批次音频节点
    输入参数:
        -audio0: 输入音频，支持音频批次
        -audio1: 输入音频，支持音频批次，和其它audio输入合并，非必须输入
        -audio2: 输入音频，支持音频批次，和其它audio输入合并，非必须输入
        -audio3: 输入音频，支持音频批次，和其它audio输入合并，非必须输入
    输出参数: 
        -audio: 输出合并后的音频
    """
    CATEGORY = CATEGORY_NAME
    INPUT_TYPES = lambda: {
        "required": {
            "audio0": ("AUDIO",),
        },
        "optional": {
            "audio1": ("AUDIO",),
            "audio2": ("AUDIO",),
            "audio3": ("AUDIO",),
        }
    }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "audio_batch_to_channel"
    def audio_batch_to_channel(self, audio0, audio1=None, audio2=None, audio3=None):
        # 合并所有输入的音频批次
        audio_list = []
        
        # 处理所有输入的音频
        for audio in [audio0, audio1, audio2, audio3]:
            if audio is not None:
                if isinstance(audio, (list, tuple)):
                    audio_list.extend(audio)
                else:
                    audio_list.append(audio)
        
        # 如果没有音频，返回空音频
        if len(audio_list) == 0:
            return ({"sample_rate": 44100, "waveform": torch.zeros(1, 1, 0)},)
        
        # 如果只有一个音频，直接返回
        if len(audio_list) == 1:
            return (audio_list[0],)
        
        # 获取所有音频的采样率
        sample_rates = [audio["sample_rate"] for audio in audio_list]
        
        # 检查所有音频的采样率是否相同
        if len(set(sample_rates)) > 1:
            # 如果采样率不同，将所有音频重采样到第一个音频的采样率
            target_sample_rate = sample_rates[0]
            for i in range(1, len(audio_list)):
                if sample_rates[i] != target_sample_rate:
                    audio_data = audio_list[i]["waveform"]
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = res.resample(audio_data, sample_rates[i], target_sample_rate)
                    elif isinstance(audio_data, np.ndarray):
                        audio_data = librosa.resample(
                            audio_data, orig_sr=sample_rates[i], target_sr=target_sample_rate)
                    audio_list[i] = {
                        "sample_rate": target_sample_rate,
                        "waveform": audio_data
                    }
        
        # 获取目标采样率
        target_sample_rate = sample_rates[0]
        
        # 找出最长的音频长度
        max_length = max([audio["waveform"].shape[-1] for audio in audio_list])
        
        # 创建新的音频数据，通道数等于音频数量
        channels = sum([audio["waveform"].shape[1] for audio in audio_list])
        merged_waveform = torch.zeros(1, channels, max_length)
        
        # 合并音频
        channel_idx = 0
        for audio in audio_list:
            waveform = audio["waveform"]
            audio_channels = waveform.shape[1]
            audio_length = waveform.shape[-1]
            
            # 将音频数据复制到合并的音频中
            merged_waveform[0, channel_idx:channel_idx+audio_channels, :audio_length] = waveform[0, :, :]
            channel_idx += audio_channels
        
        # 创建合并后的音频
        merged_audio = {
            "sample_rate": target_sample_rate,
            "waveform": merged_waveform
        }
        
        return (merged_audio,)





NODE_CLASS_MAPPINGS = {
    "audio_resample": audio_resample,
    "audio_scale": audio_scale, 
    "AudioDuration_wan": AudioDuration_wan,
    "Audio_Crop_Batch": Audio_Crop_Batch,
    "Audio_Batch_Edit": Audio_Batch_Edit,
    "Audio_MergeBatch_To_Channel": Audio_MergeBatch_To_Channel,

}
