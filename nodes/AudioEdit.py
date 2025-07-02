
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


NODE_CLASS_MAPPINGS = {
    "audio_resample": audio_resample,
    "audio_scale": audio_scale, 
    "AudioDuration_wan": AudioDuration_wan,
}
