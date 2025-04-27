
import torch
import numpy as np
import logging
import torchaudio.functional as res
import librosa

CATEGORY_NAME = "WJNode/video/Audio"

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

    def audio_scale(self, audio, scale):
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

NODE_CLASS_MAPPINGS = {
    "audio_resample": audio_resample,
    "audio_scale": audio_scale,
}
