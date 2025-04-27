# Original node author: https://github.com/prodogape
# Original project: https://github.com/prodogape/ComfyUI-EasyOCR
# Modification: Loading model separation,

import folder_paths
import os
import logging
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("ComfyUI-EasyOCR")
CATEGORY_NAME = "WJNode/Other-plugins/EasyOCR"

lang_list = {
    "English": "en",
    "简体中文": "ch_sim", "繁體中文": "ch_tra",
    "العربية": "ar",
    "Azərbaycan": "az",
    "Euskal": "eu",
    "Bosanski": "bs",
    "Български": "bg",
    "Català": "ca",
    "Hrvatski": "hr",
    "Čeština": "cs",
    "Dansk": "da",
    "Nederlands": "nl",
    "Eesti": "et",
    "Suomi": "fi",
    "Français": "fr",
    "Galego": "gl",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "עברית": "he",
    "हिन्दी": "hi",
    "Magyar": "hu",
    "Íslenska": "is",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "한국어": "ko",
    "Latviešu": "lv",
    "Lietuvių": "lt",
    "Македонски": "mk",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Српски": "sr",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Español": "es",
    "Svenska": "sv",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "Tiếng Việt": "vi",
}

def get_lang_list():
    result = []
    for key, value in lang_list.items():
        result.append(key)
    return result

def get_classes(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == value:
                result.append(value)
                break
    return result

def get_classes2(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == key:
                result.append(value)
                break
    return result

def plot_boxes_to_image(image_pil, tgt, group_id = None):
    H, W = tgt["size"]
    result = tgt["result"]
    res_mask = []

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }

    for item in result:
        formatted_points, label, threshold = item
        formatted_points = torch.round(torch.tensor(formatted_points)).int()
        x1, y1 = formatted_points[0]
        x2, y2 = formatted_points[2]

        # Save labelme json 保存json数据
        shape = {
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": group_id,
            "shape_type": "rectangle",
            "flags": {},
        }
        labelme_data["shapes"].append(shape)

        # 绘制方形遮罩
        mask = torch.zeros((1,H,W), dtype=torch.float16)
        mask[0,y1:y2,x1:x2] = 1.0
        res_mask.append(mask)

    # 未检测到返回一个全黑遮罩
    if len(res_mask) == 0:
        res_mask.append(torch.zeros((1,H,W), dtype=torch.float16))

    return res_mask, labelme_data

class load_EasyOCR_model:
    DESCRIPTION = """
    Original node author: https://github.com/prodogape
    Original project: https://github.com/prodogape/ComfyUI-EasyOCR
    Modification: Loading model separation,
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gpu": ("BOOLEAN",{"default": True},),
                "detect": (["choose", "input"],{"default": "choose"},),
                "language_list": (get_lang_list(),{"default": "English"},),
                "language_name": ("STRING",{"default": "ch_sim,en,Español", "multiline": False},),
            },
        }
    CATEGORY = CATEGORY_NAME
    FUNCTION = "load_model"
    RETURN_TYPES = ("EasyOCR_model",)
    def load_model(self, gpu, detect, language_list, language_name):
        # OCR检测模型准备
        model_storage_directory = os.path.join(folder_paths.models_dir, "EasyOCR")
        if not os.path.exists(model_storage_directory):
            os.makedirs(model_storage_directory)

        # 语言选择
        language = None
        if detect == "choose":
            language = get_classes2(language_list)
        else:
            language = get_classes(language_name)

        # 加载模型
        try :
            import easyocr
        except:
            raise ImportError("Running this node requires the -easyocr- module")
        reader = easyocr.Reader(language, model_storage_directory=model_storage_directory, gpu=gpu)
        return (reader,)

class ApplyEasyOCR_batch:
    DESCRIPTION = """
    Original node author: https://github.com/prodogape
    Original project: https://github.com/prodogape/ComfyUI-EasyOCR
    Modification: Loading model separation,
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Merge_mask": ("BOOLEAN",{"default": True},),
                "invert_mask": ("BOOLEAN",{"default": False},),
                "EasyOCR_model": ("EasyOCR_model",),
            },
        }

    CATEGORY = CATEGORY_NAME
    FUNCTION = "main"
    RETURN_TYPES = ("MASK","JSON",)

    def main(self, image, Merge_mask, invert_mask, EasyOCR_model):
        res_masks = []
        res_labels = []
        m_masks = []
        _, height, width, _ = image.shape

        i = 0
        for item in image:
            i += 1
            image_pil = Image.fromarray(np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")

            # OCR检测
            result = EasyOCR_model.readtext(np.array(image_pil))
            size = image_pil.size
            pred_dict = {"size": [size[1], size[0]],"result":result}

            # 绘制遮罩
            mask_tensor, labelme_data = plot_boxes_to_image(image_pil, pred_dict, group_id = i)

            res_masks.extend(mask_tensor)
            res_labels.append(labelme_data)

            # 未检测到返回一个全黑遮罩
            if len(res_masks) == 0:
                mask = np.zeros((height, width, 1), dtype=np.uint8)
                empty_mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
                res_masks.extend(empty_mask)

            # 合并遮罩
            if Merge_mask:
                if len(res_masks) >= 1:
                    mask_temp = res_masks[0]
                    for i in range(len(res_masks)-1):
                        mask_temp = torch.add(mask_temp,res_masks[i+1])
                    m_masks.append(mask_temp)
                
        if Merge_mask:
            res_masks = torch.cat(m_masks, dim=0)
        else :
            res_masks = torch.cat(res_masks, dim=0)
        if invert_mask:
            res_masks = 1 - res_masks
        return (res_masks,res_labels,)

NODE_CLASS_MAPPINGS = {
    "ApplyEasyOCR_batch": ApplyEasyOCR_batch,
    "load_EasyOCR_model": load_EasyOCR_model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "load_EasyOCR_model": "load_EasyOCR_model",
    "ApplyEasyOCR_batch": "ApplyEasyOCR_batch",
}
