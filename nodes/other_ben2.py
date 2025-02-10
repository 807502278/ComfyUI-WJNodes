import sys
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

import folder_paths
from ..moduel.BEN2 import BEN_Base
from ..moduel.image_utils import tensor_to_pil, pil_to_tensor, device_list

CATEGORY_NAME = "WJNode/Other-plugins"

class load_BEN_model:
    DESCRIPTION = """
    
    """
    @classmethod
    def INPUT_TYPES(s):
        device_ = list(device_list.keys())
        return {
            "required": {
                "model": (["BEN2_Base"],{"default":"BEN2_Base"}),
                "device":(device_,{"default":device_[0]}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("BEN_model",)
    RETURN_NAMES = ("BEN_model",)
    FUNCTION = "load_BEN"
    def load_BEN(self,model,device):
        models = BEN_Base().to(device_list[device]).eval()
        path = os.path.join(folder_paths.models_dir, "rembg", "ben")
        if not os.path.exists(path):
            os.mkdir(path)
        model_path = os.path.join(path, model + ".pth")
        print(f"from {model_path} load model...")
        models.loadcheckpoints(model_path)
        return (models,)

class Run_BEN_v2:
    DESCRIPTION = """
    
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "BEN_model": ("BEN_model",),
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = CATEGORY_NAME

    def process_image(self, BEN_model, input_image):
        # Handle the input tensor format from ComfyUI
        if input_image.dim() != 4:
            input_image = input_image.unsqueeze(0)
        if input_image.shape[3] == 4:
            input_image = input_image[...,0:-1]

        images = []
        for i in input_image:
            i = i.permute(2, 0, 1)
            i = tensor_to_pil(i)
            # Ensure the image is in RGBA mode
            if i.mode != 'RGBA':
                i = i.convert("RGBA")
            # Run inference to get the foreground image
            img = BEN_model.inference(i)
            img = pil_to_tensor(img)
            # Convert to ComfyUI format [B, H, W, C]
            images.append(img.permute(1, 2, 0).unsqueeze(0))
        images = torch.cat(images, dim=0)
        return (images,)

# Export mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    #"load_BEN_model": load_BEN_model,
    #"Run_BEN_v2": Run_BEN_v2
}
