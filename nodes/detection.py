from transformers import pipeline
from torchvision import transforms
import torch

CATEGORY_NAME_WJnode = "WJNode/Other"

class load_model_value: #加载模型识别特征值的模型 --待开发
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "AdamCodd/vit-base-nsfw-detector"
                }),
            },
            "optional": {
                "device": ("STRING", {
                    "default": "cpu"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY_NAME_WJnode

    def load_model(self, model_name, device):
        model = pipeline("image-classification", model=model_name, device=device)
        return model

class sort_images_batch: #识别图像批次的指定属性值，按属性值重新排序批次 --待开发
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "cuda": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process_images"
    CATEGORY = CATEGORY_NAME_WJnode

    def process_images(self, image, threshold, cuda):
        if cuda:
            device = "cuda"
        else:
            device = "cpu"
        if image.shape[0] == 1:
            predict = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector", device=device) #init pipeline
            result = (predict(transforms.ToPILImage()(image[0].cpu().permute(2, 0, 1)))) #Convert to expected format
            score = next(item['score'] for item in result if item['label'] == 'nsfw')
            output = image
            if(float(score) > threshold):
                output = torch.zeros(1, 512, 512, dtype=torch.float32) #create black image tensor
        else:
            n=[]
            for i in range(image.shape[0]):
                predict = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector", device=device) #init pipeline
                result = (predict(transforms.ToPILImage()(image[i].cpu().permute(2, 0, 1)))) #Convert to expected format
                score = next(item['score'] for item in result if item['label'] == 'nsfw')
                n.append(float(score))

        return (output, str(score))
    
NODE_CLASS_MAPPINGS = {
    #"load_model_value": load_model_value,
    #"sort_images_batch": sort_images_batch,
}