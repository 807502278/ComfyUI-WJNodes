import os
import importlib.util

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF

import folder_paths
from ..moduel.image_utils import device_input,device_list

ResNet_Name = {"resnet18":"resnet18-f37072fd.pth",
              "resnet34":"resnet34-b627a593.pth",
              "resnet50":"resnet50-11ad3fa6.pth",
              "resnet101":"resnet101-cd907fc2.pth",
              "resnet152":"resnet152-f82ba261.pth",
              "resnext50_32x4d":"resnext50_32x4d-1a0047aa.pth",
              "resnext101_32x8d":"resnext101_32x8d-110c445d.pth",
              "resnext101_64x4d":"resnext101_64x4d-173b62eb.pth",
              "wide_resnet50_2":"wide_resnet50_2-9ba9bcbe.pth",
              "wide_resnet101_2":"wide_resnet101_2-d733dc28.pth",
              }

def prepare_image_for_resnet(image):
    transform = T.Compose([
        T.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (B, H, W, C) to (B, C, H, W)
        T.Resize(256),
        T.CenterCrop(224),
        T.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),  # Ensure values are 0-1
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )  # ImageNet normalization
    ])

    return transform(image)

CATEGORY_NAME = "WJNode/Detection"


class load_Similarity:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(ResNet_Name.keys())
        download_method = ["LocalOnly","Download"]
        return {
            "required": {
                    "resnet_model": (model_list,{"default": model_list[2]}),
                    "download":(download_method,{"default": download_method[0]}),
                    "device":device_input,
                    }
        }
    
    RETURN_TYPES = ("Similarity",)
    FUNCTION = "load"
    CATEGORY = CATEGORY_NAME
    def load(self,resnet_model,download,device):
        #设置模型路径
        dir = os.path.join(folder_paths.models_dir, "torchvision", "resnet")
        if not os.path.isdir(dir): os.makedirs(dir)
        model_name = ResNet_Name[resnet_model]
        model_dir = os.path.join(dir,model_name)

        #下载模型
        if download == "Download" and not os.path.isfile(model_dir):
            print(f"Model not detected:{model_dir}Start downloading...")
            url = f"https://download.pytorch.org/models/{model_name}"
            import requests # 发送HTTP请求
            response = requests.get(url, stream=True)
            response.raise_for_status()  #检查请求状态
            print("Download completed, writing to disk...")
            with open(model_dir, 'wb') as f: # 写入文件
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Write completed.")

        #动态导入模块
        print(f"Loading model:{model_dir}")
        module = importlib.import_module("torchvision.models")
        model_class = getattr(module, resnet_model)
        model = model_class(pretrained=False)
        #from torchvision.models import resnet18
        #model = resnet18(weights=ResNet18_Weights.DEFAULT)...
        #https://pytorch.org/vision/stable/models.html

        #设置模型
        model_devic = device_list[device]
        model.to(model_devic)
        model.eval()
        model.load_state_dict(torch.load(model_dir))
        # Remove the final classification layer to get embeddings
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return ((model,model_devic),)


class Run_Similarity:
    DESCRIPTION = """
    对比相似度
    支持1对多,多对1,多对多(需批次一样)对比
    输入批次时将返回平均值的对比结果
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "image1": ("IMAGE",),
                    "image2": ("IMAGE",),
                    "Similarity_model": ("Similarity",),
                    "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.01,"step":0.0001}),
                    }
            }
    
    RETURN_TYPES = ("BOOL", "FLOAT",)
    RETURN_NAMES = ("is_similiar", "cosine_similarity",)
    FUNCTION = "compare_image"
    CATEGORY = CATEGORY_NAME

    def compare_image(self, image1, image2, Similarity_model, threshold):
        model,model_devic = Similarity_model
        val = 0.0
        image1 = prepare_image_for_resnet(image1).to(model_devic)
        image2 = prepare_image_for_resnet(image2).to(model_devic)
        with torch.no_grad():
            shape1 = list(image1.shape)
            shape2 = list(image2.shape)
            if shape1[0] == 1:
                if shape2[0] == 1:
                    val = torch.nn.functional.cosine_similarity(model(image1).squeeze().flatten().unsqueeze(0),
                                                                model(image2).squeeze().flatten().unsqueeze(0),
                                                                dim=1).item()
                elif shape2[0] > 1:
                    val = self.many1(model,image1,image2)
                else: 
                    raise ValueError("Error: Image2 is not standard image data !")
            elif shape1[0] > 1:
                if shape2[0] == 1:
                    val = self.many1(model,image2,image1)
                elif shape2[0] > 1:
                    if shape2[0] == shape1[0]:
                        val = self.many2(model,image1,image2)
                    else:
                        raise ValueError("Error: Two input batches with different quantities !")
                else:
                    raise ValueError("Error: Image1 is not standard image data !")

        return (val >= threshold, val)
    
    def many1(self,model,img,img_many):
        val = []
        for i in img_many:
            v_test = torch.nn.functional.cosine_similarity(model(img).squeeze().flatten().unsqueeze(0),
                                                        model(i.unsqueeze(0)).squeeze().flatten().unsqueeze(0),
                                                        dim=1).item()
            val.append(v_test)
        return sum(val) / len(val)

    def many2(self,model,img,img_many):
        val = []
        is_similiar = []
        for i in range(list(img_many.shape)[0]):
            v_test = torch.nn.functional.cosine_similarity(model(img[i].unsqueeze(0)).squeeze().flatten().unsqueeze(0),
                                                            model(img_many[i].unsqueeze(0)).squeeze().flatten().unsqueeze(0),
                                                            dim=1).item()
            val.append(v_test)
        return sum(val) / len(val)


NODE_CLASS_MAPPINGS = {
    "load_Similarity": load_Similarity,
    "Run_Similarity": Run_Similarity,
}