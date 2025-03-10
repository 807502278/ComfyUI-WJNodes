import os
import importlib.util

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF

import folder_paths
from ..moduel.image_utils import device_input,device_list


CATEGORY_NAME = "WJNode/Detection"


Models_Dict = {
    # ResNet系列
    "resnet18": {"file": "resnet18-f37072fd.pth", "type": "resnet"},
    "resnet34": {"file": "resnet34-b627a593.pth", "type": "resnet"},
    "resnet50": {"file": "resnet50-11ad3fa6.pth", "type": "resnet"},
    "resnet101": {"file": "resnet101-cd907fc2.pth", "type": "resnet"},
    "resnet152": {"file": "resnet152-f82ba261.pth", "type": "resnet"},
    # ResNeXt系列
    "resnext50_32x4d": {"file": "resnext50_32x4d-1a0047aa.pth", "type": "resnet"},
    "resnext101_32x8d": {"file": "resnext101_32x8d-110c445d.pth", "type": "resnet"},
    "resnext101_64x4d": {"file": "resnext101_64x4d-173b62eb.pth", "type": "resnet"},
    # Wide ResNet系列
    "wide_resnet50_2": {"file": "wide_resnet50_2-9ba9bcbe.pth", "type": "resnet"},
    "wide_resnet101_2": {"file": "wide_resnet101_2-d733dc28.pth", "type": "resnet"},
    # DenseNet系列
    "densenet121": {"file": "densenet121-a639ec97.pth", "type": "densenet"},
    "densenet169": {"file": "densenet169-b2777c0a.pth", "type": "densenet"},
    "densenet201": {"file": "densenet201-c1103571.pth", "type": "densenet"},
    "densenet161": {"file": "densenet161-8d451a50.pth", "type": "densenet"},
    # EfficientNet系列
    "efficientnet_b0": {"file": "efficientnet_b0-3b5e5de7.pth", "type": "efficientnet"},
    "efficientnet_b1": {"file": "efficientnet_b1-c27df63c.pth", "type": "efficientnet"},
    "efficientnet_b2": {"file": "efficientnet_b2-0f7e5a76.pth", "type": "efficientnet"},
    # MobileNet系列
    "mobilenet_v2": {"file": "mobilenet_v2-b0353104.pth", "type": "mobilenet"},
    "mobilenet_v3_small": {"file": "mobilenet_v3_small-047dcff4.pth", "type": "mobilenet"},
    "mobilenet_v3_large": {"file": "mobilenet_v3_large-8738ca79.pth", "type": "mobilenet"},
    # VGG系列
    "vgg11": {"file": "vgg11-8a719046.pth", "type": "vgg"},
    "vgg13": {"file": "vgg13-19584684.pth", "type": "vgg"},
    "vgg16": {"file": "vgg16-397923af.pth", "type": "vgg"},
    "vgg19": {"file": "vgg19-dcbb9e9d.pth", "type": "vgg"},
}


def prepare_image_for_model(image, model_type="resnet"):
    """根据不同模型类型准备图像"""
    # 基本转换
    base_transform = [
        T.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (B, H, W, C) to (B, C, H, W)
        T.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),  # 确保值在0-1之间
    ]
    
    # 根据模型类型添加特定的预处理
    if model_type in ["resnet", "resnext", "wide_resnet", "densenet", "efficientnet"]:
        specific_transform = [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )  # ImageNet标准化
        ]
    elif model_type == "vgg":
        specific_transform = [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    elif model_type == "mobilenet":
        specific_transform = [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:  # 默认处理
        specific_transform = [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    transform = T.Compose(base_transform + specific_transform)
    return transform(image)


class load_torchvision_model:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(Models_Dict.keys())
        download_method = ["LocalOnly", "Download"]
        return {
            "required": {
                    "model_name": (model_list, {"default": model_list[0]}),
                    "download": (download_method, {"default": download_method[0]}),
                    "device": device_input,
                    }
        }
    
    RETURN_TYPES = ("Similarity",)
    FUNCTION = "load"
    CATEGORY = CATEGORY_NAME
    
    def load(self, model_name, download, device):
        # 获取模型信息
        model_info = Models_Dict[model_name]
        model_file = model_info["file"]
        model_type = model_info["type"]
        
        # 设置模型路径
        dir = os.path.join(folder_paths.models_dir, "torchvision", model_type)
        if not os.path.isdir(dir): 
            os.makedirs(dir)
        model_dir = os.path.join(dir, model_file)

        # 下载模型
        if download == "Download" and not os.path.isfile(model_dir):
            print(f"模型未检测到: {model_dir}，开始下载...")
            url = f"https://download.pytorch.org/models/{model_file}"
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            print("下载完成，正在写入磁盘...")
            with open(model_dir, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"写入完成。")

        # 动态导入模块
        print(f"加载模型: {model_dir}")
        module = importlib.import_module("torchvision.models")
        model_class = getattr(module, model_name)
        model = model_class(pretrained=False)

        # 设置模型
        model_device = device_list[device]
        model.to(model_device)
        model.eval()
        
        # 加载预训练权重
        model.load_state_dict(torch.load(model_dir))
        
        # 移除最后的分类层以获取特征向量
        if model_type == "resnet":
            model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_type == "densenet":
            # DenseNet的特征提取
            features = list(model.children())[:-1]
            model = torch.nn.Sequential(*features)
        elif model_type == "efficientnet":
            # EfficientNet的特征提取
            features = list(model.children())[:-1]
            model = torch.nn.Sequential(*features)
        elif model_type == "mobilenet":
            # MobileNet的特征提取
            features = list(model.children())[:-1]
            model = torch.nn.Sequential(*features)
        elif model_type == "vgg":
            # VGG的特征提取
            features = list(model.children())[:-1]
            model = torch.nn.Sequential(*features)
        else:
            # 默认处理
            model = torch.nn.Sequential(*list(model.children())[:-1])
        
        return ((model, model_device, model_type),)


class Run_torchvision_model:
    DESCRIPTION = """
    对比相似度
    支持1对多,多对1,多对多(需批次一样)对比
    输入批次时将返回平均值的对比结果
    支持多种torchvision模型进行特征提取和相似度计算
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "image1": ("IMAGE",),
                    "image2": ("IMAGE",),
                    "Similarity_model": ("Similarity",),
                    "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.01, "step": 0.0001}),
                    }
            }
    
    RETURN_TYPES = ("BOOLEAN", "FLOAT",)
    RETURN_NAMES = ("is_similiar", "cosine_similarity",)
    FUNCTION = "compare_image"
    CATEGORY = CATEGORY_NAME

    def compare_image(self, image1, image2, Similarity_model, threshold):
        model, model_device, model_type = Similarity_model
        val = 0.0
        
        # 根据模型类型准备图像
        image1 = prepare_image_for_model(image1, model_type).to(model_device)
        image2 = prepare_image_for_model(image2, model_type).to(model_device)
        
        with torch.no_grad():
            shape1 = list(image1.shape)
            shape2 = list(image2.shape)
            if shape1[0] == 1:
                if shape2[0] == 1:
                    # 单张图片对比
                    features1 = model(image1).squeeze().flatten().unsqueeze(0)
                    features2 = model(image2).squeeze().flatten().unsqueeze(0)
                    val = torch.nn.functional.cosine_similarity(features1, features2, dim=1).item()
                elif shape2[0] > 1:
                    # 一对多对比
                    val = self.many1(model, image1, image2)
                else: 
                    raise ValueError("错误: Image2 不是标准图像数据!")
            elif shape1[0] > 1:
                if shape2[0] == 1:
                    # 多对一对比
                    val = self.many1(model, image2, image1)
                elif shape2[0] > 1:
                    # 多对多对比
                    if shape2[0] == shape1[0]:
                        val = self.many2(model, image1, image2)
                    else:
                        raise ValueError("错误: 两个输入批次的数量不同!")
                else:
                    raise ValueError("错误: Image1 不是标准图像数据!")

        return (val >= threshold, val)
    
    def many1(self, model, img, img_many):
        """一对多的相似度计算"""
        val = []
        img_features = model(img).squeeze().flatten().unsqueeze(0)
        
        for i in img_many:
            i_features = model(i.unsqueeze(0)).squeeze().flatten().unsqueeze(0)
            v_test = torch.nn.functional.cosine_similarity(img_features, i_features, dim=1).item()
            val.append(v_test)
        
        return sum(val) / len(val)

    def many2(self, model, img1_batch, img2_batch):
        """多对多的相似度计算"""
        val = []
        batch_size = img1_batch.shape[0]
        
        for i in range(batch_size):
            img1_features = model(img1_batch[i].unsqueeze(0)).squeeze().flatten().unsqueeze(0)
            img2_features = model(img2_batch[i].unsqueeze(0)).squeeze().flatten().unsqueeze(0)
            v_test = torch.nn.functional.cosine_similarity(img1_features, img2_features, dim=1).item()
            val.append(v_test)
        
        return sum(val) / len(val)


NODE_CLASS_MAPPINGS = {
    "load_torchvision_model": load_torchvision_model,
    "Run_torchvision_model": Run_torchvision_model,
}