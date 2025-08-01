import torch
import numpy as np
import random
import os
from PIL import Image, ImageOps
from pathlib import Path

import folder_paths 
from comfy.utils import ProgressBar
import comfy.model_management as mm

from ..moduel.str_edit import str_edit
from ..moduel.custom_class import any
from ..moduel.image_utils import pil_to_mask


# ------------------GetData nodes------------------
CATEGORY_NAME = "WJNode/GetData"

class Mask_Detection:
    DESCRIPTION = """
    中文/CH
    说明：
        输入mask或image,使用去重检测遮罩属性
        若mask或image都输入，默认检测image
    输入：
        1:Exist_threshold判断遮罩是否存在的阈值，
    输出：
        1:遮罩是否存在(非纯色)
        2:是否为硬边缘(二值)
        3:是否为全白遮罩
        4:是否为全黑遮罩
        5:是否为灰度遮罩
        6:输出色值(当mask不为单色时输出0)
    EN/英文
    Description:
        Input mask or image, use unique detection to check the mask attributes
        If both mask and image are input, the default is to detect image
    Input:
        1:Exist_threshold, the threshold for determining if the mask exists
    Output:
        1:Whether the mask exists (not pure color)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "accuracy": ("INT", {"default": 255, "min": 2, "max": 255, 
                                        "step": 1, "display": "slider"}),
                "Exist_threshold": ("INT", {"default": 3, "min": 2, "max": 255, 
                                        "step": 1, "display": "slider"}),
                },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN",
                    "BOOLEAN", "BOOLEAN", "int")
    RETURN_NAMES = ("Exist?", "HardEdge", "PureWhite",
                    "PureBlack", "PureGray", "PureColorValue")
    FUNCTION = "MaskDetection"

    def MaskDetection(self, accuracy, Exist_threshold, image=None, mask=None):
        #初始化值
        Exist = False
        binary = False
        PureWhite = False
        PureBlack = False
        PureGray = False
        PureColorValue = float(0)

        #检测输入
        if image is None:
            if mask is None:
                print("Warning: Image input is empty!")
                return (Exist, binary, PureWhite, PureBlack, PureGray, PureColorValue)
            else:
                image = mask

        #统计不同像素值的数量
        image = (image*accuracy).to(torch.uint8)
        data = torch.unique(image).tolist() #去重
        n = len(data)

        if n == 1: #只有一个值，是纯色不是遮罩
            PureColorValue = data[0]
            PureGray = True
            if data[0] == 0:
                PureBlack = True
            elif data[0] == 1:
                PureWhite = True
        elif n == 2: #有2个值，是二值遮罩
            Exist = True
            binary = True
        elif n <=Exist_threshold: #小于自定义个值，是遮罩
            Exist = True

        return (Exist, binary, PureWhite, PureBlack, PureGray, PureColorValue)


class get_image_data:
    DESCRIPTION = """
    EN/English:
    Obtain image size data including batch count, height, width, channels and shape information.
    If both image and mask are input at the same time, image data will be prioritized for output.

    Output description:
        "images","masks": Input images and masks
        "N","H","W","C": Batch count/Height/Width/Channels of the image
        "shape": Complete array information of count/size/channels

    中文/Chinese:
    获取图像尺寸数据
        若同时输入image和mask,会优先输出image数据
    输出说明：
        "images","masks"：输入的图像和遮罩
        "N","H","W","C"：图像的批次/高度/宽度/通道数
        "shape"：图像的完整数量/尺寸/通道数组信息
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "images":("IMAGE",),
                "masks":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","INT","INT","INT","INT","LIST")
    RETURN_NAMES = ("images","masks","N","H","W","C","shape")
    FUNCTION = "get_size_data"

    def get_size_data(self, images = None, masks = None):
        # Initialize default values
        shape = [0, 0, 0, 0]  # [N, H, W, C]
        N = H = W = C = 0

        # Determine which data to use (prioritize images over masks)
        if images is not None:
            shape = list(images.shape)  # [N, H, W, C]
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        elif masks is not None:
            if len(masks.shape) == 3:  # [N, H, W]
                shape = list(masks.shape) + [1]  # Add channel dimension
                N, H, W, C = shape[0], shape[1], shape[2], 1
            else:  # [N, H, W, C] if masks already have channel dimension
                shape = list(masks.shape)
                N, H, W, C = shape[0], shape[1], shape[2], shape[3]

        return (images, masks, N, H, W, C, shape)

class get_image_ratio:
    DESCRIPTION = """
    EN/English:
    Obtain image aspect ratio data including maximum/minimum dimensions, ratio values and ratio classification.
    If both image and mask are input at the same time, image data will be prioritized for output.

    Output description:
        "images","masks": Input images and masks
        "max_HW","min_HW": Maximum and minimum values of height/width
        "ratio_float": Actual aspect ratio as float
        "ratio_str": Nearest integer ratio in string format (e.g., "16:9", "4:3")
        "ratio_class": Ratio type classification (0=square, 1=wide, 2=tall)

    中文/Chinese:
    获取图像比例数据
        若同时输入image和mask,会优先输出image数据
    输出说明：
        "images","masks"：输入的图像和遮罩
        "max_HW","min_HW"：尺寸最值
        "ratio_float"：尺寸实际比例
        "ratio_str"：最接近的整数比例字符串形式
        "ratio_class"：比例类型(0方形，1宽图，2长图)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "images":("IMAGE",),
                "masks":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","INT","INT","FLOAT","STRING","INT")
    RETURN_NAMES = ("images","masks","max_HW","min_HW","ratio_float","ratio_str","ratio_class")
    FUNCTION = "get_ratio_data"

    def get_ratio_data(self, images = None, masks = None):
        # Initialize default values
        max_HW = 0
        min_HW = 0
        ratio_float = 1.0
        ratio_str = "1:1"
        ratio_class = 0  # 0=square, 1=wide, 2=tall

        # Determine which data to use (prioritize images over masks)
        data_tensor = None
        shape = []

        if images is not None:
            data_tensor = images
            shape = list(images.shape)  # [N, H, W, C]
        elif masks is not None:
            data_tensor = masks
            shape = list(masks.shape)  # [N, H, W] or [N, H, W, C]

        # Calculate dimensions if we have data
        if data_tensor is not None and len(shape) >= 3:
            H, W = shape[1], shape[2]

            # Calculate max and min dimensions
            max_HW = max(H, W)
            min_HW = min(H, W)

            # Calculate aspect ratio
            if min_HW > 0:
                ratio_float = max_HW / min_HW

                # Determine ratio class
                if abs(ratio_float - 1.0) < 0.1:  # Nearly square (within 10% tolerance)
                    ratio_class = 0  # Square
                elif W > H:
                    ratio_class = 1  # Wide
                else:
                    ratio_class = 2  # Tall

                # Calculate nearest integer ratio string
                ratio_str = self._calculate_ratio_string(W, H)

        return (images, masks, max_HW, min_HW, ratio_float, ratio_str, ratio_class)

    def _calculate_ratio_string(self, width, height):
        """Calculate the nearest integer ratio string like '16:9', '4:3', etc."""

        # Find GCD to simplify the ratio
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        # Handle edge cases
        if width == 0 or height == 0:
            return "1:1"

        # Calculate GCD and simplify ratio
        common_divisor = gcd(width, height)
        simplified_w = width // common_divisor
        simplified_h = height // common_divisor

        # If the simplified ratio is too large, find a close approximation
        if simplified_w > 50 or simplified_h > 50:
            # Common aspect ratios to check against
            common_ratios = [
                (1, 1), (4, 3), (3, 2), (16, 10), (5, 3), (16, 9),
                (21, 9), (2, 1), (3, 1), (4, 1), (5, 1)
            ]

            target_ratio = width / height
            best_match = (1, 1)
            best_diff = float('inf')

            for w_ratio, h_ratio in common_ratios:
                ratio = w_ratio / h_ratio
                diff = abs(ratio - target_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_match = (w_ratio, h_ratio)

                # Also check the inverse ratio
                ratio = h_ratio / w_ratio
                diff = abs(ratio - target_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_match = (h_ratio, w_ratio)

            simplified_w, simplified_h = best_match

        return f"{simplified_w}:{simplified_h}"
    


CATEGORY_NAME = "WJNode/Other-node"


class Any_Pipe: # 任意数据打组
    DESCRIPTION = """
    中文/CH
    将任意数据打包成一个列表以减少面条
    自带拆分数据，可嵌套

    EN/英文
    Any data grouping
    Pack any data into a list to reduce noodles
    It has data splitting, and can be nested
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "data_array": ("LIST",),
                "data_1": (any,),
                "data_2": (any,),
                "data_3": (any,),
                "data_4": (any,),
                "data_5": (any,),
                "data_6": (any,),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST", any, any, any, any, any, any,)
    RETURN_NAMES = ("data_array", "data_1", "data_2",
                    "data_3", "data_4", "data_5", "data_6",)
    FUNCTION = "any_data_array"

    def any_data_array(self, data_array=None,**kwargs):
        #初始化输入值
        keys = ["data_1","data_2","data_3","data_4","data_5","data_6"]
        input_data = [None for i in keys]
        if kwargs != {}:
            i = 0
            for k,v in kwargs.items():
                if k in keys: input_data[i] = v
                else: input_data[i] = None
                i+=1

        #刷新输出
        if data_array is None: return (input_data, *input_data)
        else:
            output_data = []
            for i in range(len(keys)):
                if input_data[i] == None: output_data.append(data_array[i])
                else: output_data.append(input_data[i])
            return (output_data, *output_data)


class Folder_link: # 创建符号链接 ******************开发中
    """
    创建符号链接
    """
    nodes_list = os.listdir(os.path.join(folder_paths.base_path,"custom_nodes"))
    @classmethod
    def INPUT_TYPES(s):
        nodes_list_name = s.nodes_list.insert(0,"** Do not map **")
        nodes_list = nodes_list.insert(1,"** All map !**")
        return {
            "required": {
                        "path_class":(["autodl-tmp","autodl-fs","Win-Parent", "Win-Ancestors"],),
                        "Check_link":("BOOLEAN",{"default":True}),
                        "FileConflict":(["Ignore", "REname", "Move(overwrite)", "Move(not overwrite)", "Delete"],{"default":"Ignore"}),
                        "link_models":("BOOLEAN",{"default":True}),
                        "link_aux":("BOOLEAN",{"default":True}),
                        "custom_nodes":(nodes_list_name,{"default":nodes_list_name[0]}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = (None,)
    FUNCTION = "Folder_link"
    def Folder_link(self,path_class,Check_link,FileConflict,link_models,link_aux,custom_nodes):
        path_class = "./"
        if path_class == "autodl-tmp" or path_class == "autodl-fs":
            path = os.path.join("root",path_class)
        elif path_class == "Win-Parent":
            path = str(Path(folder_paths.base_path).parent)
        elif path_class == "Win-Ancestors":
            path = str(Path(folder_paths.base_path).parent.parent)

        path_model = os.path.join(path,"models")
        path_custom = os.path.join(folder_paths.base_path,"custom_nodes")
        path_aux = os.path.join(path_custom, "comfyui/controlnet_aux", "ckpts")
        
        path_model_link = folder_paths.models_dir
        path_custom_link = folder_paths.folder_names_and_paths["custom_nodes"]
        path_aux_link = os.path.join(path_custom_link, "comfyui/controlnet_aux", "ckpts")


        if link_models:
            self.run_path_link(path_model,path_model_link,Check_link,FileConflict)
        if link_aux:
            self.run_path_link(path_aux,path_aux_link,Check_link,FileConflict)
        if custom_nodes == "** Do not map **":
            pass
        elif custom_nodes == "** All map !**":
            ...
        else:
            ...
        return(None,)
    
    def run_path_link(path,path_link,Check_link,FileConflict):
        p = Path(path)
        p_link = Path(path_link)
        if Check_link:
            if p_link.is_symlink(): #link存在
                if not p_link.readlink().exists(): #link失效则重建
                    p_link.unlink()
                    if p.exists(): p.mkdir(parents=True) #路径不存在则创建
                    p_link.symlink_to(p)
                    print(f"原链接 {path_link} 已失效，\n更新为 {path}")
                else:
                    print(f"链接 {path_link} --> \n{path} 正常,已跳过")
            else: #p_link不是符号链接
                if p_link.exists():
                    if FileConflict == "Ignore":
                        pass
                    elif FileConflict == "REname":
                        p_link.replace(p_link.parts[-1])
                    elif FileConflict == "Move(overwrite)":
                        p_link.rename(p)
                    elif FileConflict == "Move(not overwrite)":
                        ...
                    elif FileConflict == "Delete":
                        ...
                if p.exists(): p.mkdir(parents=True) #路径不存在则创建
                p_link.symlink_to(p)


class Determine_Type: # 判断输入的数据类型是否和所选类型匹配
    DESCRIPTION = """
    中文/CH
    判断输入的数据类型是否和所选类型匹配并输出数据类型
    类型含图像、张量、模型、采样器、基础类型：
        图像类型含255、4通道、3通道、批次图像、单张图像
        张量类型含bool、int、float
        模型类型含unet、vae、clip
        采样器类型含conditioning、latent、latent_noise_mask
        基础类型含None、int、float、string、complex、tuple、list、set、dict、Tensor
    EN/英文
    Determine if the input data type matches the selected type and output the data type
    The type includes image, tensor, model, sampler, and basic type:
        The image type includes 255, 4 channels, 3 channels, batch images, and single images
        The tensor type includes bool, int, float
        The model type includes unet, vae, clip
        The sampler type includes conditioning, latent, latent_noise_mask
        The basic type includes None, int, float, string, complex, tuple, list, set, dict, Tensor
    """
    type_dict = {
        "comfyui/image": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/image255": ["torch.ByteTensor","torch.CharTensor","torch.ShortTensor"],
        "comfyui/image4channels": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/image3channels": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/image_bath": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/image_Single": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/mask": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/mask255": ["torch.ByteTensor","torch.CharTensor","torch.ShortTensor"],
        "comfyui/mask_bath": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "comfyui/mask_Single": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor"],
        "value/None": None,
        "value/int": int,"value/float": float,"value/string": str,"value/complex": complex,
        "value/tuple":tuple,"value/list": list,"value/set":set,"value/dict": dict,"Tensor/Tensor": torch.Tensor,
        "Tensor/torch.bool": ["torch.bool",],
        "Tensor/torch.int": ["torch.ByteTensor","torch.CharTensor","torch.ShortTensor","torch.IntTensor","torch.LongTensor"],
        "Tensor/torch.float": ["torch.HalfTensor","torch.BFloat16Tensor","torch.FloatTensor","torch.DoubleTensor"],
        "model/unet":"ModelPatcher", "model/vae":"VAE", "model/clip":"CLIP",
        "sampler/conditioning":"list", "sampler/latent":"dict", "sampler/latent_noise_mask":"dict"
    }
    type_class = {
        "comfyui":["comfyui/image","comfyui/image255","comfyui/mask","comfyui/mask255"],
        "value":["value/None","Tensor/Tensor","value/int","value/float","value/string","value/complex",
                 "value/tuple","value/list","value/set","value/dict"],
        "Tensor":["Tensor/torch.bool","Tensor/torch.int","Tensor/torch.float"],
        "numpy":["np.array",],
    }

    @classmethod
    def INPUT_TYPES(s):
        type_list = list(s.type_dict.keys())
        return {
            "required": {
                "type_name": (type_list, {"default": type_list[0]}),
                #"subobject": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "data": (any,),
            }
        }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = (any,"BOOLEAN", "STRING")
    RETURN_NAMES = ("data","is select type", "type name")
    FUNCTION = "select_type"

    def select_type(self, type_name, data=None):
        target_types = self.__class__.type_dict[type_name]
        data_type = data.__class__.__name__
        is_select_type = False
        if target_types is None: #空值
            if data is None: is_select_type = True
        elif "comfyui" in type_name: #图像
            if data_type == "Tensor":
                data_type = str(data.type())
            if data_type in target_types:
                if type_name == "comfyui/image":
                    if data.dim() == 4 : is_select_type = True
                elif type_name == "comfyui/image255":
                    if data_type in target_types: is_select_type = True
                elif type_name == "comfyui/image4channels":
                    if data.dim() == 4 : 
                        if data.shape[-1] == 4: is_select_type = True
                elif type_name == "comfyui/image3channels":
                    if data.dim() == 4 : 
                        if data.shape[-1] == 3: is_select_type = True
                elif type_name == "comfyui/image_bath":
                    if data.shape[0] > 1 : is_select_type = True
                elif type_name == "comfyui/image_Single":
                    if data.shape[0] == 1 : is_select_type = True
                elif type_name == "comfyui/mask":
                    if data.dim() == 3 : is_select_type = True
                elif type_name == "comfyui/mask255":
                    if data_type in target_types: is_select_type = True
                elif type_name == "comfyui/mask_bath":
                    if data.shape[0] > 1 : is_select_type = True
                elif type_name == "comfyui/mask_Single":
                    if data.shape[0] == 1 : is_select_type = True
        elif type_name in self.__class__.type_class["value"]: #基础
            if "value/" + data_type == type_name: is_select_type = True
            if "Tensor/" + data_type == type_name: is_select_type = True
        elif type_name in self.__class__.type_class["Tensor"] and data_type=="Tensor": #张量
            data_type = str(data.type())
            if data_type in target_types: is_select_type = True
        elif "model" in type_name: #模型
            if data_type == target_types: is_select_type = True
        elif "sampler" in type_name: #采样
            if data_type == target_types:
                if type_name == "sampler/conditioning":
                    if data[0][0].__class__.__name__ == "Tensor": is_select_type = True
                if type_name == "sampler/latent":
                    if "samples" in list(data.keys()) : is_select_type = True
                if type_name == "sampler/latent_noise_mask":
                    if "noise_mask" in list(data.keys()) : is_select_type = True
        return (data,is_select_type, data_type)


CATEGORY_NAME = "WJNode/Other-plugins/WAS"


class WAS_Mask_Fill_Region_batch:
    DESCRIPTION = """
    Original plugin: was-node-suite-comfyui
    Original node: WAS_Mask_Fill_Region
    change: batch bug in mask processing
    原始节点：WAS_Mask_Fill_Region
    更改：支持批次
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "mask": ("MASK",),
                        "invert_mask":("BOOLEAN",{"default":False}),
                    }
                }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "fill_region"
    def fill_region(self, mask, invert_mask):
        n = mask.shape[0]
        mask_output = torch.zeros((0,*mask.shape[1:]), dtype=torch.float)
        if n != 1:
            for i in range(n):
                mask_temp = mask[i].repeat(1,1,1)
                mask_temp = self.fill_run(mask_temp)[0]
                mask_output = torch.cat((mask_output, mask_temp), dim=0)
        else:
            mask_output = self.fill_run(mask)[0]

        if invert_mask:
            mask_output = 1.0 - mask_output
        return (mask_output,)
        
    def fill_run(self, mask):
        mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(mask_np, mode="L")
        region_mask = self.fill_1(pil_image)
        return pil_to_mask(region_mask).unsqueeze(0).unsqueeze(1)

    def fill_1(self, image):
        from scipy.ndimage import binary_fill_holes
        image = image.convert("L")
        binary_mask = np.array(image) > 0
        filled_mask = binary_fill_holes(binary_mask)
        filled_image = Image.fromarray(filled_mask.astype(np.uint8) * 255, mode="L")
        return ImageOps.invert(filled_image.convert("RGB"))


CATEGORY_NAME = "WJNode/Other-plugins/ImpackPack"


class SegmDetectorCombined_batch:
    DESCRIPTION = """
    Original plugin: Impact-Pack
    Original node: SegmDetectorCombined
    change 1: batch detection of masks
    Change 2: Supports both modes simultaneously
    原始插件：Impact-Pack
    原始节点：SegmDetectorCombined
    更改1：支持批次
    更改2：支持两个模式
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                    },
                "optional": {
                    "bbox_detector": ("BBOX_DETECTOR", ),
                    "segm_detector": ("SEGM_DETECTOR", ),
                    }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY_NAME

    def doit(self, image, threshold, dilation, segm_detector=None, bbox_detector=None):
        #图像预处理
        if image.dim() == 3:
            image = image.unsqueeze(0)
        mask = torch.zeros((0,*image.shape[1:-1]), dtype=torch.float, device="cpu")
        mask_0 = torch.zeros((1,*image.shape[1:-1]), dtype=torch.float32, device="cpu")

        #检测器类型
        seg_class = segm_detector
        if segm_detector is None:
            seg_class = bbox_detector
        else:
            print("Error: No detector selected, Return empty mask !")
            return(mask_0.unsqueeze(0),)

        #运行检测
        for i in range(image.shape[0]):
            mask_temp = seg_class.detect_combined(image[i].unsqueeze(0), threshold, dilation)
            if mask_temp is None:
                mask_temp = mask_0
            else:
                mask_temp = mask_temp.unsqueeze(0)
            mask = torch.cat((mask, mask_temp), dim=0)
        mask = mask.squeeze(-1)
        return (mask,)


class bbox_restore_mask:
    DESCRIPTION = """
    Original plugin: impack-pack
    crop_region:Restore cropped image (SEG editing)
    crop_region：恢复裁剪后的图像（SEG编辑）
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image":("IMAGE",),
                "mask":("MASK",),
                "crop_region": ("SEG_ELT_crop_region",),#SEG_ELT_crop_region , SEG_ELT_bbox
                "fill_color":("INT",{"default":0,"min":0,"max":255,"step":1,"display":"slider"}),
            },
            "optional": {}
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "restore_mask"
    def restore_mask(self, reference_image, mask, crop_region, fill_color):
        fill_color = fill_color/255.0
        bath_bbox = not isinstance(crop_region[0], int)
        bath_mask = mask.shape[0] != 1
        h,w = reference_image.shape[1], reference_image.shape[2]

        if not bath_bbox:
            x1,y1,x2,y2 = crop_region
            mask = torch.nn.functional.pad(mask, (x1,w-x2,y1,h-y2), "constant", fill_color)
        elif bath_bbox and bath_mask:
            if len(crop_region) == mask.shape[0]:
                for i in range(len(crop_region)):
                    x1,y1,x2,y2 = crop_region[i]
                    mask[i] = torch.nn.functional.pad(mask[i], (x1,w-x2,y1,h-y2), "constant", fill_color)
            else:
                print("Error-bbox_restore_mask: The number of crop_region does not match the number of masks")
        else:
            print("Error-bbox_restore_mask: There are multiple crop_region quantities and one mask quantity, which cannot be matched") 
        return (mask,)


class run_yolo_bboxs:
    DESCRIPTION = """
    使用YOLO模型检测图像序列中的目标，并输出边界框详细信息
    功能：运行yolo模型输入图像序列输出bboxs支持批量处理图像
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "bbox_detector": ("BBOX_DETECTOR", ),
                    "input_size": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1}),
                    }
                }
    RETURN_TYPES = ("bboxs",)
    RETURN_NAMES = ("bboxs",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY_NAME
    def doit(self, image, threshold, bbox_detector, input_size):
        from .yolo_utils import subcore
        import torchvision.transforms as transforms
        if image.dim() == 3: # 图像预处理
            image = image.unsqueeze(0)
        all_bboxs = [] # 存储所有图像的边界框结果
        for i in range(image.shape[0]): # 对每张图像进行处理
            img = image[i].unsqueeze(0) # 获取单张图像
            # 计算缩放比例
            scale_factor = 1.0
            h, w = img.shape[1], img.shape[2]
            max_dim = max(h, w)
            # 如果input_size大于0且原图长边大于max_size，则进行缩放
            if input_size > 0 and max_dim > input_size:
                scale_factor = input_size / max_dim
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                # 确保图像格式正确，先转换为PIL图像再调整大小
                pil_img = subcore.utils.tensor2pil(img)
                pil_img_resized = pil_img.resize((new_w, new_h))
                # 不需要再次转换回tensor，因为inference_bbox函数接受PIL图像
                img_resized = pil_img_resized
            else:
                # 直接转换为PIL图像
                img_resized = subcore.utils.tensor2pil(img)
            # 使用缩放后的图像进行检测
            detected_results = subcore.inference_bbox(bbox_detector.bbox_model, 
                                                     img_resized, 
                                                     threshold)
            image_bboxs = [] # 提取边界框信息
            if len(detected_results[1]) > 0:  # 如果检测到目标
                for j in range(len(detected_results[1])):
                    # 获取边界框坐标并根据缩放比例调整回原始尺寸
                    bbox = detected_results[1][j].tolist()
                    if scale_factor != 1.0:
                        # 将边界框坐标按比例放大回原始尺寸 [y1, x1, y2, x2]
                        bbox = [int(coord / scale_factor) for coord in bbox]
                        
                    bbox_info = { # 创建边界框信息字典
                        "label": detected_results[0][j],  # 类别标签
                        "bbox": bbox,  # 边界框坐标 [y1, x1, y2, x2]
                        "confidence": float(detected_results[3][j])  # 置信度
                    }
                    image_bboxs.append(bbox_info)
            all_bboxs.append(image_bboxs) # 将当前图像的边界框结果添加到总结果中
        return (all_bboxs,)


class run_yolo_bboxs_v2:
    DESCRIPTION = """
    使用YOLO模型检测图像序列中的目标，并输出边界框详细信息
    功能：运行yolo模型输入图像序列输出bboxs支持批量处理图像
    支持选择设备(CPU/GPU/CPU+GPU)以提高处理效率
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "bbox_detector": ("BBOX_DETECTOR", ),
                    "input_size": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1}),
                    "device": (["auto", "cpu", "cuda", "mps"], {"default": "auto"}),
                    "batch_size": ("INT", {"default": 4, "min": 1, "max": 1024, "step": 1}),
                    }
                }
    RETURN_TYPES = ("bboxs",)
    RETURN_NAMES = ("bboxs",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY_NAME
    def doit(self, image, threshold, bbox_detector, input_size, device="auto", batch_size=1):
        from .yolo_utils import subcore
        import torchvision.transforms as transforms
        import torch
        import math
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from contextlib import nullcontext
        
        # 添加安全的全局变量到 torch.serialization
        # 解决 PyTorch 2.6 中 weights_only=True 的兼容性问题
        try:
            import torch.serialization
            # 添加 Sequential 和其他常见模块为安全全局变量
            torch.serialization.add_safe_globals([
                torch.nn.modules.container.Sequential,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.pooling.MaxPool2d,
                torch.nn.Module
            ])
        except (ImportError, AttributeError):
            # 如果是较早版本的 PyTorch，add_safe_globals 可能不存在
            pass
        
        # 确定使用的设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"使用设备: {device} 进行YOLO检测")
        
        if image.dim() == 3: # 图像预处理
            image = image.unsqueeze(0)
        
        total_images = image.shape[0]
        all_bboxs = [[] for _ in range(total_images)]  # 预分配结果列表，默认为空列表
        
        # 修补 inference_bbox 函数，以正确处理 PyTorch 2.6 的兼容性问题
        def patched_inference_bbox(model, image, confidence, device=None):
            # 使用上下文管理器临时修改 torch.load 行为
            with torch.serialization.safe_globals([
                torch.nn.modules.container.Sequential,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.pooling.MaxPool2d,
                torch.nn.Module
            ]) if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals') else nullcontext():
                try:
                    return subcore.inference_bbox(model, image, confidence, device=device)
                except Exception as e:
                    print(f"YOLO 检测错误: {str(e)}")
                    # 返回一个空的检测结果结构
                    return [], [], [], []
        
        # 批处理函数
        def process_batch(batch_indices):
            batch_results = []
            batch_images = []
            scale_factors = []
            original_shapes = []
            
            # 第一步：预处理所有图像，在循环外进行缩放计算
            for idx in batch_indices:
                img = image[idx].unsqueeze(0)  # 获取单张图像
                h, w = img.shape[1], img.shape[2]
                original_shapes.append((h, w))
                max_dim = max(h, w)
                
                # 计算缩放比例
                scale_factor = 1.0
                if input_size > 0 and max_dim > input_size:
                    scale_factor = input_size / max_dim
                
                scale_factors.append(scale_factor)
            
            # 第二步：批量转换为PIL图像并进行缩放
            for i, idx in enumerate(batch_indices):
                img = image[idx].unsqueeze(0)
                scale_factor = scale_factors[i]
                
                if scale_factor != 1.0:
                    h, w = original_shapes[i]
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    # 转换为PIL图像并调整大小
                    pil_img = subcore.utils.tensor2pil(img)
                    pil_img_resized = pil_img.resize((new_w, new_h))
                    batch_images.append(pil_img_resized)
                else:
                    # 直接转换为PIL图像
                    batch_images.append(subcore.utils.tensor2pil(img))
            
            # 第三步：对每个图像进行检测
            for i, idx in enumerate(batch_indices):
                try:
                    img_resized = batch_images[i]
                    scale_factor = scale_factors[i]
                    
                    # 使用修补后的函数进行检测
                    detected_results = patched_inference_bbox(
                        bbox_detector.bbox_model, 
                        img_resized, 
                        threshold,
                        device=device
                    )
                    
                    image_bboxs = []  # 提取边界框信息
                    if detected_results and len(detected_results) >= 4 and len(detected_results[1]) > 0:  # 确保检测结果有效
                        for j in range(len(detected_results[1])):
                            # 获取边界框坐标并根据缩放比例调整回原始尺寸
                            bbox = detected_results[1][j].tolist()
                            if scale_factor != 1.0:
                                # 将边界框坐标按比例放大回原始尺寸 [y1, x1, y2, x2]
                                bbox = [int(coord / scale_factor) for coord in bbox]
                                
                            bbox_info = {  # 创建边界框信息字典
                                "label": detected_results[0][j],  # 类别标签
                                "bbox": bbox,  # 边界框坐标 [y1, x1, y2, x2]
                                "confidence": float(detected_results[3][j])  # 置信度
                            }
                            image_bboxs.append(bbox_info)
                    
                    batch_results.append((idx, image_bboxs))
                except Exception as e:
                    print(f"处理图像 {idx} 时出错: {str(e)}")
                    batch_results.append((idx, []))  # 返回空的检测结果
            
            return batch_results
        
        # 根据批处理大小划分任务
        if device == "cpu" and batch_size > 1:
            # 使用多线程处理CPU任务
            num_workers = min(batch_size, torch.get_num_threads())
            batches = []
            for i in range(0, total_images, batch_size):
                batches.append(list(range(i, min(i + batch_size, total_images))))
            
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_batch, batch) for batch in batches]
                    for future in as_completed(futures):
                        try:
                            for idx, result in future.result():
                                all_bboxs[idx] = result
                        except Exception as e:
                            print(f"处理批次结果时出错: {str(e)}")
            except Exception as e:
                print(f"多线程处理时出错: {str(e)}")
                # 回退到单线程处理
                for i in range(total_images):
                    try:
                        results = process_batch([i])
                        if results and len(results) > 0:
                            _, result = results[0]
                            all_bboxs[i] = result
                    except Exception as ex:
                        print(f"回退处理图像 {i} 时出错: {str(ex)}")
        else:
            # GPU模式或单线程CPU模式
            for i in range(total_images):
                try:
                    results = process_batch([i])
                    if results and len(results) > 0:
                        _, result = results[0]
                        all_bboxs[i] = result
                except Exception as e:
                    print(f"处理图像 {i} 时出错: {str(e)}")
        
        return (all_bboxs,)


CATEGORY_NAME = "WJNode/Other-plugins/SAM2"


class Sam2AutoSegmentation_data:
    DESCRIPTION = """
    Original plugin: ComfyUI-segment-anything-2 
    Original node: Sam2AutoSegmentation
    change: add data output
    purpose: Get coordinates/Get object mask
    更改：增加数据输出
    目的：获取坐标/获取对象掩码
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", ),
                "image": ("IMAGE", ),
                "points_per_side": ("INT", {"default": 32}),
                "points_per_batch": ("INT", {"default": 64}),
                "pred_iou_thresh": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stability_score_thresh": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stability_score_offset": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_n_layers": ("INT", {"default": 0}),
                "box_nms_thresh": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_nms_thresh": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_overlap_ratio": ("FLOAT", {"default": 0.34, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_n_points_downscale_factor": ("INT", {"default": 1}),
                "min_mask_region_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_m2m": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "BBOX","LIST","LIST")
    RETURN_NAMES =("mask", "segmented_image", "bbox" ,"Color_list","point_coords")
    FUNCTION = "segment"
    CATEGORY = CATEGORY_NAME

    def segment(self, image, sam2_model, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, 
                stability_score_offset, crop_n_layers, box_nms_thresh, crop_n_points_downscale_factor, min_mask_region_area, 
                use_m2m, mask_threshold, crop_nms_thresh, crop_overlap_ratio, keep_model_loaded):
        offload_device = mm.unet_offload_device()
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model["segmentor"]
        
        if segmentor != 'automaskgenerator':
            raise ValueError("Loaded model is not SAM2AutomaticMaskGenerator")
        
        model.points_per_side=points_per_side
        model.points_per_batch=points_per_batch
        model.pred_iou_thresh=pred_iou_thresh
        model.stability_score_thresh=stability_score_thresh
        model.stability_score_offset=stability_score_offset
        model.crop_n_layers=crop_n_layers
        model.box_nms_thresh=box_nms_thresh
        model.crop_n_points_downscale_factor=crop_n_points_downscale_factor
        model.crop_nms_thresh=crop_nms_thresh
        model.crop_overlap_ratio=crop_overlap_ratio
        model.min_mask_region_area=min_mask_region_area
        model.use_m2m=use_m2m
        model.mask_threshold=mask_threshold
        
        model.predictor.model.to(device)
        
        B, H, W, C = image.shape
        image_np = (image.contiguous() * 255).byte().numpy()

        out_list = []
        segment_out_list = []
        mask_list=[]
        color_list = []
        point_coords = []
        
        pbar = ProgressBar(B)
        autocast_condition = not mm.is_device_mps(device)
        
        
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for img_np in image_np:
                result_dict = model.generate(img_np)
                mask_list = [item['segmentation'] for item in result_dict]
                bbox_list = [item['bbox'] for item in result_dict]
                point_coords = [item['point_coords'] for item in result_dict]

                # Generate random colors for each mask
                num_masks = len(mask_list)
                colors = [tuple(random.choices(range(256), k=3)) for _ in range(num_masks)]
                color_list.append(colors)
                
                # Create a blank image to overlay masks
                overlay_image = np.zeros((H, W, 3), dtype=np.uint8)

                # Create a combined mask initialized to zeros
                combined_mask = np.zeros((H, W), dtype=np.uint8)

                # Iterate through masks and color them
                for mask, color in zip(mask_list, colors):

                    # Combine masks using logical OR
                    combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                    
                    # Convert mask to numpy array
                    mask_np = mask.astype(np.uint8)
                    
                    # Color the mask
                    colored_mask = np.zeros_like(overlay_image)
                    for i in range(3):  # Apply color channel-wise
                        colored_mask[:, :, i] = mask_np * color[i]
                    
                    # Blend the colored mask with the overlay image
                    overlay_image = np.where(colored_mask > 0, colored_mask, overlay_image)
                out_list.append(torch.from_numpy(combined_mask))
                segment_out_list.append(overlay_image)
                pbar.update(1)

        stacked_array = np.stack(segment_out_list, axis=0)
        segment_image_tensor = torch.from_numpy(stacked_array).float() / 255

        if not keep_model_loaded:
           model.predictor.model.to(offload_device)
        
        mask_tensor = torch.stack(out_list, dim=0)
        return (mask_tensor.cpu().float(), segment_image_tensor.cpu().float(), bbox_list, color_list, point_coords)


NODE_CLASS_MAPPINGS = {
    #WJNode/GetData
    "Mask_Detection": Mask_Detection,
    "get_image_data": get_image_data,
    "get_image_ratio": get_image_ratio,
    #WJNode/Other-functions
    "Any_Pipe": Any_Pipe,
    "Determine_Type": Determine_Type,
    #WJNode/Other-plugins
    "WAS_Mask_Fill_Region_batch": WAS_Mask_Fill_Region_batch,
    "SegmDetectorCombined_batch": SegmDetectorCombined_batch,
    "bbox_restore_mask": bbox_restore_mask,
    "Sam2AutoSegmentation_data": Sam2AutoSegmentation_data,
    "run_yolo_bboxs": run_yolo_bboxs,
    "run_yolo_bboxs_v2": run_yolo_bboxs_v2,
}
