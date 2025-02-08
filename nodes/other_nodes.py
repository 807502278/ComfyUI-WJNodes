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
from ..moduel.str_edit import str_edit

def pil_to_mask(image):  # PIL to Mask
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

class AnyType(str):
    def __init__(self, _):
        self.is_any_type = True

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False
any = AnyType("*")


# ------------------GetData nodes------------------
CATEGORY_NAME = "WJNode/GetData"

class Select_Images_Batch:
    DESCRIPTION = """
        返回指定批次编号处的图像(第1张编号为0,可以任意重复和排列组合)
        超出范围的编号将被忽略，若输入为空则一个都不选，可识别中文逗号。\n
        Return the image at the specified batch number (the first image is numbered 0, and can be arbitrarily repeated and combined) 
        numbers out of range will be ignored, If the input is empty, none will be selected, Chinese commas can be recognized.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "indexes": ("STRING", {"default":"1,2,","multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE","MASK","MASK")
    RETURN_NAMES = ("select_img", "exclude_img","select_mask", "exclude_mask")
    FUNCTION = "SelectImages"

    def SelectImages(self, indexes, images = None, masks = None):
        select_list = np.array(str_edit.tolist_v2(indexes, to_oneDim=True, to_int=True, positive=True))

        #选择图像批次
        if images is not None:
            n_i = images.shape[0]
            s_i = select_list[(select_list >= 1) & (select_list <= n_i)]-1
            if len(s_i) < 1:  # 若输入的编号全部不在范围内则返回原输入
                print("Warning:The input value is out of range, return to the original input.")
                exclude_img, select_img = images, None
            else:
                e_i = np.setdiff1d(np.arange(0, n_i), s_i)  # 排除的图像
                select_img = images[torch.tensor(s_i, dtype=torch.int)]
                exclude_img = images[torch.tensor(e_i, dtype=torch.int)]
        else:
            select_img,exclude_img = None, None
        
        #选择遮罩批次
        if masks is not None:
            n_m = masks.shape[0]
            s_m = select_list[(select_list >= 1) & (select_list <= n_m)]-1
            if len(s_m) < 1:  # 若输入的编号全部不在范围内则返回原输入
                print("Warning:The input value is out of range, return to the original input.")
                exclude_mask, select_mask = masks, None
            else:
                e_m = np.setdiff1d(np.arange(0, n_m), s_m)  # 排除的图像
                select_mask = masks[torch.tensor(s_m, dtype=torch.int)]
                exclude_mask = masks[torch.tensor(e_m, dtype=torch.int)]
        else:
            select_mask,exclude_mask = None, None

        return (select_img, exclude_img,
                select_mask, exclude_mask)


class Select_Batch_v2:
    DESCRIPTION = """
        功能：
        返回指定批次编号处的图像，第1张编号为0,可以任意重复和排列组合
        指定的批次编号可按一定规则处理
        输入参数：
        indexes：若为空则为全选，可识别中文逗号。
        loop：将选择的批次复制指定次数
        loop_method：批次复制方式，tile直接增加，repeat每个编号往后复制loop个
        limit：超出范围的编号处理方式，Clamp钳制到最大批次内，Loop在最大批次内循环，ignore忽略
        images/masks：图像遮罩批次，批次数量可以不相同
        输出参数：
        select_img/exclude_img：选择/反选的图像(反选的将不进行批次处理)
        select_mask/exclude_mask：反选的遮罩(反选的将不进行批次处理)
        img_order/mask_order：选择的图像/遮罩批次原编号数据\n
        Functionality:
        Return the image at the specified batch number, with the first image numbered as 0, allowing for arbitrary repetition and combination.
        The specified batch numbers can be processed according to certain rules.
        Input parameters:
        indexes: If empty, select all; capable of recognizing Chinese commas.
        loop: The number of times to duplicate the selected batches.
        loop_method: The method of batch duplication, with 'tile' directly adding, and 'repeat' copying each number backward by the loop count.
        limit: The handling method for numbers out of range, 'Clamp' restricts to the maximum batch, 'Loop' cycles within the maximum batch, and 'ignore' disregards them.
        images/masks: Image/mask batches, which can have varying quantities.
        Output parameters:
        select_img/exclude_img: Selected/inverse-selected images (inverse-selected will not undergo batch processing).
        select_mask/exclude_mask: Inverse-selected masks (inverse-selected will not undergo batch handling).
        img_order/mask_order: The original batch number data of the selected images/masks.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "indexes": ("STRING", {"default": "1,2,","multiline": True}),
                "loop": ("INT", {"default":1,"min":1,"max":2048}),
                "loop_method": (["tile","repeat"],{"default":"tile"}),
                "limit": (["Clamp","Loop","ignore"],{"default":"Clamp"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE","MASK","MASK","LIST","LIST",)
    RETURN_NAMES = ("select_img", "exclude_img","select_mask", "exclude_mask","img_order","mask_order")
    FUNCTION = "SelectImages"

    def SelectImages(self, indexes, loop, loop_method, limit, images=None, masks=None):
        #初始值
        data_none = [None,None]
        img_order = []
        mask_order = []
        select_list = np.array(str_edit.tolist_v2(indexes, to_oneDim=True, to_int=True, positive=True))

        #如果选择列表为空值则直接输出
        n_s = len(select_list) 
        if n_s == 0 : 
            print("Warning:No valid input detected, original data will be output!")
            return (images,None,masks,None,img_order,mask_order)

        #选择图像
        if images is not None : 
            img_order = self.handle_data(select_list,loop,images.shape[0],limit,loop_method,indexes)
            select_img,exclude_img,_ = self.select_data(images,img_order)
        else: 
            print("Warning: Image input is empty, output will be empty")
            select_img,exclude_img = data_none

        #选择遮罩
        if masks is not None : 
            mask_order = self.handle_data(select_list,loop,masks.shape[0],limit,loop_method,indexes)
            select_mask,exclude_mask,_ = self.select_data(masks,img_order)
        else:
            print("Warning: Mask input is empty, output will be empty")
            select_mask,exclude_mask = data_none

        return (select_img, exclude_img, 
                select_mask, exclude_mask, 
                img_order, mask_order)
    
    #处理选择数据
    def handle_data(self,select_list,loop,n,limit,loop_method,indexes):
        """
        select_list：原选择列表
        loop：循环复制数量
        n:原批次总数
        limit：超过的序号补充方式(3种)
        loop_method：循环复制的方式
        indexes:输入的原字符串编号，若为""则为全选

        返回：处理后的选择列表
        """
        if indexes == "":
            select_list = np.arange(0,n)
        if limit == "Loop": #将超过最大批次的序号在最大批次内循环
            if loop_method == "tile":
                return np.mod(np.tile(select_list,loop),n)
            elif loop_method == "repeat":
                return np.mod(np.repeat(select_list,loop),n)
        elif limit == "Clamp": #钳制超过最大批次的序号到最大批次以内
            if loop_method == "tile":
                return np.clip(np.tile(select_list,loop),0,n-1)
            elif loop_method == "repeat":
                return np.clip(np.repeat(select_list,loop),0,n-1)
        elif limit == "ignore": #忽略超过最大批次的序号
            if indexes != "":
                select_list = select_list[(select_list >= 1) & (select_list <= n)]-1
            if loop_method == "tile":
                return np.tile(select_list,loop)
            elif loop_method == "repeat":
                return np.repeat(select_list,loop)
        else:
            return None
    
    #选择批次
    def select_data(self, t, list):
        """
        参数：
        t:批次图像/遮罩
        list:选择编号列表

        返回: 
        e_s: 反选编号列表
        t:重新组合的批次数据
        e_t:反选批次数据
        """
        e_s = np.setdiff1d(np.arange(0, t.shape[0]), list)
        e_t = t[torch.tensor(e_s, dtype=torch.int)]
        t = t[torch.tensor(list, dtype=torch.int)]
        return(t,e_t,e_s)


class SelectBatch_paragraph: #开发中
    DESCRIPTION = """
    功能：
    返回指定批次段(第一张图像编号为0)
    独立计算图像和遮罩输入(批次和大小可以不同)
    输入参数：
    Item：起始批次编号(包含此编号的图像)，为负数时从后面计数
    length：选择批次长度，默认从后面计数，为负数时从前面计数
    extend：当选择长度超过第一或最后一张时的填充方式
    *extend-no_extend：不填充，
    *extend-start_extend：填充第一张到指定长度
    *extend-end_extend：填充最后一张到指定长度
    *extend-loop：循环填充到指定长度
    *extend-loop_mirror：填充镜像循环填充到指定长度
    reversal_batch：反转批次
    输入参数：
    select_img/exclude_img：选择的图像批次/排除的图像批次
    select_mask/exclude_mask：选择的遮罩批次/排除的遮罩批次
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Item": ("INT", {"default": 0,"min":-4096,"max":4096}),
                "length": ("INT", {"default": 1,"min":-4096,"max":4096}),
                "extend": (["no_extend","start_extend","end_extend","loop","loop_mirror"],{"defailt":"no_extend"}),
                "reversal_batch": ("BOOLEAN",{"default":False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE","MASK","MASK",)
    RETURN_NAMES = ("select_img", "exclude_img","select_mask", "exclude_mask")
    FUNCTION = "SelectImages"

    def SelectImages(self, images, mode1_indexes):
        select_list = np.array(str_edit.tolist_v2(
            mode1_indexes, to_oneDim=True, to_int=True, positive=True))
        select_list1 = select_list[(select_list >= 1) & (select_list <= len(images))]-1
        if len(select_list1) < 1:  # 若输入的编号全部不在范围内则返回原输入
            print(
                "Warning:The input value is out of range, return to the original input.")
            return (images, None)
        else:
            exclude_list = np.arange(1, len(images) + 1)-1
            exclude_list = np.setdiff1d(exclude_list, select_list1)  # 排除的图像
            if len(select_list1) < len(select_list):  # 若输入的编号超出范围则仅输出符合编号的图像
                n = abs(len(select_list)-len(select_list1))
                print(
                    f"Warning:The maximum value entered is greater than the batch number range, {n} maximum values have been removed.")
            print(f"Selected the first {select_list1} image")
            return (images[torch.tensor(select_list1, dtype=torch.float)], 
                    images[torch.tensor(exclude_list, dtype=torch.float)])
 

class Batch_Average: #开发中
    DESCRIPTION = """
    功能：
    将批次平均切割
    输入：
    Item：分段开始序号
    division：分段数
    select：选择输出第几段
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Item": ("INT", {"default": 0,"min":-4096,"max":4096}),
                "division": ("INT", {"default": 1,"min":1,"max":4096}),
                "select": ("INT", {"default": 1,"min":-4096,"max":4096}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE","MASK","MASK",)
    RETURN_NAMES = ("select_img", "exclude_img","select_mask", "exclude_mask")
    FUNCTION = "SelectImages"

    def SelectImages(self, images):
        ...


class Mask_Detection:
    DESCRIPTION = """
    Input mask and perform duplicate detection:
        1: Whether it exists.
        2: Whether it is a hard edge (binary value).
        3: Whether it is an all-white mask.
        4: Whether it is an all-black mask.
        5: Is it a grayscale mask
        6: Output color value (output 0 when mask is not monochrome)

    输入mask,使用去重检测：
        1:是否存在
        2:是否为硬边缘(二值)
        3:是否为全白遮罩
        4:是否为全黑遮罩
        5:是否为灰度遮罩
        6:输出色值(当mask不为单色时输出0)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN",
                    "BOOLEAN", "BOOLEAN", "int")
    RETURN_NAMES = ("Exist?", "HardEdge", "PureWhite",
                    "PureBlack", "PureGray", "PureColorValue")
    FUNCTION = "MaskDetection"

    def MaskDetection(self, mask):
        Exist = True
        binary = False
        PureWhite = False
        PureBlack = False
        PureGray = False
        PureColorValue = int(0)
        data = torch.unique(mask).tolist()
        n = len(data)
        if n == 1:
            Exist = False
            PureColorValue = int(round(data[0] * 255))
            if data[0] == 0:
                PureBlack = True
            elif data[0] == 1:
                PureWhite = True
            else:
                PureGray = True
        if n <= 2:
            binary = True
        return (Exist, binary, PureWhite, PureBlack, PureGray, PureColorValue)


class get_image_data:
    DESCRIPTION = """
    Obtain image data
    获取图像数据
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "image":("IMAGE",),
                "mask":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("INT","INT","INT","INT","INT","INT")
    RETURN_NAMES = ("N","H","W","C","max_HW","min_HW",)
    FUNCTION = "element_count"

    def element_count(self, image = None, mask = None):
        shape = [0,0,0,0]
        if mask is not None:
            shape = list(mask.shape)
            shape.append(1)
        if image is not None:
            shape = list(image.shape)
        m = [max(shape[1:3]),min(shape[1:3])]
        return (*shape,*m)


class get_TypeName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": (any,),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TypeName",)
    OUTPUT_NODE = True
    FUNCTION = "TypeName"

    def TypeName(self, data, ):
        name = str(type(data).__name__)
        print(f"Prompt:The input data type is --->{name}")
        return (name,)


class array_count:
    DESCRIPTION = """
    Retrieve the shape of array class data and count the number of elements
    获取数组类数据的形状，统计元素数量
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_data": (any,),
                "select_dim":("INT",{"default":0,"min":0,"max":64}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST","INT","INT","INT","INT","INT","INT")
    RETURN_NAMES = ("shape","image-N","image-H","image-W","image-C","sum_count","sel_count",)
    FUNCTION = "element_count"

    def element_count(self, any_data, select_dim):
        n, n1= 1, 1
        s = [0,0,0,0]
        try:
            s = list(any_data.shape)
        except:
            print("Warning: This object does not have a shape property, default output is 0")
        #try:
        shape = list(any_data.shape)
        if len(shape) == 0:
            n, n1= 0, 0
        else:
            for i in range(len(shape)):
                n *= shape[i]
                if i >= select_dim:
                    n1 *= shape[i]
        #except:
        #    print("Error: The input data does not have array characteristics.")
        return (s,*s,n,n1)


CATEGORY_NAME = "WJNode/Other-node"


class any_data:
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

    def any_data_array(self, data_array=[None, None, None, None, None, None],
                       data_1=None,
                       data_2=None,
                       data_3=None,
                       data_4=None,
                       data_5=None,
                       data_6=None):

        if data_1 is None:
            data_1 = data_array[0]
        else:
            data_array[0] = data_1

        if data_2 is None:
            data_2 = data_array[1]
        else:
            data_array[1] = data_2

        if data_3 is None:
            data_3 = data_array[2]
        else:
            data_array[2] = data_3

        if data_4 is None:
            data_4 = data_array[3]
        else:
            data_array[3] = data_4

        if data_5 is None:
            data_5 = data_array[4]
        else:
            data_array[4] = data_5

        if data_6 is None:
            data_6 = data_array[5]
        else:
            data_array[5] = data_6

        return (data_array, data_1, data_2, data_3, data_4, data_5, data_6,)


class Folder_link: #开发中
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
        path_aux = os.path.join(path_custom, "comfyui_controlnet_aux", "ckpts")
        
        path_model_link = folder_paths.models_dir
        path_custom_link = folder_paths.folder_names_and_paths["custom_nodes"]
        path_aux_link = os.path.join(path_custom_link, "comfyui_controlnet_aux", "ckpts")


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


CATEGORY_NAME = "WJNode/Other-plugins"


class WAS_Mask_Fill_Region_batch:
    DESCRIPTION = """
    Original plugin: was-node-suite-comfyui
    Original node: WAS_Mask_Fill_Region
    change: batch bug in mask processing
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


class SegmDetectorCombined_batch:
    DESCRIPTION = """
    Original plugin: impack-pack
    Original node: SegmDetectorCombined
    change: batch detection of masks
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segm_detector": ("SEGM_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY_NAME

    def doit(self, segm_detector, image, threshold, dilation):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        mask = torch.zeros((0,*image.shape[1:-1]), dtype=torch.float, device="cpu")
        mask_0 = torch.zeros((1,*image.shape[1:-1]), dtype=torch.float32, device="cpu")
        if image.shape[0] != 1:
            for i in range(image.shape[0]):
                mask_temp = segm_detector.detect_combined(image[i].unsqueeze(0), threshold, dilation)
                if mask_temp is None:
                    mask_temp = mask_0
                else:
                    mask_temp = mask_temp.unsqueeze(0)
                mask = torch.cat((mask, mask_temp), dim=0)
        else:
            mask = segm_detector.detect_combined(image, threshold, dilation)
            if mask is None:
                mask = mask_0
            else:
                mask = mask.unsqueeze(0)
        return (mask,)


class bbox_restore_mask:
    DESCRIPTION = """
    Original plugin: impack-pack
    crop_region:Restore cropped image (SEG editing)
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


class Sam2AutoSegmentation_data:
    DESCRIPTION = """
    Original plugin: ComfyUI-segment-anything-2 
    Original node: Sam2AutoSegmentation
    change: data output
    purpose: Get coordinates/Get object mask
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
    "Select_Images_Batch": Select_Images_Batch,
    "Select_Batch_v2": Select_Batch_v2,
    "Mask_Detection": Mask_Detection,
    #WJNode/Other-functions
    "any_data": any_data,
    "get_TypeName": get_TypeName,
    "array_count": array_count,
    "get_image_data": get_image_data,
    #WJNode/Other-plugins
    "WAS_Mask_Fill_Region_batch": WAS_Mask_Fill_Region_batch,
    "SegmDetectorCombined_batch": SegmDetectorCombined_batch,
    "bbox_restore_mask": bbox_restore_mask,
    "Sam2AutoSegmentation_data": Sam2AutoSegmentation_data,
}
