import torch
import os
import json
import re
from typing import List, Dict, Tuple

import folder_paths
from ..moduel.str_edit import str_edit
from ..moduel.color_utils import distance_color,convert_color,convert_rgb_1_255
from ..config.color_name.color_data_edit import select_region,NameSelect_ColorData
from ..config.seg_color.color_data import DensePose_Data
CATEGORY_NAME = "WJNode/Color"


class load_color_config:
    DESCRIPTION = """
    说明：
    tag_language 只有在 version 1.1 以上时才有效

    describe(en):
    Tag_Language is only valid when version 1.1 or above
    """
    config_file_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-WJNodes/config/seg_color")
    @classmethod
    def INPUT_TYPES(s):
        config_file_list = []
        for filename in os.listdir(s.config_file_path):
            if filename.endswith('.json'):
                config_file_list.append(filename)
        version_list = ["1.0","1.1"]
        language_list = ["EN","CH"]
        return {
            "required": {
                "config_file":(config_file_list,{"default":config_file_list[0]}),
                "version":(version_list,{"default":version_list[0]}),
                "tag_language":(language_list,{"default":language_list[0]})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST","DICT","STRING")
    RETURN_NAMES = ("Color_list","Color_dict","tag_list")
    FUNCTION = "color_config"
    def color_config(self, config_file,version,tag_language):
        with open(os.path.join(self.config_file_path, config_file),'r',encoding='utf-8') as file:
            color_data = json.load(file)
        color_list = []
        color_dict = {}
        tag_list = list(color_data.keys())
        if version == "1.0":
            color_list = list(color_data.values())
            color_dict = color_data
        elif version == "1.1":
            color_list = [v["RGB255"] for k,v in color_data.items()]
            if tag_language == "EN":
                color_dict = {k:v["RGB255"] for k,v in color_data.items()}
            elif tag_language == "CH":
                color_dict = {v["CH"]:v["RGB255"] for k,v in color_data.items()}
                tag_list = list(color_dict.keys())
            else:
                raise ValueError("Error:Unsupported language types!")
        else:
            raise ValueError("Error:There is no such version available!")
        return (color_list,color_dict,tag_list)


class filter_DensePose_color:
    DESCRIPTION = """
    说明：
    筛选 DensePose 色块数据

    explain:
    Filter DensePose color block data
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Color_dict":("DICT",),
                "arm":("BOOLEAN",{"default":True}),
                "forearm":("BOOLEAN",{"default":True}),
                "thigh":("BOOLEAN",{"default":True}),
                "calf":("BOOLEAN",{"default":True}),
                "head":("BOOLEAN",{"default":True}),
                "body":("BOOLEAN",{"default":True}),
                "hand":("BOOLEAN",{"default":False}),
                "foot":("BOOLEAN",{"default":False}),
                "left":("BOOLEAN",{"default":True}),
                "right":("BOOLEAN",{"default":True}),
                "front":("BOOLEAN",{"default":True}),
                "behind":("BOOLEAN",{"default":False}),
                "inside": ("BOOLEAN",{"default":False}),
                "outside": ("BOOLEAN",{"default":False}),
                "background":("BOOLEAN",{"default":False})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("DICT","DICT")
    RETURN_NAMES = ("Selective","Excluded")
    FUNCTION = "Select"
    def Select(self,Color_dict,**kwargs):
        #新建关键字用于筛选
        tag_list = []
        for k,v in kwargs.items():
            if v : tag_list.append(str(k)) #收集启用的关键字
        tag_list = list(tag_list) #generator对象转列表

        #全选和全不选时直接输出
        if tag_list == []:
            return ({},Color_dict)
        elif set(tag_list) == set(Color_dict.keys()):
            return (Color_dict,{})

        filter_list = {} #定义查找集
        DP_filter_en = DensePose_Data['DP_filter_en']
        if self.is_en(Color_dict):
            #若Color_dict输入为英文则获取与Color_dict等长的英文二维查找集
            if len(Color_dict.keys()) == len(DP_filter_en.keys()):
                filter_list = DP_filter_en
            else:
                filter_list = {k:DP_filter_en[k] for k,v in Color_dict.items()}
        else:
            #若Color_dict输入为中文则获取与Color_dict等长的中文二维查找集
            DP_filter_ch = DensePose_Data['DP_filter_ch']
            if len(Color_dict.keys()) == len(DP_filter_en.keys()):
                filter_list = DP_filter_ch
            else:
                ch_k = list(DP_filter_ch.keys())
                for i in range(len(list(Color_dict.keys()))):
                    k = ch_k[i]
                    filter_list[k] = DP_filter_ch[k]
            #若Color_dict输入为中文则将关键字转中文
            tag_list = [DensePose_Data['DP_tag_en2ch'][i] for i in tag_list]
        return self.filter(tag_list,Color_dict,filter_list)

    #判断元素/key/字符串的第一个字符是否为字母
    def is_en(self,str): 
        if isinstance(str,dict): str = list(str.keys())[0]
        if isinstance(str,list): str = str[0]
        if isinstance(str,(int,bool,float)): return False
        if str and re.match(r'^[a-zA-Z]', str): return True
        return False
    
    #查找关键字是否在每个查找集中，若存在则收集对应的Color_dict数据
    def filter(self,tag_list,Color_dict,filter_list):
        color_s , color_e = {} , {}
        filter_k = list(filter_list.keys())
        for i in range(len(filter_k)):
            for j in tag_list:
                k = filter_k[i]
                if j in filter_list[k]:
                    try:
                        color_s[k] = Color_dict[k]
                    except:
                        pass
                    break
        for i in list(Color_dict.keys()):
            if not i in color_s.keys():
                color_e[i] = Color_dict[i]
        return (color_s,color_e)


class color_segmentation:
    DESCRIPTION = """
    
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "select_batch":("INT",{"default":0,"min":0,"max":999999}),
                "skip_threshold":("FLOAT",{"default":0.05,"min":0,"max":1,"step":0.001,"display":"slider"}),
                "Color_list":("LIST",),
                "merge_mask":("BOOLEAN",{"default":False}),
                "invert_mask":("BOOLEAN",{"default":False})
            },
            "optional": {
                "select_mask":("MASK",),
                "invert_select_mask":("BOOLEAN",{"default":False}),
                "invert_select":("BOOLEAN",{"default":False})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "separate_color_blocks"
    def separate_color_blocks(self, image, select_batch, skip_threshold, Color_list, merge_mask, invert_mask, invert_select_mask, invert_select, select_mask=None):
        #select batch and to 3 dimensions
        try:
            if image.shape[0] != 1:
                image = image[select_batch]
                if len(Color_list) == 1:
                    Color_list = Color_list[0]
                else:
                    Color_list = Color_list[select_batch]
            else:
                image = image[0]
                Color_list = Color_list[0]
        except:
            print("warn-color_segmentation: The selected batch exceeds the input batch and has been changed to the 0th batch")
            image = image[0]
            Color_list = Color_list[0]

        #create mask
        mask = self.color_to_mask(image, Color_list, skip_threshold)
        mask = self.handle_mask(mask, invert_mask)

        #select mask
        if select_mask is not None:
            mask = self.select_mask(mask, select_mask, invert_select_mask, invert_select)
        if merge_mask:
            mask = self.merge_maks(mask)

        return (mask.float(),)

    #color to mask
    def color_to_mask(self, image, Color_list, skip_threshold):
        device = image.device
        shape = image.shape[:-1]
        
        skip_threshold = int(skip_threshold / 100 * shape[0] * shape[1])+1
        image = (torch.round(image * 255)).int()
        mask = torch.zeros([0,*shape], dtype=torch.int, device=device).bool()
        for i in range(len(Color_list)):
            b=False
            mask_i = torch.zeros([0,*shape], dtype=torch.int, device=device).bool()
            for j in range(3):
                mask_temp = abs(image[:,:,j] - Color_list[i][j]).bool()
                if torch.sum(~mask_temp) <= skip_threshold:
                    b=True
                    break
                mask_i = torch.cat((mask_i, mask_temp.repeat(1,1,1)), dim=0)
            if b:
                continue
            for j in range(2):
                mask_i[0] = mask_i[0] | mask_i[j+1]
            mask = torch.cat((mask, mask_i[0].repeat(1,1,1)), dim=0)
        return mask

    #invert mask
    def handle_mask(self, mask, invert, inspect=True):
        if not invert:
            mask = ~mask
        if inspect:
            new_mask = torch.zeros((0,*mask.shape[1:])).bool()
            for i in range(mask.shape[0]):
                if torch.sum(mask[i]) > 1:
                    new_mask = torch.cat((new_mask, mask[i].repeat(1,1,1)), dim=0)
        return new_mask
    
    #merge mask
    def merge_maks(self, mask):
        for i in range(len(mask)-1):
            mask[0] = mask[0] | mask[i+1]
        mask = mask[0].repeat(1,1,1)
        return mask
    
    #select mask
    def select_mask(self, mask, select_mask, invert_select_mask, invert_select):
        select_mask = torch.round(select_mask).bool()[0]
        if invert_select_mask:
            select_mask = ~select_mask
        new_mask = torch.zeros((0,*mask.shape[1:])).bool()
        new_mask_i = torch.zeros((0,*mask.shape[1:])).bool()
        for i in range(mask.shape[0]):
            mask_temp = mask[i] & select_mask
            if torch.sum(mask_temp) > 1:
                new_mask = torch.cat((new_mask, mask[i].repeat(1,1,1)), dim=0)
            else:
                new_mask_i = torch.cat((new_mask_i, mask[i].repeat(1,1,1)), dim=0)
        if invert_select:
            return new_mask_i
        else:
            return new_mask


class color_segmentation_v2:
    DESCRIPTION = """
    功能：
    1：用于将seg色块类输出图像批量转为遮罩
        支持输入配置文件内的tag筛选，所有tag在加载配置节点可查看
    2：使用遮罩来选择seg色块图内的对象
        例如手绘遮罩或深度遮罩选择对象，注意输入尺寸要一致
    说明：
    1：单张没有查找到颜色时，输出全黑遮罩(打开invert_mask反转时全白)
    2：部分批次没有查找到颜色时，对应顺序输出黑色，其它正常
    3：合并遮罩merge_mask在输入为批次时，即使关闭也默认开启
    4：选择遮罩select_mask功能目前仅支持单张,不支持批次

    Function:
    1: Used to batch convert SEG color block class output images into masks
        Support tag filtering within input configuration files, 
        all tags can be viewed on the loading configuration node
    2: Use masks to select objects within the SEG color block image
        For example, when selecting objects for hand drawn masks or depth masks, 
        be sure to input consistent sizes
    explain:
    1: When no color is found for a single sheet,
        output a full black mask (full white when inverted with invert_mask turned on)
    2: When no color is found in some batches, black is outputted in the corresponding order, 、
        while others are normal
    3: When the merge mask is input as a batch, it is enabled by default even if it is turned off
    4: The select_mask function currently only supports single masks and does not support batches
    """
    @classmethod
    def INPUT_TYPES(s):
        # 定义输入参数的类型和默认值
        tag_mod = ["all tag", "background tag", "custom tag"]
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "skip_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.001, "display": "slider"}),  # 跳过阈值
                "Color_dict": ("DICT",),  # 颜色字典
                "merge_mask": ("BOOLEAN", {"default": False}),  # 是否合并遮罩
                "invert_mask": ("BOOLEAN", {"default": False}),  # 是否反转遮罩
                "output_type": (tag_mod, {"default": tag_mod[0]}),  # 输出类型
                "custom_keys": ("STRING", {"default": "wall,sky,floor,ceiling,windowpane,traffic", "multiline": True}),  # 自定义键
            },
            "optional": {
                "select_mask": ("MASK",),  # 选择遮罩
                "invert_select_mask": ("BOOLEAN", {"default": False}),  # 是否反转选择遮罩
                "invert_select": ("BOOLEAN", {"default": False})  # 是否反转选择
            }
        }
    CATEGORY = CATEGORY_NAME  # 类别名称
    RETURN_TYPES = ("MASK",)  # 返回类型
    RETURN_NAMES = ("mask",)  # 返回名称
    FUNCTION = "separate_color_blocks"  # 主要函数

    def separate_color_blocks(self, image, skip_threshold, Color_dict, merge_mask, 
                              invert_mask, invert_select_mask, invert_select, 
                              output_type, custom_keys, select_mask=None):
        """
        主函数：根据颜色分割图像并生成遮罩
        """
        # 检查颜色字典是否为空
        if Color_dict == {}: 
            return self.none_data(image, invert_mask)

        # 根据输出类型生成颜色列表
        Color_list = []
        if output_type == "all tag":
            Color_list = [list(Color_dict.values()),]  # 使用所有颜色
        elif output_type == "custom tag":
            custom_keys = self.remove_trailing_comma(custom_keys)  # 移除自定义键末尾的逗号
            if custom_keys == "":  # 自定义键为空直接返回结果
                return self.none_data(image, invert_mask)
            keys = re.split(r',\s*|，\s*', custom_keys)  # 分割自定义键
            if set(keys) & set(Color_dict.keys()) == set():  # 自定义键无效直接返回结果
                return self.none_data(image, invert_mask)
            Color_list = [[Color_dict[key] for key in keys if key in Color_dict],]  # 使用自定义键对应的颜色
        elif output_type == "background tag":
            keys = ['wall', 'sky', 'floor', 'ceiling', 'windowpane', 'traffic', 'background', 'back']  # 背景键
            Color_list = [[Color_dict[key] for key in keys if key in Color_dict],]  # 使用背景键对应的颜色
        else:
            raise ValueError('Error:Invalid tag output mode!')  # 抛出无效输出类型异常

        # 初始化遮罩
        mask = torch.zeros((0, *image.shape[1:-1]), dtype=torch.float)

        # 处理图像批次
        try:
            n = image.shape[0]  # 获取图像批次大小
            if n != 1:
                if len(Color_list) != n and len(Color_list) != 1:  # 检查颜色列表与图像批次是否匹配
                    print("Error-color_segmentation_v2: The number of color_list does not match the number of images")
                elif len(Color_list) == 1:  # 如果颜色列表只有一个元素
                    Color_list = Color_list[0]
                    for i in range(n):
                        image_temp = image[i]  # 获取当前图像
                        mask_temp = self.run_color_segmentation(image_temp, Color_list, skip_threshold, select_mask, invert_select_mask, invert_select, True)  # 运行颜色分割
                        mask = torch.cat((mask, mask_temp), dim=0)  # 将结果添加到遮罩中
                else:
                    for i in range(n):
                        image_temp = image[i]  # 获取当前图像
                        Color_list_temp = Color_list[i]  # 获取当前颜色列表
                        mask_temp = self.run_color_segmentation(image_temp, Color_list_temp, skip_threshold, select_mask, invert_select_mask, invert_select, True)  # 运行颜色分割
                        mask = torch.cat((mask, mask_temp), dim=0)  # 将结果添加到遮罩中
            else:
                mask = self.run_color_segmentation(image[0], Color_list, skip_threshold, select_mask, invert_select_mask, invert_select, merge_mask)  # 运行颜色分割
        except:
            print("warn-color_segmentation: The selected batch exceeds the input batch and has been changed to the 0th batch")
            mask = self.run_color_segmentation(image[0], Color_list[0], skip_threshold, select_mask, invert_select_mask, invert_select, merge_mask)  # 捕获异常并处理

        if invert_mask : 
            if mask.dtype == torch.bool:
                mask = (~mask).float()
            else:
                mask = 1.0 - mask

        return (mask,)  # 返回结果

    def remove_trailing_comma(self, s):
        """
        移除字符串末尾的逗号、空格或中文逗号
        """
        if s.endswith((',', "，", " ")):
            return self.remove_trailing_comma(s[:-1])
        return s

    def run_color_segmentation(self, image, Color_list, skip_threshold, select_mask, invert_select_mask, invert_select, merge_mask):
        """
        运行单张图像颜色分割
        """
        mask = self.color_to_mask(image, Color_list, skip_threshold)  # 将颜色转换为遮罩
        if select_mask is not None:  # 如果有选择遮罩
            mask = self.select_mask(mask, select_mask, invert_select_mask, invert_select)  # 选择遮罩
        if merge_mask:  # 如果需要合并遮罩
            mask = self.merge_maks(mask)  # 合并遮罩
        return mask

    def color_to_mask(self, image, Color_list, skip_threshold):
        """
        将单张图像颜色转换为遮罩批次
        """
        device = image.device  # 获取设备
        shape = image.shape[:-1]  # 获取图像形状

        # 计算跳过阈值
        skip_threshold = int(skip_threshold / 100 * shape[0] * shape[1]) + 1
        image = (torch.round(image * 255)).int()  # 将图像值四舍五入并转换为整数
        mask = torch.zeros([0, *shape], dtype=torch.int, device=device).bool()  # 初始化遮罩

        # 遍历颜色列表
        for i in range(len(Color_list)):
            b = False
            mask_i = torch.zeros([0, *shape], dtype=torch.int, device=device).bool()  # 初始化当前颜色遮罩
            for j in range(3):  # 遍历颜色通道
                mask_temp = abs(image[:, :, j] - Color_list[i][j]).bool()  # 计算当前通道遮罩
                if torch.sum(~mask_temp) <= skip_threshold:  # 如果当前通道遮罩的false值数量小于跳过阈值
                    b = True
                    break
                else:
                    mask_i = torch.cat((mask_i, mask_temp.repeat(1, 1, 1)), dim=0)  # 将当前通道遮罩添加到当前颜色遮罩中
            if b:  # 如果当前颜色遮罩无效
                continue
            for j in range(2):  # 合并当前颜色遮罩的通道
                mask_i[0] = mask_i[0] | mask_i[j + 1]
            mask = torch.cat((mask, mask_i[0].repeat(1, 1, 1)), dim=0)  # 将当前颜色遮罩添加到遮罩中
        mask = ~mask
        if mask.shape[0] == 0:
            mask = torch.zeros([1,*shape], dtype=torch.int, device=device)
        return mask

    def handle_mask(self, mask, invert, inspect=True): #废弃
        """
        处理遮罩
        """
        if invert:  # 如果需要反转遮罩
            mask = ~mask
        #if inspect:  # 如果需要检查遮罩
        #    new_mask = torch.zeros((0, *mask.shape[1:])).bool()  # 初始化新遮罩
        #    for i in range(mask.shape[0]):  # 遍历遮罩
        #        if torch.sum(mask[i]) > 1:  # 如果当前遮罩的值数量大于1
        #            new_mask = torch.cat((new_mask, mask[i].repeat(1, 1, 1)), dim=0)  # 将当前遮罩添加到新遮罩中
                
        return mask

    def merge_maks(self, mask):
        """
        合并遮罩
        """
        if mask.shape[0] > 1:
            for i in range(len(mask) - 1):  # 遍历遮罩
                mask[0] = mask[0] | mask[i + 1]  # 合并当前遮罩
            mask = mask[0].repeat(1, 1, 1)  # 将合并后的遮罩复制到所有位置
        return mask

    def select_mask(self, mask, select_mask, invert_select_mask, invert_select):
        """
        选择遮罩
        """
        select_mask = torch.round(select_mask).bool()[0]  # 将选择遮罩四舍五入并转换为布尔值
        if invert_select_mask:  # 如果需要反转选择遮罩
            select_mask = ~select_mask
        new_mask = torch.zeros((0, *mask.shape[1:])).bool()  # 初始化新遮罩
        new_mask_i = torch.zeros((0, *mask.shape[1:])).bool()  # 初始化新遮罩的逆
        for i in range(mask.shape[0]):  # 遍历遮罩
            mask_temp = mask[i] & select_mask  # 计算当前遮罩与选择遮罩的交集
            if torch.sum(mask_temp) > 1:  # 如果交集的值数量大于1
                new_mask = torch.cat((new_mask, mask[i].repeat(1, 1, 1)), dim=0)  # 将当前遮罩添加到新遮罩中
            else:
                new_mask_i = torch.cat((new_mask_i, mask[i].repeat(1, 1, 1)), dim=0)  # 将当前遮罩添加到新遮罩的逆中
        if invert_select:  # 如果需要反转选择
            return new_mask_i
        else:
            return new_mask

    def none_data(self, image, invert_mask):
        """
        处理空数据
        """
        mask = None
        if invert_mask:  # 如果需要反转遮罩
            mask = torch.ones(image.shape[0:-1])
        else:
            mask = torch.zeros(image.shape[0:-1])
        return (mask,)


class load_ColorName_config:
    DESCRIPTION = """
    """
    config_file_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-WJNodes/config/color_name")
    @classmethod
    def INPUT_TYPES(s):
        config_file_list = []
        for filename in os.listdir(s.config_file_path):
            if filename.endswith('.json'):
                config_file_list.append(filename)
        config_class = ['All', 'Common', 'Unusual', 'CoolColor', 'WarmColor', 'DarkColor', 'LightColour']
        return {
            "required": {
                "config_file":(config_file_list,{"default":config_file_list[0]}),
                "filter_class":(config_class,{"default":config_class[0]}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("DICT","DICT")
    RETURN_NAMES = ("Color_data_select","Color_data_excluded")
    FUNCTION = "load_config"
    def load_config(self, config_file,filter_class):
        #读取配置文件
        config_file_path = os.path.join(self.__class__.config_file_path,config_file)
        with open(config_file_path, 'r', encoding='utf-8') as file:
            color_dict = json.load(file)
        if filter_class == 'All':
            return (color_dict,None)
        else:
            return NameSelect_ColorData(color_dict,color_dict["default1_class"][filter_class])


class ColorData_HSV_Capture: #待测试
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Color_data":("DICT",),
                "filter_H_min":("FLOAT",{"default":0.0,"min":0.0,"max":360.0,"step":0.001}),
                "filter_H_max":("FLOAT",{"default":360.0,"min":0.0,"max":360.0,"step":0.001}),
                "filter_S_min":("FLOAT",{"default":0.0,"min":0.0,"max":1.0,"step":0.001}),
                "filter_S_max":("FLOAT",{"default":1.0,"min":0.0,"max":1.0,"step":0.001}),
                "filter_V_min":("FLOAT",{"default":0.0,"min":0.0,"max":1.0,"step":0.001}),
                "filter_V_max":("FLOAT",{"default":1.0,"min":0.0,"max":1.0,"step":0.001})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("DICT","DICT")
    RETURN_NAMES = ("Color_data_select","Color_data_excluded")
    FUNCTION = "load_config"
    def load_config(self, Color_data,
                    filter_H_min,filter_H_max,
                    filter_S_min,filter_S_max,
                    filter_V_min,filter_V_max):
#        filter_list = [
#                    [filter_H_min,filter_H_max],
#                    [filter_S_min,filter_S_max],
#                    [filter_V_min,filter_V_max]
#                    ]

        #根据HSV值筛选输出数据
        H_min = min(filter_H_min,filter_H_max)
        H_max = min(filter_H_min,filter_H_max)
        S_min = min(filter_S_min,filter_S_max)
        S_max = min(filter_S_min,filter_S_max)
        V_min = min(filter_V_min,filter_V_max)
        V_max = min(filter_V_min,filter_V_max)

        if sum([H_min,S_min,V_min]) == 0.0 and sum([H_max,S_max,V_max]) == 362.0:
            return (
                    Color_data,
                    {"default1_RGB_D":[],"default1_HSV_D":[],"default1_value_all":[],"default1_class":{},}
                    )
        else:
            output_v = Color_data["default1_value_all"]
            if H_min != 0.0:
                output_v = select_region(output_v,[0,H_min])
            if H_max != 360.0:
                output_v = select_region(output_v,[0,H_max],is_min=False)
            if S_min != 0.0:
                output_v = select_region(output_v,[1,S_min])
            if S_max != 360.0:
                output_v = select_region(output_v,[1,S_max],is_min=False)
            if V_min != 0.0:
                output_v = select_region(output_v,[2,V_min])
            if V_max != 360.0:
                output_v = select_region(output_v,[2,V_max],is_min=False)
            name = [i["Name_EN"] for i in output_v]
            return NameSelect_ColorData(Color_data,name)


class Color_check_Name:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        ColorType = ['RGB0-1', 'RGB','HEX', 'HSV']
        distance_type = ["RGB_distance","HSV_distance"]
        return {
            "required": {
                "Color_data":("DICT",),
                "Color_InputType":(ColorType,{"default":ColorType[0]}),
                "method":(distance_type,{"default":distance_type[0]}),
                "threshold":("FLOAT",{"default":0.01,"min":0.001,"max":0.999,"step":0.001}),
            },
            "optional": {
                "color_value1":("LIST",),
                "color_value2":("STRINT",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("DICT","DICT")
    RETURN_NAMES = ("Color_data","Color_data_exclude")
    FUNCTION = "check"
    def check(self,Color_data,Color_InputType,method,threshold,
              color_value1=None,color_value2=None):
        
        #输入检查
        if color_value1 is None:
            color_value = color_value2
        else: color_value = color_value1
        if color_value is None:
            raise ValueError("Error: Color value is None, Please enter the color value !")
        if len(color_value) == 0 :
            print("Warning: Color value is empty, return empty !")
            return (None,None,None,None,)
        #规范化颜色数据
        if len(color_value) == 1 :
            color_value = self.get_data(color_value)
            #适用于 WAS 插件的 image color palette 节点输出的HEX多换行符颜色
            if isinstance(color_value,str):
                color_value = color_value.split("\n")

            ##适用于多个三元色值输入处理，暂时不加入
            #elif len(color_value) == 3 and (isinstance(color_value[0],int) or isinstance(color_value[0],float)):
            #    color_value = [color_value]

            #无法识别的数据
            else:
                print("Warning: Color value cannot be recognized, return None !")
                return (None,None,None,None,)

        Distance_Name_Mapping = {"RGB_distance":"default1_RGB_D","HSV_distance":"default1_HSV_D"}
        method = Distance_Name_Mapping[method]
        #选择类型和数据初始化
        output_type = "RGB"
        threshold = threshold * 255 * (3**0.5)
        if method == "default1_HSV_D":
            output_type = "HSV"
            threshold = threshold * (3**0.5)

        #计算输入颜色在色彩空间的距离
        color_distance = distance_color(color_value,Color_InputType,output_type)
        #范围内选择颜色数据
        Color_list = []
        for i in color_distance:
            select,exclude = self.Distance_Select(Color_data[method],i,threshold)
            Color_list = Color_list + select #合并的数据组
            #Color_list.append(select) #数据分组
            #Color_list_exclude.append(exclude)
        name = [i["Name_EN"] for i in Color_list]
        return NameSelect_ColorData(Color_data,name)


    def Distance_Select(self, #根据色彩空间距离(List[Dict])查找相近的颜色数据
                            sorted_data: List[Dict],
                            target_v: float,
                            threshold: float
                            ) -> Tuple[List[Dict], List[Dict]]:
        """
        在数组中找到与目标值在 ±threshold 范围内的所有字典。

        参数:
        - sorted_data: 输入的字典数组，每个字典包含目标键（如 'v1'）。
        - target_k: 目标键（如 'v1'）。
        - target_v: 目标值，用于匹配范围内的字典。
        - threshold: 阈值，用于判断范围。

        返回:
        - closest_dicts: 在 ±threshold 范围内的所有字典。
        - excluded_dicts: 排除的字典列表。
        """
        target_k = "distance"
        # 二分查找第一个小于或等于目标值的索引
        left, right = 0, len(sorted_data) - 1
        first_index = -1

        while left <= right:
            mid = (left + right) // 2
            if sorted_data[mid][target_k] <= target_v:
                first_index = mid
                left = mid + 1
            else:
                right = mid - 1

        # 如果没有找到小于或等于目标值的索引，返回空列表
        if first_index == -1:
            return [], sorted_data

        # 从第一个索引开始，向前后扩展，找到所有在 ±threshold 范围内的字典
        closest_dicts = []
        excluded_dicts = []

        # 向前扩展
        for i in range(first_index, -1, -1):
            if abs(sorted_data[i][target_k] - target_v) <= threshold:
                closest_dicts.append(sorted_data[i])
            else:
                excluded_dicts.append(sorted_data[i])

        # 向后扩展
        for i in range(first_index + 1, len(sorted_data)):
            if abs(sorted_data[i][target_k] - target_v) <= threshold:
                closest_dicts.append(sorted_data[i])
            else:
                excluded_dicts.append(sorted_data[i])

        return closest_dicts, excluded_dicts
    
    def get_data(self,data:list):
        if len(data) == 1:
            data = data[0]
            if isinstance(data,list) or isinstance(data,tuple):
                self.get_data(data)
            else:
                if isinstance(data,tuple) or isinstance(data,set) :
                    data = list(data)
                return data
        else:
            if isinstance(data,tuple) or isinstance(data,set) :
                data = list(data)
            return data


class Color_Data_Break:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        ColorType = ['RGB0-1','RGB','HEX','HSV']
        language = ['Name_EN','Name_CH']
        return {
            "required": {
                "Color_data":("DICT",),
                "Color_OutputType":(ColorType,{"default":ColorType[0]}),
                "output_language":(language,{"default":language[0]}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING","LIST","LIST")
    RETURN_NAMES = ("ColorPrompt","PromptList","Colorlist")
    FUNCTION = "run"
    def run(self,Color_data,Color_OutputType,output_language):

        #获取颜色值
        Colorlist = []
        if Color_OutputType != 'RGB0-1':
            Colorlist = [i[Color_OutputType] for i in Color_data["default1_value_all"]]
        else:
            Colorlist = [i["RGB"] for i in Color_data["default1_value_all"]]
            Colorlist = convert_rgb_1_255(Colorlist)

        #获取颜色名
        PromptList = [i[output_language] for i in Color_data["default1_value_all"]]

        return (str_edit.list_to_str(PromptList),PromptList,Colorlist)



class Name_check_Color: #待开发
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Color_data":("DICT",),
                "color_name":("STRING",{"default":"red"}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST","DICT")
    RETURN_NAMES = ("Color_list","Color_dict")
    FUNCTION = "check"
    def check(self,Color_data,color_value):
        pass


class ColorData_HSV_Selection: #待开发
    ...


NODE_CLASS_MAPPINGS = {
    #WJNode/Color
    "load_color_config": load_color_config,
    "filter_DensePose_color": filter_DensePose_color,
    "color_segmentation": color_segmentation,
    "color_segmentation_v2": color_segmentation_v2,

    "load_ColorName_config": load_ColorName_config,
    #"ColorData_HSV_Capture": ColorData_HSV_Capture,
    "Color_check_Name": Color_check_Name,
    "Color_Data_Break": Color_Data_Break,
}
