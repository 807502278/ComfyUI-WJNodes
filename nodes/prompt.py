import json
import os
import random
import numpy as np
from typing import List, Dict, Tuple

import folder_paths
from ..moduel.str_edit import str_edit
from ..moduel.color_utils import distance_color,convert_color,convert_rgb_1_255
from ..moduel.list_edit import random_select
from ..config.color_name.color_data_edit import select_region,NameSelect_ColorData

CATEGORY_NAME = "WJNode/Prompt"

class Random_Select_Prompt:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Prompt":("STRING",{"default":"","multiline": True}),
                "select_number":("INT",{"default":1,"min":1,"max":4096,"step":1}),
                "Original_data_deduplication":("BOOLEAN",{"default":False}),
                "allow_duplicates":("BOOLEAN",{"default":False}),
                "keep_order":("BOOLEAN",{"default":True}),
                "random_seed":("INT",{"default":1,"min":0,"max":99999999,"step":1})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("Prompt","Prompt_list")
    FUNCTION = "random_tag"
    def random_tag(self,Prompt,select_number,Original_data_deduplication,allow_duplicates,keep_order,random_seed):
        #输入检查
        str_list = []
        if isinstance(Prompt,str):
            Prompt = Prompt.replace(', ', ',')
            Prompt = Prompt.replace('，', ',')
            Prompt = Prompt.replace('， ', ',')
            str_list = Prompt.split(",")
        elif isinstance(Prompt,list) and isinstance(Prompt[0],str):
            str_list = Prompt
        elif isinstance(Prompt,(tuple,set)) and isinstance(Prompt[0],str):
            str_list = Prompt.tolist()
        else :
            raise TypeError("Error:Prompt data type error, can only input string or string list !")
        #...
        if Original_data_deduplication:
            str_list = list(set(str_list))
        str_list = random_select(str_list, select_number, random_seed, allow_duplicates, keep_order)
        return (str_edit.list_to_str(str_list),str_list)


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
    #WJNode/Prompt
    "Random_Select_Prompt": Random_Select_Prompt,
    "load_ColorName_config": load_ColorName_config,
    #"ColorData_HSV_Capture": ColorData_HSV_Capture,
    "Color_check_Name": Color_check_Name,
    "Color_Data_Break": Color_Data_Break,
}