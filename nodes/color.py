import torch
import os
import json
import re

import folder_paths
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
    """
    @classmethod
    def INPUT_TYPES(s):
        tag_mod = ["all tag","background tag","custom tag"]
        return {
            "required": {
                "image": ("IMAGE",),
                "skip_threshold":("FLOAT",{"default":0.05,"min":0,"max":1,"step":0.001,"display":"slider"}),
                "Color_dict":("DICT",),
                "merge_mask":("BOOLEAN",{"default":False}),
                "invert_mask":("BOOLEAN",{"default":False}),
                "output_type":(tag_mod,{"default":tag_mod[0]}),
                "custom_keys":("STRING",{"default":"wall,sky,floor,ceiling,windowpane,traffic","multiline": True}),
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
    def separate_color_blocks(self, image, skip_threshold, Color_dict, merge_mask, 
                              invert_mask, invert_select_mask, invert_select, 
                              output_type, custom_keys, select_mask=None):
        #select batch and to 3 dimensions

        #Empty data detection
        if Color_dict == {}: return self.none_data(image,invert_mask)

        Color_list = []
        if output_type == "all tag":
            Color_list = [list(Color_dict.values()),]
        elif output_type == "custom tag":
            custom_keys = self.remove_trailing_comma(custom_keys)
            if custom_keys == "": #自定义tag为空直接返回结果
                return self.none_data(image,invert_mask)
            keys = re.split(r',\s*|，\s*', custom_keys)
            if set(keys) & set(Color_dict.keys()) == set():#自定义tag无效直接返回结果
                return self.none_data(image,invert_mask)
            Color_list = [[Color_dict[key] for key in keys if key in Color_dict],]
        elif output_type == "background tag":
            keys = ['wall','sky','floor','ceiling','windowpane','traffic','background','back']
            Color_list = [[Color_dict[key] for key in keys if key in Color_dict],]
        else:
            raise ValueError('Error:Invalid tag output mode!')

        mask = torch.zeros((0,*image.shape[1:-1]), dtype=torch.float)
        #select batch and to 3 dimensions
        try:
            n=image.shape[0]
            if n != 1:
                if len(Color_list) != n and len(Color_list) != 1:
                    print("Error-color_segmentation_v2: The number of color_list does not match the number of images")
                elif len(Color_list) == 1:
                    Color_list = Color_list[0]
                    for i in range(n):
                        image_temp = image[i]
                        mask_temp = self.run_color_segmentation(image_temp, Color_list, skip_threshold, invert_mask, select_mask, invert_select_mask, invert_select, True)
                        mask = torch.cat((mask, mask_temp), dim=0)
                else:
                    for i in range(n):
                        image_temp = image[i]
                        Color_list_temp = Color_list[i]
                        mask_temp = self.run_color_segmentation(image_temp, Color_list_temp, skip_threshold, invert_mask, select_mask, invert_select_mask, invert_select, True)
                        mask = torch.cat((mask, mask_temp), dim=0)
            else:
                mask = self.run_color_segmentation(image[0], Color_list, skip_threshold, invert_mask, select_mask, invert_select_mask, invert_select, merge_mask)
        except:
            print("warn-color_segmentation: The selected batch exceeds the input batch and has been changed to the 0th batch")
            mask = self.run_color_segmentation(image[0], Color_list[0], skip_threshold, invert_mask, select_mask, invert_select_mask, invert_select, merge_mask)

        return (mask.float(),)
    
    def remove_trailing_comma(self,s): # 移除字符串末尾的逗号、空格或中文逗号
        if s.endswith((',',"，"," ")):
            return self.remove_trailing_comma(s[:-1])
        return s
    
    def run_color_segmentation(self, image, Color_list, skip_threshold, invert_mask, select_mask, invert_select_mask, invert_select, merge_mask):
        mask = self.color_to_mask(image, Color_list, skip_threshold)
        mask = self.handle_mask(mask, invert_mask)
        if select_mask is not None:
            mask = self.select_mask(mask, select_mask, invert_select_mask, invert_select)
        if merge_mask:
            mask = self.merge_maks(mask)
        return mask
    
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
    
    #空数据则直接输出结果
    def none_data(self,image,invert_mask):
        mask = None
        if invert_mask: mask = torch.ones(image.shape[0:-1])
        else: mask = torch.zeros(image.shape[0:-1])
        return (mask,)
     

NODE_CLASS_MAPPINGS = {
    #WJNode/Color
    "load_color_config": load_color_config,
    "filter_DensePose_color": filter_DensePose_color,
    "color_segmentation": color_segmentation,
    "color_segmentation_v2": color_segmentation_v2,
}