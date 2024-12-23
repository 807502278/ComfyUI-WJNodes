from PIL import Image, ImageOps
from io import BytesIO
import torch
import numpy as np
import os
from PIL import Image, ImageOps
import json
import re
import ast

import folder_paths


CATEGORY_NAME = "WJNode/MaskEdit"

class load_color_config:
    DESCRIPTION = """
    
    """
    config_file_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-WJNodes/seg_color")
    @classmethod
    def INPUT_TYPES(s):
        config_file_list = []
        for filename in os.listdir(s.config_file_path):
            if filename.endswith('.json'):
                config_file_list.append(filename)
        return {
            "required": {
                "config_file":(config_file_list,{"default":"ADE20K.json"}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST","DICT")
    RETURN_NAMES = ("Color_list","Color_dict")
    FUNCTION = "color_config"
    def color_config(self, config_file):
        with open(os.path.join(self.config_file_path, config_file)) as file:
            color_dict = json.load(file)
        color_data = list(color_dict.values())
        return ([color_data],color_dict)


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
        return {
            "required": {
                "image": ("IMAGE",),
                "skip_threshold":("FLOAT",{"default":0.05,"min":0,"max":1,"step":0.001,"display":"slider"}),
                "Color_dict":("DICT",),
                "merge_mask":("BOOLEAN",{"default":False}),
                "invert_mask":("BOOLEAN",{"default":False}),
                "output_type":(["all","custom",],{"default":"all"}),
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
        Color_list = []
        if output_type == "all":
            Color_list = [list(Color_dict.values()),]
        elif output_type == "custom":
            custom_keys = self.remove_trailing_comma(custom_keys)
            keys = re.split(r',\s*|，\s*', custom_keys)
            Color_list = [[Color_dict[key] for key in keys if key in Color_dict],]

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


class bbox_restore_mask:
    DESCRIPTION = """
    
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
            

class mask_select_mask:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask":("MASK",),
                "select_mask":("MASK",),
                "invert_select":("BOOLEAN",{"default":False}),
                "invert_select_mask":("BOOLEAN",{"default":False}),
                "merge_mask":("BOOLEAN",{"default":False}),
                "Overlap_threshold":("FLOAT",{"default":0.001,"min":0,"max":1,"step":0.001,"display":"slider"}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_select_mask"
    def mask_select_mask(self, mask, select_mask, invert_select, invert_select_mask, merge_mask, Overlap_threshold):
        if mask.dim() == 2:
            mask = mask.repeat(1,1,1)
        Overlap_threshold = int(Overlap_threshold * mask.shape[1] * mask.shape[2])+1

        mask = self.select_mask(mask, select_mask, invert_select, invert_select_mask, Overlap_threshold)
        if merge_mask:
            mask = self.merge_maks(mask)
        return (mask,)

    #select mask
    def select_mask(self, mask, select_mask, invert_select, invert_select_mask, Overlap_threshold):
        select_mask = torch.round(select_mask).bool()[0]
        mask_bool = torch.round(mask).bool()
        if invert_select_mask:
            select_mask = ~select_mask
        new_mask = torch.zeros((0,*mask.shape[1:])).bool()
        new_mask_i = torch.zeros((0,*mask.shape[1:])).bool()
        for i in range(mask.shape[0]):
            mask_temp = mask_bool[i] & select_mask
            if torch.sum(mask_temp) > Overlap_threshold:
                new_mask = torch.cat((new_mask, mask[i].repeat(1,1,1)), dim=0)
            else:
                new_mask_i = torch.cat((new_mask_i, mask[i].repeat(1,1,1)), dim=0)

        if new_mask.shape[0] == 0:
            new_mask = torch.zeros(mask.shape, dtype=torch.float)
        if new_mask_i.shape[0] == 0:
            new_mask_i = torch.zeros(mask.shape, dtype=torch.float)

        if invert_select:
            return new_mask_i
        else:
            return new_mask
        
    #merge mask
    def merge_maks(self, mask, to_bool=False):
        if to_bool:
            mask = mask.bool()
            for i in range(len(mask)-1):
                mask[0] = mask[0] | mask[i+1]
            mask = mask[0].repeat(1,1,1)
        else:
            for i in range(len(mask)-1):
                mask[0] = torch.clamp_max(mask[0] + mask[i+1], 1)
            mask = mask[0].repeat(1,1,1)
        return mask


class coords_select_mask:
    DESCRIPTION = """
    Under development...
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask":("MASK",),
                "point_coords":("LIST",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK","MASK","BOOLEAN","MASK")
    RETURN_NAMES = ("mask","unselected_mask","Selected","Preview")
    FUNCTION = "coords_select"
    def coords_select(self,mask,point_coords):
        mask_preview = torch.zeros((1,*mask.shape[1:]),dtype=torch.float)
        print(f"************0{point_coords}")
        if mask.dim() == 2: 
            mask = mask.repeat(1,1,1)
        mask_bool = torch.round(mask).bool()

        if self.depth(point_coords) == 3:
            point_coords = point_coords[0]

        if self.depth(point_coords) == 2:
            point_coords = torch.tensor(point_coords).round().int()
        else:
            print("Error-coords_select: The number of point_coords does not match the number of masks")
            return (None,None,None,mask_preview)
        print(f"************1{point_coords.shape}")
        print(f"************1{point_coords}")

        if point_coords.shape[-1] == 2:
            
            #预览点
            mask_preview = torch.zeros((1,*mask.shape[1:]),dtype=torch.float)
            for i in point_coords: # 绘制点
                mask_preview[0,i[0],i[1]] = 1.0
            mask_preview = mask_preview.unsqueeze(0)
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
            mask_preview = torch.nn.functional.conv2d(mask_preview, kernel, padding=1, stride=1)
            mask_preview = mask_preview.squeeze()
            print(f"************2{mask_preview.shape}")

            return (None,None,None,mask_preview)
        else:
            print("Error-coords_select: The number of point_coords does not match the number of masks")
            return (None,None,None,None)
    def depth(self,lst):
        if not isinstance(lst, list): return 0
        if not lst: return 1
        return 1 + max(self.depth(item) for item in lst)


class mask_line_mapping:
    DESCRIPTION = """
    Automatically detect the minimum value when min_target is -1
    Automatically detect the maximum value when max_target is 256
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_target":("INT",{"default":-1,"min":-1,"max":256,"step":1}),
                "max_target":("INT",{"default":256,"min":-1,"max":256,"step":1}),
                "min_result":("INT",{"default":0,"min":0,"max":255,"step":1,"display":"slider"}),
                "max_result":("INT",{"default":255,"min":0,"max":255,"step":1,"display":"slider"}),
                "clamp_min":("INT",{"default":0,"min":0,"max":255,"step":1,"display":"slider"}),
                "clamp_max":("INT",{"default":255,"min":0,"max":255,"step":1,"display":"slider"}),
            },
            "optional": {
                "image":("IMAGE",),
                "mask":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ("image","mask")
    FUNCTION = "mask_0_1"
    def mask_0_1(self, min_target, max_target, min_result, max_result, clamp_min, clamp_max, image = None, mask = None):
        min_result = min_result/255.0 ; max_result = max_result/255.0
        clamp_min = clamp_min/255.0 ; clamp_max = clamp_max/255.0

        if mask is not None:
            mask = torch.nan_to_num(mask)
            if min_target == -1: 
                min_target = float(torch.min(mask))
            else: 
                min_target = min_target/255.0
            if max_target == 256:
                max_target = float(torch.max(mask))
            else: 
                max_target = max_target/255.0
            mask = self.data_clamp(mask, min_target, max_target, min_result, max_result, clamp_min, clamp_max)
        if image is not None:
            image = torch.nan_to_num(image)
            if min_target == -1: 
                min_target = float(torch.min(image))
            else: 
                min_target = min_target/255.0
            if max_target == 256: 
                max_target = float(torch.max(image))
            else: 
                max_target = max_target/255.0
            image = self.data_clamp(image, min_target, max_target, min_result, max_result, clamp_min, clamp_max)

        if image is None and mask is not None:
            image = torch.zeros((*mask.shape,3), dtype=torch.float)
        elif image is not None and mask is None:
            mask = torch.zeros(image.shape[:-1], dtype=torch.float)
        elif image is None and mask is None:
            print("Error-mask_line_mapping: At least one mask and image must be inputted")

        return (image, mask)
    
    def data_clamp(self, data, min_target, max_target, min_result, max_result, clamp_min, clamp_max):
        data = (data - min_target) / (max_target - min_target)
        data = data * (max_result - min_result) + min_result
        data = torch.clamp(data, clamp_min, clamp_max)
        return data


<<<<<<< HEAD
class mask_and_mask_math:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1":("MASK",),
                "mask2":("MASK",),
                "operation":(["-","+","*","&"],{"+":"subtract"}),
                "algorithm":(["cv2","torch"],{"default":"cv2"}),
                "invert_mask1":("BOOLEAN",{"default":False}),
                "invert_mask2":("BOOLEAN",{"default":False}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_math"
    def mask_math(self, mask1, mask2, operation, algorithm, invert_mask1, invert_mask2):
        #invert mask
        if invert_mask1:
            mask1 = 1-mask1
        if invert_mask2:
            mask2 = 1-mask2

        #repeat mask
        if mask1.dim() == 2:
            mask1 = mask1.unsqueeze(0)
        if mask2.dim() == 2:
            mask2 = mask2.unsqueeze(0)
        if mask1.shape[0] == 1 and mask2.shape[0] != 1:
            mask1 = mask1.repeat(mask2.shape[0],1,1)
        elif mask1.shape[0] != 1 and mask2.shape[0] == 1:
            mask2 = mask2.repeat(mask1.shape[0],1,1)

        #algorithm
        if algorithm == "cv2":
            if operation == "-":
                return (self.subtract_masks(mask1, mask2),)
            elif operation == "+":
                return (self.add_masks(mask1, mask2),)
            elif operation == "*":
                return (self.multiply_masks(mask1, mask2),)
            elif operation == "&":
                return (self.and_masks(mask1, mask2),)
        elif algorithm == "torch":
            if operation == "-":
                return (torch.clamp(mask1 - mask2, min=0, max=1),)
            elif operation == "+":
                return (torch.clamp(mask1 + mask2, min=0, max=1),)
            elif operation == "*":
                return (torch.clamp(mask1 * mask2, min=0, max=1),)
            elif operation == "&":
                mask1 = torch.round(mask1).bool()
                mask2 = torch.round(mask2).bool()
                return (mask1 & mask2, )

    def subtract_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1

    def add_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def multiply_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def and_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1)
        cv2_mask2 = np.array(mask2)
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
        


=======
>>>>>>> 0120d03f64301cb0ac56ea8fb46e3f26aea694da
NODE_CLASS_MAPPINGS = {
    #WJNode/MaskEdit
    "LoadColorConfig": load_color_config,
    "ColorSegmentation": color_segmentation,
    "ColorSegmentation_v2": color_segmentation_v2,
    "BboxRestoreMask": bbox_restore_mask,
    "MaskSelectMask": mask_select_mask,
    "CoordsSelectMask": coords_select_mask,
    "MaskLineMapping": mask_line_mapping,
<<<<<<< HEAD
    "MaskAndMaskMath": mask_and_mask_math,
=======
>>>>>>> 0120d03f64301cb0ac56ea8fb46e3f26aea694da
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #WJNode/MaskEdit
    "LoadColorConfig": "Load Color Config",
    "ColorSegmentation": "Color Segmentation",
    "ColorSegmentation_v2": "Color Segmentation v2",
    "BboxRestoreMask": "Bbox Restore Mask",
    "MaskSelectMask": "Mask Select Mask",
    "CoordsSelectMask": "Coords Select Mask",
    "MaskLineMapping": "Mask Line Mapping",
<<<<<<< HEAD
    "MaskAndMaskMath": "Mask And Mask Math",
=======
>>>>>>> 0120d03f64301cb0ac56ea8fb46e3f26aea694da
}