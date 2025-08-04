import torch
import numpy as np


CATEGORY_NAME = "WJNode/ImageEdit/MaskEdit"


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


class coords_select_mask: #开发中
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
    RETURN_TYPES = ("MASK","MASK","LIST")
    RETURN_NAMES = ("select_mask", "mask_preview", "select_coords")
    FUNCTION = "coords_select"
    def coords_select(self,mask,point_coords):
        mask_preview = torch.zeros((1,*mask.shape[1:]),dtype=torch.float)
        if mask.dim() == 2: 
            mask = mask.repeat(1,1,1)
        mask_bool = torch.round(mask).bool()

        # 检查坐标数据是否符合规范
        n = self.depth(point_coords)
        try:
            point_coords = torch.tensor(point_coords).round().int()
        except:
            print("Error-coords_select: Single object multi-point coordinates are not currently supported!")
            print("Error-coords_select: 暂不支持单物体多点坐标！")
            return (None,mask_preview,None)
        if n == 3:
            point_coords = torch.squeeze(point_coords)
        elif n != 2 or point_coords.shape[-1] != 2:
            print("Error-coords_select: Coordinate data must be an xy array with a depth of 2 or 3")
            print("Error-coords_select: 坐标数据深度必须为2或3的xy数组,且每个物体只有一个坐标")
            return (None,mask_preview,None)

        # 绘制点用于预览坐标
        for i in range(len(point_coords)):
            mask_preview[0,point_coords[i][1],point_coords[i][0]] = 1.0
        mask_preview = mask_preview.unsqueeze(0)
        # 扩大点
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        mask_preview = torch.nn.functional.conv2d(mask_preview, kernel, padding=1, stride=1)
        mask_preview = mask_preview.squeeze()
        print(f"************2{mask_preview.shape}")
        return (None,mask_preview,None)

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
                "smooth":("BOOLEAN",{"default":True}),
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
    def mask_0_1(self, smooth, min_target, max_target, min_result, max_result, clamp_min, clamp_max, image = None, mask = None):
        if smooth:
            self.min_result = min_result/255.0
            self.max_result = max_result/255.0
            self.clamp_min = clamp_min/255.0
            self.clamp_max = clamp_max/255.0
        else:
            self.min_result = min_result
            self.max_result = max_result
            self.clamp_min = clamp_min
            self.clamp_max = clamp_max
        self.smooth = smooth

        if mask is not None:
            mask = torch.nan_to_num(mask)
            if min_target == -1: 
                min_target = float(torch.min(mask))
            if max_target == 256:
                max_target = float(torch.max(mask))
            mask = self.data_clamp(mask, min_target, max_target)
        if image is not None:
            image = torch.nan_to_num(image)
            if min_target == -1: 
                min_target = float(torch.min(image))
            if max_target == 256: 
                max_target = float(torch.max(image))
            image = self.data_clamp(image, min_target, max_target)

        if image is None and mask is not None:
            image = torch.ones((*mask.shape,3), dtype=torch.float)
        elif image is not None and mask is None:
            mask = torch.zeros(image.shape[:-1], dtype=torch.float)
        elif image is None and mask is None:
            print("Error-mask_line_mapping: At least one mask and image must be inputted")
        return (image, mask)
    
    def data_clamp(self, data, min_target, max_target):
        if not self.smooth:
            data = (data*255.0).int()
            if isinstance(min_target, float):
                min_target = int(min_target*255.0)
            if isinstance(max_target, float):
                max_target = int(max_target*255.0)
        else:
            if isinstance(min_target, int):
                min_target = min_target/255.0
            if isinstance(max_target, int):
                max_target = max_target/255.0
        data = (data - min_target) / (max_target - min_target)
        data = data * (self.max_result - self.min_result) + self.min_result
        data = torch.clamp(data, self.clamp_min, self.clamp_max)
        if not self.smooth:
            data = data/255.0
        return data


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

        #check cv2
        if algorithm == "cv2":
            try:
                import cv2
            except:
                print("prompt-mask_and_mask_math: cv2 is not installed, Using Torch")
                print("prompt-mask_and_mask_math: cv2 未安装, 使用torch")
                algorithm = "torch"

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


NODE_CLASS_MAPPINGS = {
    #WJNode/MaskEdit
    "mask_select_mask": mask_select_mask,   
    # "coords_select_mask": coords_select_mask,
    "mask_line_mapping": mask_line_mapping,
    "mask_and_mask_math": mask_and_mask_math,
}
