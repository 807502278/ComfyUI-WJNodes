import torch
import numpy as np

from ..moduel.str_edit import str_edit

# ------------------GetData nodes------------------
CATEGORY_NAME = "WJNode/Batch"


class Select_Images_Batch:
    DESCRIPTION = """
    返回指定批次编号处的图像(第1张编号为1,可以任意重复和排列组合)
    说明：
        超出范围的编号将被忽略，若输入为空则一个都不选，
        若所有的编号超出范围则返回None，可识别中文逗号。
        若只输入图像，图像批次正常输出，遮罩输出使用对应图像批次的遮罩，
            若图像没有A通道则输出原图大小的纯白遮罩且批次与选择/排除的批次对应
        若只输入遮罩，遮罩批次正常输出，图像输出为将对应遮罩批次转为的图像
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
        
        select_img, exclude_img = None, None
        select_mask, exclude_mask = None, None
        
        # 处理图像批次
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
                
                # 若只输入图像，检查是否有A通道，为遮罩输出做准备
                if masks is None:
                    # 检查图像是否有A通道，默认RGB图像shape为(n,h,w,3)，RGBA为(n,h,w,4)
                    if images.shape[-1] == 4:  # 有A通道，将A通道作为遮罩
                        # 提取Alpha通道并转为正确的遮罩格式(批次,高,宽)
                        alpha_channel = images[..., 3]
                        select_mask = alpha_channel[torch.tensor(s_i, dtype=torch.int)]
                        exclude_mask = alpha_channel[torch.tensor(e_i, dtype=torch.int)]
                    else:  # 没有A通道，创建纯白遮罩
                        h, w = images.shape[1], images.shape[2]
                        # 创建正确维度的遮罩(批次,高,宽)
                        select_mask = torch.ones((len(s_i), h, w), dtype=torch.float32)
                        exclude_mask = torch.ones((len(e_i), h, w), dtype=torch.float32)
        
        # 处理遮罩批次
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
                
                # 若只输入遮罩，将遮罩转为图像
                if images is None:
                    # 将遮罩转为灰度图像(遮罩维度为(批次,高,宽)，需要转为(批次,高,宽,3))
                    h, w = masks.shape[1], masks.shape[2]
                    # 扩展维度后复制到三个通道
                    select_mask_expanded = select_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
                    exclude_mask_expanded = exclude_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
                    select_img = select_mask_expanded
                    exclude_img = exclude_mask_expanded

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


class SelectBatch_paragraph: # ******************开发中
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
 

class Batch_Average: # ******************开发中
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


NODE_CLASS_MAPPINGS = {
    "Select_Images_Batch": Select_Images_Batch,
    "SelectBatch_paragraph": SelectBatch_paragraph,
    "Batch_Average": Batch_Average,
}

