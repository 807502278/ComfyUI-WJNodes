import torch
import math
import copy

CATEGORY_NAME = "WJNode/Image_Blend"

# 默认设置
default_options = {
    # 全局参数
    "Global":{
        "Switch_image": False, #交换back和surface图像，不交换编辑参数
        "Switch_options": False, #交换back和surface的编辑参数，不交换图像
    },
    #画布参数，在back和surface都被移动后露出的像素，任何时候都不会移动但是可以改变大小
    "Canvas":{
        "size":None, #画布大小为[高，宽]
        "Solid_Color":True, #画布是否使用纯色，False使用image
        "color":"#ffffff", #画布颜色
        "image":None #画布默认纹理
    },
    #back背景图像编辑参数，此参数仅对back输入生效，除非Switch_options为true
    "back":{
        #是否启用编辑back
        "enable":False, 
        # 基础设置 与上一层图像的叠加方式
        "mask_mode": "normal",      # normal, multiply, screen
        "blend_mode": "normal",     # normal, multiply, screen, overlay等
        "opacity": 1.0,
        # 坐标设置
        "x": 0,
        "y": 0,
        "scale": 1.0,
        "Center": False, #为ture时仅改变计算方式不移动图像
        "flip_x": False,
        "flip_y": False,
        "limit": False, #是否限制移动时不超过画布
        # 缩放设置
        "ConditionalScale": "NotUsed", #对scale/scale_X/scale_Y/ScaleMethod生效
        "ScaleMethod": "long", # 含DoNot_AutoScale等选项
        "scale_X": 1.0,
        "scale_Y": 1.0,
        # 其它设置
        "flip_A": False, #翻转A通道，若输入的3通道False时输出全白，ture输出全黑
        "Ignore_Input": False,  #忽略输入，当back/surface/MaskMix全部忽略时输出画布并警告
        "batch_mode": "auto", #批次设置
    },
    #surface背景图像编辑参数，此参数仅对surface输入生效，除非Switch_options为true
    "surface":{
        #是否启用编辑surface
        "enable":True, 
        # 基础设置 与上一层图像的叠加方式
        "mask_mode": "normal",      # normal, multiply, screen
        "blend_mode": "normal",     # normal, multiply, screen, overlay等
        "opacity": 1.0,
        # 坐标设置
        "x": 0,
        "y": 0,
        "scale": 1.0,
        "Center": False, #为ture时仅改变计算方式不移动图像
        "flip_x": False,
        "flip_y": False,
        "limit": False, #是否限制移动时不超过画布
        # 缩放设置
        "ConditionalScale": "NotUsed", #对scale/scale_X/scale_Y/ScaleMethod生效
        "ScaleMethod": "long", # 含DoNot_AutoScale等选项
        "scale_X": 1.0,
        "scale_Y": 1.0,
        # 其它设置
        "flip_A": False, #翻转A通道，若输入的3通道False时输出全白，ture输出全黑
        "Ignore_Input": False,  #忽略输入，当back/surface/MaskMix全部忽略时输出画布并警告
        "batch_mode": "auto", #批次设置
    },
    #MaskMix背景图像编辑参数，此参数仅对MaskMix输入生效
    "MaskMix":{
        #是否启用编辑MaskMix
        "enable":True,
        # 基础设置-叠加方式(对MaskMix无效，仅方便参数更新)
        "mask_mode": "normal",      # normal, multiply, screen
        "blend_mode": "normal",     # normal, multiply, screen, overlay等
        "opacity": 1.0,
        # 坐标设置
        "x": 0,
        "y": 0,
        "scale": 1.0,
        "Center": False, #为ture时仅改变计算方式不移动图像
        "flip_x": False,
        "flip_y": False,
        "limit": False, #是否限制移动时不超过画布
        # 缩放设置
        "ConditionalScale": "NotUsed", #对scale/scale_X/scale_Y/ScaleMethod生效
        "ScaleMethod": "long", # 含DoNot_AutoScale等选项
        "scale_X": 1.0,
        "scale_Y": 1.0,
        # 其它设置
        "flip_A": False, #翻转A通道，若输入的3通道False时输出全白，ture输出全黑
        "Ignore_Input": False,  #忽略输入，当back/surface/MaskMix全部忽略时输出画布并警告
        # 新增遮罩设置
        "use_mask": "auto",  # auto, none, mask, surface_alpha, back_alpha
        "output_mask": "auto",  # auto, none, white, black, mask, surface_alpha, back_alpha, mixed
        "batch_mode": "auto", #批次设置
    },
}


def update_options(s_add, s_orig = None, key = None):
        # 初始化设置
        if s_orig is None: s_orig = copy.deepcopy(default_options)
        #默认编辑"back","surface","MaskMix"设置
        if key is None: 
            key=["back","surface","MaskMix"]
            for i in key: #更新设置
                if s_orig[i]["enable"]: 
                    s_orig[i].update(s_add)
        #编辑Global或Canvas
        else:
            s_orig[key].update(s_add)
        return (s_orig,)


class ImageCompositeMask_Adv:
    DESCRIPTION = """
    本体ImageCompositeMasked节点(图像混合)的超级变态无敌诡异强化版：
    增加功能
    1：使其支持4通道的图像混合，支持遮罩混合
    2：输出图像叠加的RGBA，RGB，A
    3：批次兼容，输入不同批次时自动应用规则，可叠加视频和视频mask
    4：多种高级设置
    各种输入情况下应用的规则：
    1：当MaskMix不为空，且surface/back都不为空时
        1.1-surface/back同时没有A通道时：按设置正常叠加图像，输出的A通道为白色
        1.2.surface/back其中一个有A通道时：按设置正常RGB模式叠加图像，输出的A通道即为这个A通道
        1.3.surface/back都有A通道时：按设置通过MaskMix混合两个RGBA图像并输出
    2：当MaskMix为空时，且surface/back都不为空时
        2.1.surface/back都没有A通道时：直接按设置叠加surface到back，输出的A通道为白色
        2.2.surface/back其中一个有A通道时：将其A通道作为MaskMix来叠加图像并作为输出的A通道(含移动缩放)
        2.3.surface和back都为RGBA(4通道)时：将surface的A通道作为MaskMix来叠加图像并作为输出的A通道
    3：当surface/back其中一个为空
        3.1.MaskMix为空，surface不为空：背景画布大小为surface，若无A通道输出的A通道为白色
            按正常设置将surface叠加白背景上(可选Composite_Coordinate节点设置其它背景)
        3.2.MaskMix为空，back不为空：直接输出back，若无A通道输出的A通道为白色
        3.3.MaskMix不为空，surface/back其中一个不为空：将MaskMix作为不为空的surface/back的A通道输出
    4：当surface/back都为空
        4.1.MaskMix不为空：将MaskMix复制到RGBA并输出
        4.2.MaskMix为空：报错-需要有输入
    5：当MaskMix/surface/back 3者中只有1个输入为多批次n时：
        5.1.将其它两个的单批次叠加到n批次后，按第1/2/3/4条规则叠加
    6：当MaskMix/surface/back 3者中有2个为多批次n和m时：
        6.1.如果n=m：则将单张的叠加到n批次后，按第1/2/3/4条规则叠加
        6.2.如果n!=m且n和m其中一个不能被另一个整除：报错-需要这2个批次数量相同
    7：当MaskMix/surface/back 3者都为多批次时：
        7.1.若所有批次数相同，按第1/2/3/4条规则叠加
        7.2.若所有批次数有1个不同或都不相同，：报错-需要这3个批次数量相同
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "back": ("IMAGE",),
                "surface": ("IMAGE",),
                "MaskMix": ("MASK",),
                "Options": ("Composite_Basic",),
            },
            "hidden": {
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("RGBA", "RGB", "A", "Help")
    FUNCTION = "composite"
    CATEGORY = CATEGORY_NAME

    def __init__(self):
        # 为每个实例创建独立的设置副本
        self.instance_options = copy.deepcopy(default_options)

    def composite(self, back=None, surface=None, MaskMix=None, Options=None):
        # 使用实例的独立设置副本
        composite_options = copy.deepcopy(self.instance_options)
        if Options is not None:
            composite_options = copy.deepcopy(Options)

        # 初始化画布数据
        canvas_size = [] #初始化画布大小
        canvas_image = None
        if back is not None: canvas_image = back
        elif surface is not None: canvas_image = surface
        elif MaskMix is not None: canvas_image = torch.stack((MaskMix,MaskMix,MaskMix),dim=-1)
        else: raise ValueError("Error：请输入至少一个图像！")
        if canvas_image.dim() == 4: canvas_image = back[...,0:-1] #画布去掉A通道
        canvas_size = canvas_image.shape[1:3] 

        # 初始化画布设置
        if composite_options["Canvas"]["size"] is None: composite_options["Canvas"]["size"] = canvas_size
        if composite_options["Canvas"]["image"] is None: composite_options["Canvas"]["image"] = canvas_image

        # 应用Global设置
        if composite_options["Global"]["Switch_image"] and back is not None and surface is not None:
            back, surface = surface, back
        if composite_options["Global"]["Switch_options"]:
            # 交换back和surface的设置
            back_settings = composite_options["back"].copy()
            surface_settings = composite_options["surface"].copy()
            composite_options["back"] = surface_settings
            composite_options["surface"] = back_settings
        
        # 处理Alpha通道翻转
        if back is not None and back.shape[-1] == 4 and composite_options["back"]["flip_A"]:
            back_rgb = back[..., :3]
            back_alpha = 1.0 - back[..., 3:4]
            back = torch.cat([back_rgb, back_alpha], dim=-1)
        if surface is not None and surface.shape[-1] == 4 and composite_options["surface"]["flip_A"]:
            surface_rgb = surface[..., :3]
            surface_alpha = 1.0 - surface[..., 3:4]
            surface = torch.cat([surface_rgb, surface_alpha], dim=-1)
        if MaskMix is not None and composite_options["MaskMix"]["flip_A"]:
            MaskMix = 1.0 - MaskMix
        
        # 检查是否需要忽略输入
        if composite_options["back"]["Ignore_Input"]:
            back = None
        if composite_options["surface"]["Ignore_Input"]:
            surface = None
        if composite_options["MaskMix"]["Ignore_Input"]:
            MaskMix = None
        
        # 如果所有输入都被忽略，使用画布
        if back is None and surface is None and MaskMix is None:
            if composite_options["Canvas"]["Solid_Color"]:
                # 创建纯色画布
                color = composite_options["Canvas"]["color"]
                # 将颜色字符串转换为RGB值
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0
                canvas = torch.ones((1, canvas_size[0], canvas_size[1], 4), dtype=torch.float32)
                canvas[..., 0] = r
                canvas[..., 1] = g
                canvas[..., 2] = b
                canvas[..., 3] = 1.0
                return (canvas, canvas[..., :3], canvas[...,3:4], self.__class__.DESCRIPTION)
            else:
                # 使用画布图像
                canvas = composite_options["Canvas"]["image"]
                return (canvas, canvas[..., :3], canvas[...,3:4].squeeze(-1), self.__class__.DESCRIPTION)

        # 应用批次处理设置
        batch_sizes = []
        if back is not None: batch_sizes.append(back.shape[0])
        if surface is not None: batch_sizes.append(surface.shape[0])
        if MaskMix is not None: batch_sizes.append(MaskMix.shape[0])
        
        # 如果有多个批次，先应用批次处理设置
        if len(set(batch_sizes)) > 1 or max(batch_sizes) > 1:
            # 应用back的批次模式
            if back is not None and back.shape[0] > 1:
                back_mode = composite_options["back"]["batch_mode"]
                if back_mode == "flatten" and back.shape[0] > 1:
                    # 将所有批次图像按A通道叠加为单张
                    if back.shape[-1] == 4:
                        # 有A通道，按A通道权重混合
                        a_weight = back[..., 3:4]
                        weighted_rgb = back[..., :3] * a_weight
                        summed_rgb = weighted_rgb.sum(dim=0, keepdim=True)
                        summed_a = a_weight.sum(dim=0, keepdim=True).clamp(0, 1)
                        back = torch.cat((summed_rgb / summed_a.clamp(min=1e-5), summed_a), dim=-1)
                    else:
                        # 无A通道，简单平均
                        back = back.mean(dim=0, keepdim=True)
                elif back_mode == "first":
                    back = back[0:1]  # 使用第一张
                elif back_mode == "last":
                    back = back[-1:] # 使用最后一张
                elif back_mode == "match_surface" and surface is not None:
                    # 匹配到surface的数量
                    if back.shape[0] != surface.shape[0]:
                        back = self.repeat_to_batch_size(back, surface.shape[0])
                elif back_mode == "match_mask" and MaskMix is not None:
                    # 匹配到MaskMix的数量
                    if back.shape[0] != MaskMix.shape[0]:
                        back = self.repeat_to_batch_size(back, MaskMix.shape[0])
            
            # 应用surface的批次模式
            if surface is not None and surface.shape[0] > 1:
                surface_mode = composite_options["surface"]["batch_mode"]
                if surface_mode == "flatten" and surface.shape[0] > 1:
                    # 将所有批次图像按A通道叠加为单张
                    if surface.shape[-1] == 4:
                        # 有A通道，按A通道权重混合
                        a_weight = surface[..., 3:4]
                        weighted_rgb = surface[..., :3] * a_weight
                        summed_rgb = weighted_rgb.sum(dim=0, keepdim=True)
                        summed_a = a_weight.sum(dim=0, keepdim=True).clamp(0, 1)
                        surface = torch.cat((summed_rgb / summed_a.clamp(min=1e-5), summed_a), dim=-1)
                    else:
                        # 无A通道，简单平均
                        surface = surface.mean(dim=0, keepdim=True)
                elif surface_mode == "first":
                    surface = surface[0:1]  # 使用第一张
                elif surface_mode == "last":
                    surface = surface[-1:] # 使用最后一张
                elif surface_mode == "match_back" and back is not None:
                    # 匹配到back的数量
                    if surface.shape[0] != back.shape[0]:
                        surface = self.repeat_to_batch_size(surface, back.shape[0])
                elif surface_mode == "match_mask" and MaskMix is not None:
                    # 匹配到MaskMix的数量
                    if surface.shape[0] != MaskMix.shape[0]:
                        surface = self.repeat_to_batch_size(surface, MaskMix.shape[0])
            
            # 应用MaskMix的批次模式
            if MaskMix is not None and MaskMix.shape[0] > 1:
                mask_mode = composite_options["MaskMix"]["batch_mode"]
                if mask_mode == "flatten" and MaskMix.shape[0] > 1:
                    # 将所有批次遮罩叠加为单张
                    MaskMix = MaskMix.mean(dim=0, keepdim=True)
                elif mask_mode == "first":
                    MaskMix = MaskMix[0:1]  # 使用第一张
                elif mask_mode == "last":
                    MaskMix = MaskMix[-1:] # 使用最后一张
                elif mask_mode == "match_back" and back is not None:
                    # 匹配到back的数量
                    if MaskMix.shape[0] != back.shape[0]:
                        MaskMix = self.repeat_to_batch_size(MaskMix, back.shape[0])
                elif mask_mode == "match_surface" and surface is not None:
                    # 匹配到surface的数量
                    if MaskMix.shape[0] != surface.shape[0]:
                        MaskMix = self.repeat_to_batch_size(MaskMix, surface.shape[0])
        
        # 处理批次兼容性
        batch_sizes = []
        if back is not None: batch_sizes.append(back.shape[0])
        if surface is not None: batch_sizes.append(surface.shape[0])
        if MaskMix is not None: batch_sizes.append(MaskMix.shape[0])
        
        if len(batch_sizes) > 0:
            max_batch = max(batch_sizes)
            # 检查批次数量是否匹配
            if len(set(batch_sizes)) > 1:
                # 检查是否可以被整除
                for size in batch_sizes:
                    if size != max_batch and max_batch % size != 0:
                        raise ValueError("Error：批次数量不匹配，需要这3个批次数量相同或可被整除")
            # 扩展批次
            if back is not None: back = self.repeat_to_batch_size(back, max_batch)
            if surface is not None: surface = self.repeat_to_batch_size(surface, max_batch)
            if MaskMix is not None: MaskMix = self.repeat_to_batch_size(MaskMix, max_batch)

        # 处理surface的缩放和移动
        if surface is not None:
            # 获取surface的尺寸
            b, h_s, w_s, c = surface.shape
            h_b, w_b = canvas_size

            # 计算缩放
            base_scale = composite_options["surface"]["scale"]  # 来自Composite_Coordinate的基础缩放
            scale_x = composite_options["surface"]["scale_X"]  # 来自Composite_Scale的X缩放
            scale_y = composite_options["surface"]["scale_Y"]  # 来自Composite_Scale的Y缩放
            
            # 应用缩放方法
            if composite_options["surface"]["ScaleMethod"] != "DoNot_AutoScale":
                # 计算目标尺寸
                if composite_options["surface"]["ScaleMethod"] == "long":
                    auto_scale = min(w_b/w_s, h_b/h_s)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["surface"]["ScaleMethod"] == "long_fill":
                    auto_scale = max(w_b/w_s, h_b/h_s)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["surface"]["ScaleMethod"] == "short":
                    auto_scale = max(w_b/w_s, h_b/h_s)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["surface"]["ScaleMethod"] == "short_crop":
                    auto_scale = max(w_b/w_s, h_b/h_s)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["surface"]["ScaleMethod"] == "stretch":
                    scale_x = w_b/w_s
                    scale_y = h_b/h_s
                elif composite_options["surface"]["ScaleMethod"] == "average":
                    auto_scale = (w_b/w_s + h_b/h_s) / 2
                    scale_x *= auto_scale
                    scale_y *= auto_scale

            # 应用条件缩放
            if composite_options["surface"]["ConditionalScale"] != "NotUsed":
                current_scale = base_scale
                if composite_options["surface"]["ConditionalScale"] == "max_if" and current_scale < 1:
                    base_scale = 1
                elif composite_options["surface"]["ConditionalScale"] == "min_if" and current_scale > 1:
                    base_scale = 1

            # 应用缩放 - 修改这里，使Composite_Coordinate的缩放与Composite_Scale的缩放正确叠加
            if base_scale != 1 or scale_x != 1 or scale_y != 1:
                # 计算最终的缩放比例
                final_scale_x = base_scale * scale_x  # 将scale_x与base_scale相乘
                final_scale_y = base_scale * scale_y  # 将scale_y与base_scale相乘
                new_h = int(h_s * final_scale_y)
                new_w = int(w_s * final_scale_x)
                surface = torch.nn.functional.interpolate(
                    surface.permute(0, 3, 1, 2),
                    size=(new_h, new_w),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

            # 应用翻转
            if composite_options["surface"]["flip_x"]:
                surface = torch.flip(surface, [2])
            if composite_options["surface"]["flip_y"]:
                surface = torch.flip(surface, [1])

            # 计算位置
            x = composite_options["surface"]["x"]
            y = composite_options["surface"]["y"]
            if composite_options["surface"]["Center"]:
                x += (w_b - surface.shape[2]) // 2
                y += (h_b - surface.shape[1]) // 2

            # 限制移动范围
            if composite_options["surface"]["limit"]:
                x = max(0, min(x, w_b - surface.shape[2]))
                y = max(0, min(y, h_b - surface.shape[1]))

        # 处理MaskMix的缩放和移动
        if MaskMix is not None:
            # 获取MaskMix的尺寸
            b, h_m, w_m = MaskMix.shape
            h_b, w_b = canvas_size

            # 计算缩放
            base_scale = composite_options["MaskMix"]["scale"]  # 来自Composite_Coordinate的基础缩放
            scale_x = composite_options["MaskMix"]["scale_X"]  # 来自Composite_Scale的X缩放
            scale_y = composite_options["MaskMix"]["scale_Y"]  # 来自Composite_Scale的Y缩放
            
            # 应用缩放方法
            if composite_options["MaskMix"]["ScaleMethod"] != "DoNot_AutoScale":
                # 计算目标尺寸
                if composite_options["MaskMix"]["ScaleMethod"] == "long":
                    auto_scale = min(w_b/w_m, h_b/h_m)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["MaskMix"]["ScaleMethod"] == "long_fill":
                    auto_scale = max(w_b/w_m, h_b/h_m)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["MaskMix"]["ScaleMethod"] == "short":
                    auto_scale = max(w_b/w_m, h_b/h_m)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["MaskMix"]["ScaleMethod"] == "short_crop":
                    auto_scale = max(w_b/w_m, h_b/h_m)
                    scale_x *= auto_scale
                    scale_y *= auto_scale
                elif composite_options["MaskMix"]["ScaleMethod"] == "stretch":
                    scale_x = w_b/w_m
                    scale_y = h_b/h_m
                elif composite_options["MaskMix"]["ScaleMethod"] == "average":
                    auto_scale = (w_b/w_m + h_b/h_m) / 2
                    scale_x *= auto_scale
                    scale_y *= auto_scale

            # 应用条件缩放
            if composite_options["MaskMix"]["ConditionalScale"] != "NotUsed":
                current_scale = base_scale
                if composite_options["MaskMix"]["ConditionalScale"] == "max_if" and current_scale < 1:
                    base_scale = 1
                elif composite_options["MaskMix"]["ConditionalScale"] == "min_if" and current_scale > 1:
                    base_scale = 1

            # 应用缩放 - 修改这里，使Composite_Coordinate的缩放与Composite_Scale的缩放正确叠加
            if base_scale != 1 or scale_x != 1 or scale_y != 1:
                # 计算最终的缩放比例
                final_scale_x = base_scale * scale_x  # 将scale_x与base_scale相乘
                final_scale_y = base_scale * scale_y  # 将scale_y与base_scale相乘
                new_h = int(h_m * final_scale_y)
                new_w = int(w_m * final_scale_x)
                MaskMix = torch.nn.functional.interpolate(
                    MaskMix.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear"
                ).squeeze(1)

            # 应用翻转
            if composite_options["MaskMix"]["flip_x"]:
                MaskMix = torch.flip(MaskMix, [2])
            if composite_options["MaskMix"]["flip_y"]:
                MaskMix = torch.flip(MaskMix, [1])

            # 计算位置
            x = composite_options["MaskMix"]["x"]
            y = composite_options["MaskMix"]["y"]
            if composite_options["MaskMix"]["Center"]:
                x += (w_b - MaskMix.shape[2]) // 2
                y += (h_b - MaskMix.shape[1]) // 2

            # 限制移动范围
            if composite_options["MaskMix"]["limit"]:
                x = max(0, min(x, w_b - MaskMix.shape[2]))
                y = max(0, min(y, h_b - MaskMix.shape[1]))

            # 移动MaskMix
            MaskMix = self.move_mask(MaskMix, x, y, w_b, h_b)

        # 创建结果画布
        if composite_options["Canvas"]["Solid_Color"]:
            # 创建纯色画布
            color = composite_options["Canvas"]["color"]
            r = int(color[1:3], 16) / 255.0
            g = int(color[3:5], 16) / 255.0
            b = int(color[5:7], 16) / 255.0
            result = torch.ones((max_batch, canvas_size[0], canvas_size[1], 4), dtype=torch.float32)
            result[..., 0] = r
            result[..., 1] = g
            result[..., 2] = b
            result[..., 3] = 1.0
        else:
            # 使用画布图像
            result = composite_options["Canvas"]["image"].clone()
            if result.shape[-1] == 3:
                result = torch.cat([result, torch.ones_like(result[..., :1])], dim=-1)

        # 处理back
        if back is not None:
            if back.shape[-1] == 3:
                back = torch.cat([back, torch.ones_like(back[..., :1])], dim=-1)
            result = self.blend_images(result, back, None, 0, 0, composite_options["back"]["blend_mode"])

        # 处理surface
        if surface is not None:
            if surface.shape[-1] == 3:
                surface = torch.cat([surface, torch.ones_like(surface[..., :1])], dim=-1)
            
            # 根据use_mask设置选择遮罩
            use_mask = composite_options["MaskMix"]["use_mask"]
            if use_mask == "auto":
                # 自动检测逻辑
                if MaskMix is not None:
                    blend_mask = MaskMix
                elif surface.shape[-1] == 4:
                    blend_mask = surface[..., 3:4]
                elif back is not None and back.shape[-1] == 4:
                    blend_mask = back[..., 3:4]
                else:
                    blend_mask = None
            elif use_mask == "none":
                blend_mask = None
            elif use_mask == "mask" and MaskMix is not None:
                blend_mask = MaskMix
            elif use_mask == "surface_alpha" and surface.shape[-1] == 4:
                blend_mask = surface[..., 3:4]
            elif use_mask == "back_alpha" and back is not None and back.shape[-1] == 4:
                blend_mask = back[..., 3:4]
            else:
                blend_mask = None

            result = self.blend_images(result, surface, blend_mask, x, y, composite_options["surface"]["blend_mode"])

        # 处理输出遮罩
        output_mask = composite_options["MaskMix"]["output_mask"]
        if output_mask == "auto":
            # 自动检测逻辑
            if MaskMix is not None:
                result_alpha = MaskMix
            elif surface is not None and surface.shape[-1] == 4:
                result_alpha = surface[..., 3:4].squeeze(-1)
            elif back is not None and back.shape[-1] == 4:
                result_alpha = back[..., 3:4].squeeze(-1)
            else:
                result_alpha = torch.ones_like(result[..., 0])
        elif output_mask == "none":
            result_alpha = torch.ones_like(result[..., 0])
        elif output_mask == "white":
            result_alpha = torch.ones_like(result[..., 0])
        elif output_mask == "black":
            result_alpha = torch.zeros_like(result[..., 0])
        elif output_mask == "mask" and MaskMix is not None:
            result_alpha = MaskMix
        elif output_mask == "surface_alpha" and surface is not None and surface.shape[-1] == 4:
            result_alpha = surface[..., 3:4].squeeze(-1)
        elif output_mask == "back_alpha" and back is not None and back.shape[-1] == 4:
            result_alpha = back[..., 3:4].squeeze(-1)
        elif output_mask == "mixed" and MaskMix is not None:
            # 混合surface和back的alpha通道
            if surface is not None and surface.shape[-1] == 4 and back is not None and back.shape[-1] == 4:
                result_alpha = self.blend_images(
                    back[..., 3:4],
                    surface[..., 3:4],
                    MaskMix,
                    x, y,
                    composite_options["surface"]["blend_mode"]
                ).squeeze(-1)
            else:
                result_alpha = MaskMix
        else:
            result_alpha = torch.ones_like(result[..., 0])

        # 分离RGB和Alpha通道
        result_rgb = result[..., :3]

        return (result, result_rgb, result_alpha, self.__class__.DESCRIPTION)
    
    def repeat_to_batch_size(self, tensor, batch_size, dim=0):
        """将张量在批次维度上重复到指定大小"""
        if tensor.shape[dim] > batch_size:
            return tensor.narrow(dim, 0, batch_size)
        elif tensor.shape[dim] < batch_size:
            repeats = [1] * tensor.dim()
            repeats[dim] = math.ceil(batch_size / tensor.shape[dim])
            return tensor.repeat(*repeats).narrow(dim, 0, batch_size)
        return tensor

    def blend_images(self, back, surface, mask, x, y, blend_mode="normal"):
        """根据混合模式和遮罩混合两个图像"""
        # 计算表面图像在背景上的位置
        b, h_s, w_s, c = surface.shape
        h_b, w_b = back.shape[1], back.shape[2]
        
        # 裁剪坐标确保在有效范围内
        x = max(0, min(x, w_b - 1))
        y = max(0, min(y, h_b - 1))
        
        # 计算重叠区域
        x_end = min(x + w_s, w_b)
        y_end = min(y + h_s, h_b)
        
        w_overlap = x_end - x
        h_overlap = y_end - y
        
        if w_overlap <= 0 or h_overlap <= 0:
            return back  # 无重叠区域
        
        # 提取重叠区域
        back_region = back[:, y:y_end, x:x_end]
        surface_region = surface[:, :h_overlap, :w_overlap]
        
        # 确保两个区域的尺寸一致
        if back_region.shape[1:3] != surface_region.shape[1:3]:
            # 调整surface_region以匹配back_region
            surface_region = torch.nn.functional.interpolate(
                surface_region.permute(0, 3, 1, 2),
                size=(back_region.shape[1], back_region.shape[2]),
                mode="bilinear"
            ).permute(0, 2, 3, 1)
        
        # 判断是否需要处理mask
        mask_region = None
        if mask is not None:
            # 检查mask是否已经是全局尺寸 (对应MaskMix已被移动的情况)
            is_global_mask = (mask.dim() == 4 and mask.shape[1] == h_b and mask.shape[2] == w_b) or \
                            (mask.dim() == 3 and mask.shape[1] == h_b and mask.shape[2] == w_b)
            
            if is_global_mask:
                # 如果mask已经是全局尺寸，直接提取重叠区域
                if mask.dim() == 4:  # B,H,W,C
                    mask_region = mask[:, y:y_end, x:x_end, :]
                else:  # B,H,W
                    mask_region = mask[:, y:y_end, x:x_end]
                    mask_region = mask_region.unsqueeze(-1).expand(-1, -1, -1, surface_region.shape[-1])
            else:
                # 如果mask与surface同尺寸，需要匹配到重叠区域
                if mask.dim() == 4:  # B,H,W,C
                    # 尝试直接使用前h_overlap x w_overlap的区域
                    if mask.shape[1] >= h_overlap and mask.shape[2] >= w_overlap:
                        mask_region = mask[:, :h_overlap, :w_overlap, :]
                    else:
                        # 否则，进行插值调整
                        mask_region = torch.nn.functional.interpolate(
                            mask.permute(0, 3, 1, 2),
                            size=(h_overlap, w_overlap),
                            mode="bilinear"
                        ).permute(0, 2, 3, 1)
                elif mask.dim() == 3:  # B,H,W
                    # 尝试直接使用前h_overlap x w_overlap的区域
                    if mask.shape[1] >= h_overlap and mask.shape[2] >= w_overlap:
                        mask_region = mask[:, :h_overlap, :w_overlap]
                    else:
                        # 否则，进行插值调整
                        mask_region = torch.nn.functional.interpolate(
                            mask.unsqueeze(1),
                            size=(h_overlap, w_overlap),
                            mode="bilinear"
                        ).squeeze(1)
                    mask_region = mask_region.unsqueeze(-1).expand(-1, -1, -1, surface_region.shape[-1])
            
            # 确保mask_region与surface_region的空间尺寸匹配
            if mask_region is not None and mask_region.shape[1:3] != surface_region.shape[1:3]:
                if mask_region.dim() == 4:
                    mask_region = torch.nn.functional.interpolate(
                        mask_region.permute(0, 3, 1, 2),
                        size=surface_region.shape[1:3],
                        mode="bilinear"
                    ).permute(0, 2, 3, 1)
                else:
                    mask_region = torch.nn.functional.interpolate(
                        mask_region.unsqueeze(1),
                        size=surface_region.shape[1:3],
                        mode="bilinear"
                    ).squeeze(1)
                    mask_region = mask_region.unsqueeze(-1).expand(-1, -1, -1, surface_region.shape[-1])
        
        # 如果没有遮罩或处理失败，创建全1遮罩
        if mask_region is None:
            mask_region = torch.ones_like(surface_region)
        
        # 应用混合模式
        if blend_mode == "normal":
            blended = back_region * (1 - mask_region) + surface_region * mask_region
        elif blend_mode == "multiply":
            blended = back_region * ((1 - mask_region) + surface_region * mask_region)
        elif blend_mode == "screen":
            blended = 1.0 - (1.0 - back_region) * (1.0 - surface_region * mask_region)
        elif blend_mode == "overlay":
            low = 2.0 * back_region * surface_region
            high = 1.0 - 2.0 * (1.0 - back_region) * (1.0 - surface_region)
            mask_low = (back_region <= 0.5).float()
            blended = mask_low * low + (1 - mask_low) * high
            blended = back_region * (1 - mask_region) + blended * mask_region
        elif blend_mode == "darken":
            blended = torch.min(back_region, surface_region)
            blended = back_region * (1 - mask_region) + blended * mask_region
        elif blend_mode == "lighten":
            blended = torch.max(back_region, surface_region)
            blended = back_region * (1 - mask_region) + blended * mask_region
        elif blend_mode == "color_dodge":
            # 避免除以0
            eps = 1e-7
            temp = back_region / (1.0 - surface_region + eps)
            temp = torch.clamp(temp, 0.0, 1.0)
            blended = back_region * (1 - mask_region) + temp * mask_region
        elif blend_mode == "color_burn":
            # 避免除以0
            eps = 1e-7
            temp = 1.0 - (1.0 - back_region) / (surface_region + eps)
            temp = torch.clamp(temp, 0.0, 1.0)
            blended = back_region * (1 - mask_region) + temp * mask_region
        elif blend_mode == "hard_light":
            # 应用overlay公式但互换表面和背景
            low = 2.0 * surface_region * back_region
            high = 1.0 - 2.0 * (1.0 - surface_region) * (1.0 - back_region)
            mask_low = (surface_region <= 0.5).float()
            temp = mask_low * low + (1 - mask_low) * high
            blended = back_region * (1 - mask_region) + temp * mask_region
        elif blend_mode == "soft_light":
            # 软光混合
            dark = 2.0 * back_region * surface_region
            light = 1.0 - 2.0 * (1.0 - back_region) * (1.0 - surface_region)
            temp = back_region * (1.0 - back_region) * (light - dark) + back_region
            blended = back_region * (1 - mask_region) + temp * mask_region
        elif blend_mode == "difference":
            blended = torch.abs(back_region - surface_region)
            blended = back_region * (1 - mask_region) + blended * mask_region
        elif blend_mode == "exclusion":
            blended = back_region + surface_region - 2.0 * back_region * surface_region
            blended = back_region * (1 - mask_region) + blended * mask_region
        else:
            # 默认为normal模式
            blended = back_region * (1 - mask_region) + surface_region * mask_region
        
        # 确保像素值在有效范围内
        blended = torch.clamp(blended, 0.0, 1.0)
        
        # 将混合区域写回背景
        result = back.clone()
        result[:, y:y_end, x:x_end] = blended
        
        return result

    def move_mask(self, mask, x, y, width, height):
        """
        移动遮罩并处理边界填充
        参数:
            mask: 要移动的遮罩
            x, y: 移动的坐标
            width, height: 目标画布的宽高
        返回:
            移动后的遮罩
        """
        b, h_m, w_m = mask.shape
        
        # 创建新的空白遮罩
        moved_mask = torch.zeros((b, height, width), dtype=mask.dtype, device=mask.device)
        
        # 计算有效的边界
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(width, x + w_m)
        y_end = min(height, y + h_m)
        
        # 计算从原始遮罩中获取的区域
        x_src_start = max(0, -x)
        y_src_start = max(0, -y)
        x_src_end = min(w_m, width - x)
        y_src_end = min(h_m, height - y)
        
        # 检查是否有有效重叠区域
        if x_start < x_end and y_start < y_end and x_src_start < x_src_end and y_src_start < y_src_end:
            # 复制有效区域
            try:
                moved_mask[:, y_start:y_end, x_start:x_end] = mask[:, y_src_start:y_src_end, x_src_start:x_src_end]
            except Exception as e:
                print(f"复制区域失败: {e}")
            
            # 对边界进行智能填充处理
            # 检查边缘的平均值来决定填充类型
            
            # 左边界
            if x_start > 0 and x_src_start > 0:
                edge_values = mask[:, y_src_start:y_src_end, x_src_start].unsqueeze(-1)
                edge_avg = edge_values.mean()
                # 如果边缘平均值小于0.5，渐变到0，否则渐变到1
                fade_to_zero = edge_avg < 0.5
                moved_mask[:, y_start:y_end, :x_start] = self.fade_edge(edge_values, x_start, fade_to_zero)
            
            # 右边界
            right_width = width - x_end
            if right_width > 0 and x_src_end < w_m:
                edge_values = mask[:, y_src_start:y_src_end, x_src_end-1].unsqueeze(-1)
                edge_avg = edge_values.mean()
                fade_to_zero = edge_avg < 0.5
                moved_mask[:, y_start:y_end, x_end:] = self.fade_edge(edge_values, right_width, fade_to_zero)
            
            # 上边界
            if y_start > 0 and y_src_start > 0:
                edge_values = mask[:, y_src_start, x_src_start:x_src_end].unsqueeze(1)
                edge_avg = edge_values.mean()
                fade_to_zero = edge_avg < 0.5
                moved_mask[:, :y_start, x_start:x_end] = self.fade_edge(edge_values, y_start, fade_to_zero, horizontal=False)
            
            # 下边界
            bottom_height = height - y_end
            if bottom_height > 0 and y_src_end < h_m:
                edge_values = mask[:, y_src_end-1, x_src_start:x_src_end].unsqueeze(1)
                edge_avg = edge_values.mean()
                fade_to_zero = edge_avg < 0.5
                moved_mask[:, y_end:, x_start:x_end] = self.fade_edge(edge_values, bottom_height, fade_to_zero, horizontal=False)
            
            # 处理角落区域，使用相邻边缘的平均值
            # 左上角
            if x_start > 0 and y_start > 0:
                try:
                    corner_fill = moved_mask[:, y_start:y_start+1, :x_start] * 0.5 + moved_mask[:, :y_start, x_start:x_start+1] * 0.5
                    moved_mask[:, :y_start, :x_start] = corner_fill
                except:
                    # 如果出现任何问题，使用边缘平均值填充
                    edge_avg = moved_mask[:, y_start:y_end, :x_start].mean() * 0.5 + moved_mask[:, :y_start, x_start:x_end].mean() * 0.5
                    moved_mask[:, :y_start, :x_start] = edge_avg
            
            # 右上角
            if right_width > 0 and y_start > 0:
                try:
                    corner_fill = moved_mask[:, y_start:y_start+1, x_end:] * 0.5 + moved_mask[:, :y_start, x_end-1:x_end] * 0.5
                    moved_mask[:, :y_start, x_end:] = corner_fill
                except:
                    # 如果出现任何问题，使用边缘平均值填充
                    edge_avg = moved_mask[:, y_start:y_end, x_end:].mean() * 0.5 + moved_mask[:, :y_start, x_end-1:x_end].mean() * 0.5
                    moved_mask[:, :y_start, x_end:] = edge_avg
            
            # 左下角
            if x_start > 0 and bottom_height > 0:
                try:
                    corner_fill = moved_mask[:, y_end-1:y_end, :x_start] * 0.5 + moved_mask[:, y_end:, x_start:x_start+1] * 0.5
                    moved_mask[:, y_end:, :x_start] = corner_fill
                except:
                    # 如果出现任何问题，使用边缘平均值填充
                    edge_avg = moved_mask[:, y_end-1:y_end, :x_start].mean() * 0.5 + moved_mask[:, y_end:, x_start:x_end].mean() * 0.5
                    moved_mask[:, y_end:, :x_start] = edge_avg
            
            # 右下角
            if right_width > 0 and bottom_height > 0:
                try:
                    corner_fill = moved_mask[:, y_end-1:y_end, x_end:] * 0.5 + moved_mask[:, y_end:, x_end-1:x_end] * 0.5
                    moved_mask[:, y_end:, x_end:] = corner_fill
                except:
                    # 如果出现任何问题，使用边缘平均值填充
                    edge_avg = moved_mask[:, y_end-1:y_end, x_end:].mean() * 0.5 + moved_mask[:, y_end:, x_end-1:x_end].mean() * 0.5
                    moved_mask[:, y_end:, x_end:] = edge_avg
        
        return moved_mask

    def fade_edge(self, edge_values, size, fade_to_zero=True, horizontal=True):
        """
        创建从边缘值渐变的填充
        参数:
            edge_values: 边缘的值
            size: 需要填充的尺寸
            fade_to_zero: 是否渐变至0，否则渐变至1
            horizontal: 是否为水平方向的填充
        返回:
            填充区域
        """
        # 确保size至少为1
        size = max(1, size)
        
        b = edge_values.shape[0]
        target = 0.0 if fade_to_zero else 1.0
        
        # 创建渐变因子，使用指数衰减获得更平滑的过渡
        if horizontal:
            # 创建渐变因子 [1, exp(-1), exp(-2), ..., exp(-size+1)]
            if fade_to_zero:
                gradient = torch.exp(torch.linspace(0, -3, size, device=edge_values.device))
            else:
                gradient = 1.0 - torch.exp(torch.linspace(0, -3, size, device=edge_values.device))
            
            # 扩展维度以便广播
            gradient = gradient.view(1, 1, size)
            edge_values_expanded = edge_values.expand(b, edge_values.shape[1], size)
            # 创建渐变填充
            faded = edge_values_expanded * gradient + target * (1 - gradient)
        else:
            # 垂直方向的渐变，使用相同的指数衰减
            if fade_to_zero:
                gradient = torch.exp(torch.linspace(0, -3, size, device=edge_values.device))
            else:
                gradient = 1.0 - torch.exp(torch.linspace(0, -3, size, device=edge_values.device))
            
            gradient = gradient.view(1, size, 1)
            edge_values_expanded = edge_values.expand(b, size, edge_values.shape[2])
            faded = edge_values_expanded * gradient + target * (1 - gradient)
        
        return faded


class Composite_Coordinate:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-坐标偏移设置
        输入1：CompositeOptions 接收其它的CompositeOptions输入
        参数1：x 整型，surface在back上的像素X方向上位置可为负数
        参数2：y 整型，surface在back上的像素Y方向上位置可为负数
        参数3：scale 浮点，图像缩放比例，缩放方式默认采用等比缩放，默认倍数1.0
              如果连入Composite_AutoScale节点则缩放倍数和其叠加，缩放方式采用Composite_AutoScale的
        参数4：Center_back 布尔值，是否以back的中心为xy坐标0点,
        参数5：Center_surface 布尔值，是否以surface_*输入图像的中心计算xy
        参数6：flip_x 布尔值，是否在X方向镜像
        参数7：flip_y 布尔值，是否在Y方向镜像
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "y": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "Center": ("BOOLEAN", {"default": False}),
                "flip_x": ("BOOLEAN", {"default": False}),
                "flip_y": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, x, y, scale, Center, flip_x, flip_y, Options=None):
        #收集当前设置
        set_s = {
            "x": x,
            "y": y,
            "scale": scale,  # 确保scale被正确传递
            "Center": Center,
            "flip_x": flip_x,
            "flip_y": flip_y,
        }
        return update_options(set_s, Options)


class Composite_Scale:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-缩放surface
        输入：接收其它的CompositeOptions输入
        参数1：scale_X 浮点，宽度缩放
        参数2：scale_Y 浮点，高度缩放
        参数3：ScaleMethod 列表(下拉列表)，缩放方式
              DoNot_AutoScale 不应用自动缩放
              long 等比例将surface的长边缩放到back内，默认值
              long_fill 等比例将surface的长边缩放到back内并将短边补充到铺满画布
              short 等比例将surface的短边缩放到back内，此时缩小或移动后会显示多余的部分
              short_crop 等比例将surface的短边缩放到back内并裁剪掉多余的部分
              stretch 通过拉伸将surface缩放到与back一样大小
              average 将surface的长宽平均值缩放到与back长宽平均值一样大小
        参数4：ConditionalScale 列表(下拉列表)，条件缩放
              NotUsed 不使用条件缩放，默认值
              max_if 当surface需要放大时才按设置缩放
              min_if 当surface需要缩小时才按设置缩放
              
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scale_X": ("FLOAT", {"default": 1, "min": 0.01, "max": 2048, "step": 0.001}),
                "scale_Y": ("FLOAT", {"default": 1, "min": 0.01, "max": 2048, "step": 0.001}),
                "ScaleMethod": (["DoNot_AutoScale", "long", "long_fill", "short", "short_crop", "stretch", "average"], {"default": "long"}),
                "ConditionalScale": (["NotUsed", "max_if", "min_if"], {"default": "NotUsed"}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, scale_X, scale_Y, ScaleMethod, ConditionalScale, Options=None):
        set_s = {
            "ConditionalScale": ConditionalScale, 
            "ScaleMethod": ScaleMethod, 
            "scale_X": scale_X,
            "scale_Y": scale_Y
        }
        return update_options(set_s,Options)


class Composite_Basic:
    DESCRIPTION = """用于创建基础混合选项的节点
    混合图像-缩放surface
        输入：接收其它的CompositeOptions输入
        参数1：blend_mode 列表(下拉列表)，可选与底图的叠加算法
        参数2：mask_mode 列表(下拉列表)，应用遮罩时的方法
        参数3：opacity 浮点，透明度，为负数时为反色+透明度
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blend_mode": (["normal", "multiply", "screen", "overlay", "darken", "lighten", "color_dodge", "color_burn", "hard_light", "soft_light", "difference", "exclusion"], {"default": "normal"}),
                "mask_mode": (["normal", "multiply", "screen"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, blend_mode, mask_mode, opacity, Options=None):
        set_s = {
            "blend_mode": blend_mode, 
            "mask_mode": mask_mode, 
            "opacity": opacity,
        }
        return update_options(set_s,Options)


class Composite_Other:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-其他设置(本节点建议设置影响范围)
        输入：接收其它的CompositeOptions输入
        参数1：flip_A 布尔值，是否翻转alpha通道
        参数2：Ignore_Input 布尔值，是否忽略输入
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flip_A": ("BOOLEAN", {"default": False}),
                "Ignore_Input": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, flip_A, Ignore_Input, Options=None):
        set_s = {
            "flip_A": flip_A,
            "Ignore_Input": Ignore_Input,
        }
        return update_options(set_s,Options)


class Composite_Global_adv:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-全局设置
        输入：接收其它的CompositeOptions输入
        参数1：Switch_image 布尔值，是否交换back和surface图像
        参数2：Switch_options 布尔值，是否交换back和surface的编辑参数
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Switch_image": ("BOOLEAN", {"default": False}),
                "Switch_options": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, Switch_image, Switch_options, Options=None):
        set_s = {
            "Switch_image": Switch_image,
            "Switch_options": Switch_options,
        }
        return update_options(set_s,Options,"Global")


class Composite_Canvas_adv:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-画布设置
        输入：接收其它的CompositeOptions输入
        参数1：color 字符串，画布颜色（十六进制颜色字符串）
        参数2：image 图像，当有输入时画布使用的图像而不是纯色
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("STRING", {"default": "#ffffff"}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, color, image=None, Options=None):
        Solid_Color = True
        if image is not None: Solid_Color = False
        set_s = {
            "Solid_Color": Solid_Color,
            "image": image,
            "color": color,
        }
        return update_options(set_s,Options,"Canvas")


class Composite_Application_pro:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，控制所有设置的应用范围
    混合图像-应用范围设置
        输入：接收其它的CompositeOptions输入
        参数1：apply_back 布尔值，是否将输入设置应用到back
        参数2：apply_surface 布尔值，是否将输入设置应用到surface
        参数3：apply_mask 布尔值，是否将输入设置应用到MaskMix
        参数4：res_back 布尔值，是将back的设置恢复默认
        参数5：res_surface 布尔值，是否将surface的设置恢复默认
        参数6：res_mask 布尔值，是否将MaskMix的设置恢复默认
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "res_back": ("BOOLEAN", {"default": False}),
                "res_surface": ("BOOLEAN", {"default": False}),
                "res_mask": ("BOOLEAN", {"default": False}),
                "apply_back": ("BOOLEAN", {"default": False}),
                "apply_surface": ("BOOLEAN", {"default": True}),
                "apply_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, apply_back, apply_surface, apply_mask,
                    res_back, res_surface, res_mask, Options=None):
        
        if Options is None:
            Options = default_options.copy()

        key = ["back","surface","MaskMix"]
        value_res = [res_back, res_surface, res_mask]
        value_apply = [apply_back,apply_surface,apply_mask]
        for i in range(3):
            if value_res[i]: 
                Options[key[i]] = default_options[key[i]]
            Options[key[i]]["enable"] = value_apply[i]

        return (Options,)


class Composite_Merge_pro:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，用于合并多个设置
    混合图像-设置合并，若无输入则输出默认设置
    注意：非覆盖模式可能会产生错误的设置
        输入1：Options1 第一个设置输入
        输入2：Options2 第二个设置输入
        参数1：merge_mode 列表(下拉列表)，值的合并方式
              override 覆盖模式，当Options2中有非默认值时覆盖Options1后输出
              add 叠加模式
              subtract 相减模式
              multiplication 相减模式
              max 取最大值模式，取两个设置中的最大值
              min 取最小值模式，取两个设置中的最小值
        参数2：Cannot_merged 列表(下拉列表)，当有无法合并的(下拉列表)用哪个设置
        参数3：merge_Application 布尔值，是否合并Application
        参数4：merge_back 布尔值，是否合并back设置
        参数5：merge_surface 布尔值，是否合并surface设置
        参数6：merge_MaskMix 布尔值，是否合并MaskMix设置
        参数7：merge_Canvas 布尔值，是否合并Canvas设置
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "merge_mode": (["override", "add", "subtract", "multiplication", "division", "max", "min"], {"default": "override"}),
                "Cannot_merged":(["default", "Options1", "Options2"], {"default": "Options1"}),
                "merge_Application": ("BOOLEAN", {"default": True}),
                "merge_back": ("BOOLEAN", {"default": False}),
                "merge_surface": ("BOOLEAN", {"default": False}),
                "merge_MaskMix": ("BOOLEAN", {"default": False}),
                "merge_Canvas": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "Options1": ("Composite_Basic",),
                "Options2": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, merge_mode, Cannot_merged, merge_Application, merge_back, merge_surface, merge_MaskMix, merge_Canvas, Options1=None, Options2=None):
        # 获取默认设置
        Options0 = default_options.copy()
        # 如果输入都为空，返回默认
        if Options1 is None and Options2 is None:
            return (Options0,)
        # 只输入一个，直接返回
        if Options1 is None:
            return (Options2 if Options2 is not None else Options0,)
        if Options2 is None:
            return (Options1,)

        # 合并结果初始化
        result = Options1.copy()
        # 分类合并开关
        merge_keys = [
            ("Global", merge_Application),
            ("back", merge_back),
            ("surface", merge_surface),
            ("MaskMix", merge_MaskMix),
            ("Canvas", merge_Canvas),
        ]
        for key, do_merge in merge_keys:
            if not do_merge:
                continue  # 不合并，保留Options1的原值
            # 合并该分类下所有子项
            for sub_key in result[key]:
                v1 = Options1[key][sub_key]
                v2 = Options2[key][sub_key]
                v0 = Options0[key][sub_key]
                # 合并逻辑
                if merge_mode == "override":
                    if v2 != v0:
                        result[key][sub_key] = v2
                else:
                    # 只对数值/布尔类型做运算
                    if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
                        result[key][sub_key] = self.calculation(v1, v2, merge_mode)
                    elif isinstance(v1, bool) and isinstance(v2, bool):
                        result[key][sub_key] = bool(self.calculation(v1, v2, merge_mode))
                    else:
                        # 其它类型按Cannot_merged策略
                        if Cannot_merged == "default":
                            result[key][sub_key] = v0
                        elif Cannot_merged == "Options1":
                            result[key][sub_key] = v1
                        elif Cannot_merged == "Options2":
                            result[key][sub_key] = v2
        return (result,)
    
    def calculation(self,a, b, mode: str):
        # 定义一个操作模式到函数的映射
        operations = {
            "override": lambda x, y: y,
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiplication": lambda x, y: x * y,
            "division": lambda x, y: x / y,
            "max": max,
            "min": min
        }

        # 获取对应的函数并执行
        operation = operations.get(mode)
        if operation:
            return operation(a, b)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class Composite_Mask:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-遮罩设置
        输入：接收其它的CompositeOptions输入
        参数1：use_mask 列表(下拉列表)，指定使用哪个遮罩
              auto 自动检测，默认值
              none 不使用遮罩
              mask 使用MaskMix输入
              surface_alpha 使用surface的A通道
              back_alpha 使用back的A通道
        参数2：output_mask 列表(下拉列表)，指定输出哪个遮罩
              auto 自动检测，默认值
              none 不输出遮罩
              white 输出纯白遮罩
              black 输出纯黑遮罩
              mask 输出MaskMix
              surface_alpha 输出surface的A通道
              back_alpha 输出back的A通道
              mixed 通过MaskMix混合surface和back的A通道
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_mask": (["auto", "none", "mask", "surface_alpha", "back_alpha"], {"default": "auto"}),
                "output_mask": (["auto", "none", "white", "black", "mask", "surface_alpha", "back_alpha", "mixed"], {"default": "auto"}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, use_mask, output_mask, Options=None):
        set_s = {
            "use_mask": use_mask,
            "output_mask": output_mask,
        }
        return update_options(set_s, Options, "MaskMix")


class Composite_Batch:
    DESCRIPTION = """
    ImageCompositeMask_adv图像混合专用设置，注意相同的后续设置会覆盖前面的设置
    混合图像-批次处理方式设置
        输入：接收其它的CompositeOptions输入
        参数1：back_mode 列表(下拉列表)，背景图像批次处理方式
              auto 自动处理，根据当前批次自动匹配（默认值）
              flatten 将多批次图像按A通道叠加为单张图像
              first 使用批次的第一张图像
              last 使用批次的最后一张图像
              match_surface 匹配到surface的数量
              match_mask 匹配到MaskMix的数量
        参数2：surface_mode 列表(下拉列表)，表面图像批次处理方式
              auto 自动处理，根据当前批次自动匹配（默认值）
              flatten 将多批次图像按A通道叠加为单张图像
              first 使用批次的第一张图像
              last 使用批次的最后一张图像
              match_back 匹配到back的数量
              match_mask 匹配到MaskMix的数量
        参数3：mask_mode 列表(下拉列表)，遮罩图像批次处理方式
              auto 自动处理，根据当前批次自动匹配（默认值）
              flatten 将多批次遮罩叠加为单张遮罩
              first 使用批次的第一张遮罩
              last 使用批次的最后一张遮罩
              match_back 匹配到back的数量
              match_surface 匹配到surface的数量
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "back_mode": (["auto", "flatten", "first", "last", "match_surface", "match_mask"], {"default": "auto"}),
                "surface_mode": (["auto", "flatten", "first", "last", "match_back", "match_mask"], {"default": "auto"}),
                "mask_mode": (["auto", "flatten", "first", "last", "match_back", "match_surface"], {"default": "auto"}),
            },
            "optional": {
                "Options": ("Composite_Basic",),
            }
        }
    RETURN_TYPES = ("Composite_Basic",)
    FUNCTION = "get_options"
    CATEGORY = CATEGORY_NAME

    def get_options(self, back_mode, surface_mode, mask_mode, Options=None):
        set_s = {
            "batch_mode": back_mode,
        }
        back_set = update_options(set_s, Options, "back")
        
        set_s = {
            "batch_mode": surface_mode,
        }
        surface_set = update_options(set_s, back_set[0], "surface")
        
        set_s = {
            "batch_mode": mask_mode,
        }
        return update_options(set_s, surface_set[0], "MaskMix")


NODE_CLASS_MAPPINGS = {
    "ImageCompositeMask_Adv": ImageCompositeMask_Adv,
    "Composite_Coordinate": Composite_Coordinate,
    "Composite_Scale": Composite_Scale,
    "Composite_Basic": Composite_Basic,
    "Composite_Other": Composite_Other,
    "Composite_Global_adv": Composite_Global_adv,
    "Composite_Canvas_adv": Composite_Canvas_adv,
    "Composite_Application_pro": Composite_Application_pro,
    "Composite_Merge_pro": Composite_Merge_pro,
    "Composite_Mask": Composite_Mask,
    "Composite_Batch": Composite_Batch,
}