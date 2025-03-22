import torch
import torch.nn.functional as F
import numpy as np

CATEGORY_NAME = "WJNode/mask_crop"

class Accurate_mask_clipping:
    DESCRIPTION = """
    Accurately find mask boundaries and optionally crop to those boundaries.
    Features:
    1. Find the bounding box of non-zero areas in masks
    2. Apply offset to expand/shrink the bounding box
    3. Optional cropping to the bounding box
    4. Adjustable threshold for determining foreground pixels
    
    精确查找遮罩边界并可选裁剪
    功能：
    1. 查找遮罩中非零区域的边界框
    2. 应用偏移量扩展/缩小边界框
    3. 可选择是否裁剪到边界框
    4. 可调整的灰度阈值用于确定前景像素
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "crop": ("BOOLEAN", {"default": False}),
                "offset": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "min_y", "max_y", "min_x", "max_x")
    FUNCTION = "accurate_mask_clipping"

    def accurate_mask_clipping(self, mask, crop, offset, threshold):
        """高效查找遮罩边界并可选裁剪"""
        # 获取非零值的索引
        if len(mask.shape) == 3:
            mask_batch = []
            bbox_data = []
            
            for m in mask:
                # 应用阈值，将低于阈值的像素视为背景
                thresholded_mask = (m > threshold).float()
                
                # 获取非零索引
                y_indices, x_indices = torch.nonzero(thresholded_mask, as_tuple=True)
                
                if len(y_indices) == 0:
                    # 如果没有前景像素，保持原样
                    mask_batch.append(m.unsqueeze(0))
                    bbox_data.append((0, m.shape[0]-1, 0, m.shape[1]-1))
                    continue
                    
                # 计算边界框
                min_y, max_y = y_indices.min().item(), y_indices.max().item()
                min_x, max_x = x_indices.min().item(), x_indices.max().item()
                
                # 应用偏移
                min_y = max(min_y - offset, 0)
                min_x = max(min_x - offset, 0)
                max_y = min(max_y + offset, m.shape[0]-1)
                max_x = min(max_x + offset, m.shape[1]-1)
                
                bbox_data.append((min_y, max_y, min_x, max_x))
                
                # 根据crop参数决定是否裁剪
                if crop:
                    cropped = m[min_y:max_y+1, min_x:max_x+1]
                    mask_batch.append(cropped.unsqueeze(0))
                else:
                    mask_batch.append(m.unsqueeze(0))
            
            # 计算所有边界框的平均值作为返回值
            avg_min_y = sum(bbox[0] for bbox in bbox_data) // len(bbox_data)
            avg_max_y = sum(bbox[1] for bbox in bbox_data) // len(bbox_data)
            avg_min_x = sum(bbox[2] for bbox in bbox_data) // len(bbox_data)
            avg_max_x = sum(bbox[3] for bbox in bbox_data) // len(bbox_data)
            
            return (torch.cat(mask_batch, dim=0), avg_min_y, avg_max_y, avg_min_x, avg_max_x)
        else:
            raise ValueError("输入遮罩维度必须为3 (batch, height, width)")

class mask_crop_square:
    DESCRIPTION = """
        说明：
            功能1：输入图像和遮罩，按遮罩区域的中心裁剪图像和遮罩，输出为方形且固定大小的图像/遮罩/裁剪数据
            注意1：若图像为多张，遮罩为1张则按这个遮罩裁剪所有图像
            注意2：若图像为1张，遮罩为多张则按这些遮罩裁剪这个图像
            注意3：若图像与遮罩批次均大于1且数量不一样则丢弃多的批次
            注意4：若根据遮罩区域的中心计算的裁剪外框超出图像边界，则补充超出边界的部分为Exceeding指定的颜色(含图像和遮罩)

            功能2：输入裁剪数据，将输入的图像按crop_data数据贴回原图像或图像批次
        输入：
            images：输入图像或图像批次，若masks无输入则使用A通道裁剪
            inversion_mask：是否翻转输入遮罩
            mask_threshold：裁剪遮罩阈值，忽略浅色遮罩区域和零碎区域
            size：指定输出大小(像素值)，若为0则使用输入遮罩区域的最大值
            Exceeding: 超出裁剪范围的像素值填充方式,
            masks：输入用于获取裁剪数据的遮罩，若无输入则则使用images的A通道
                若images输入无A通道则直接输出原图
            crop_data：若此输入为空，则使用裁剪功能(功能1)
                若此输入不为空，则将images贴回crop_data数据内含的原图上，此时有关于裁剪的参数将无效
        输出：
            images：原输入图像或图像批次
            masks：原输入遮罩或遮罩批次
            crop_data：用于贴回原图像或图像批次，(包含图像大小/原图/位置信息)
                若crop_data输入不为空则直接返回原crop_data输入
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images":("IMAGE",),
                "inversion_mask":("BOOLEAN",{"default":False}),
                "mask_threshold": ("FLOAT",{"default":0.01,"min":0.0,"max":1.0,"step":0.001}),
                "size": ("INT",{"default":0,"min":0,"max":8192,"step":1}),
                "Exceeding": (["White","Black","Gray"],{"default":"White"}),
            },
            "optional": {
                "masks":("MASK",),
                "crop_data":("crop_data",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","crop_data")
    RETURN_NAMES = ("crop_images","crop_masks","crop_data")
    FUNCTION = "restore_mask"
    def restore_mask(self, images, inversion_mask, mask_threshold, size, Exceeding="White", masks=None, crop_data=None):
        """按遮罩区域裁剪图像或将裁剪图像贴回原图"""

        # 处理填充颜色
        if Exceeding == "White":
            fill_value = 1.0
        elif Exceeding == "Black":
            fill_value = 0.0
        else:  # Gray
            fill_value = 0.5
            
        # 如果没有提供masks，尝试使用图像的alpha通道
        if masks is None:
            if images.shape[3] == 4:  # 有alpha通道
                masks = images[:, :, :, 3]
            else:
                # 没有alpha通道，直接返回原图
                dummy_mask = torch.ones((images.shape[0], images.shape[1], images.shape[2]), device=images.device)
                return images, dummy_mask, {"images": images, "masks": dummy_mask, "positions": []}

        # 如果提供了crop_data，执行回贴功能
        if crop_data is not None:
            # 获取原始图像和位置信息
            original_images = crop_data["images"]
            original_masks = crop_data["masks"]
            positions = crop_data["positions"]
            
            # 确保批次大小一致
            batch_size = min(images.shape[0], len(positions))
            images = images[:batch_size]
            
            # 创建结果图像和遮罩的副本
            result_images = original_images.clone()
            result_masks = original_masks.clone()
            
            # 对每个位置执行回贴操作
            for i in range(batch_size):
                img = images[i]
                pos = positions[i]
                mask = masks[i]
                
                # 如果提供了遮罩，检查是否为全黑或低于阈值
                mask_mean = mask.mean().item()
                if mask_mean < mask_threshold:
                    continue
                
                # 解析位置信息
                crop_min_y, crop_min_x, crop_max_y, crop_max_x = pos
                
                # 计算裁剪区域的大小
                crop_height = crop_max_y - crop_min_y
                crop_width = crop_max_x - crop_min_x
                
                # 如果输入图像大小与裁剪区域不匹配，需要调整
                if img.shape[0] != crop_height or img.shape[1] != crop_width:
                    # 计算缩放比例
                    scale_y = crop_height / img.shape[0]
                    scale_x = crop_width / img.shape[1]
                    
                    # 如果需要缩放，使用插值方法调整大小
                    if scale_y != 1.0 or scale_x != 1.0:
                        img = F.interpolate(
                            img.unsqueeze(0).permute(0, 3, 1, 2),
                            size=(crop_height, crop_width),
                            mode='bilinear',
                            align_corners=False
                        ).permute(0, 2, 3, 1).squeeze(0)
                        
                        if masks is not None and i < masks.shape[0]:
                            mask = masks[i]
                            mask = F.interpolate(
                                mask.unsqueeze(0).unsqueeze(0),
                                size=(crop_height, crop_width),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)
                
                # 计算有效区域（原图范围内）
                valid_dst_min_y = max(0, crop_min_y)
                valid_dst_max_y = min(original_images.shape[1], crop_max_y)
                valid_dst_min_x = max(0, crop_min_x)
                valid_dst_max_x = min(original_images.shape[2], crop_max_x)
                
                # 计算源区域（裁剪图像中对应的区域）
                valid_src_min_y = valid_dst_min_y - crop_min_y
                valid_src_max_y = valid_src_min_y + (valid_dst_max_y - valid_dst_min_y)
                valid_src_min_x = valid_dst_min_x - crop_min_x
                valid_src_max_x = valid_src_min_x + (valid_dst_max_x - valid_dst_min_x)
                
                # 确保区域有效
                if (valid_src_max_y > valid_src_min_y and valid_src_max_x > valid_src_min_x and
                    valid_src_min_y >= 0 and valid_src_min_x >= 0 and
                    valid_src_max_y <= img.shape[0] and valid_src_max_x <= img.shape[1]):
                    
                    # 检查是否需要边缘过渡
                    if "edge_blend" in crop_data and crop_data["edge_blend"] > 0:
                        edge_blend = crop_data["edge_blend"]
                        
                        # 计算过渡区域的宽度
                        blend_y = int((valid_dst_max_y - valid_dst_min_y) * edge_blend / 2)
                        blend_x = int((valid_dst_max_x - valid_dst_min_x) * edge_blend / 2)
                        
                        # 确保过渡区域至少为1像素
                        blend_y = max(1, blend_y)
                        blend_x = max(1, blend_x)
                        
                        # 创建过渡权重掩码
                        weight_mask = torch.ones((valid_dst_max_y - valid_dst_min_y, valid_dst_max_x - valid_dst_min_x), 
                                               device=img.device)
                        
                        # 判断哪些边需要过渡（只有不在原图边缘的边才需要过渡）
                        top_edge_needs_blend = valid_dst_min_y > 0 and crop_min_y < 0
                        bottom_edge_needs_blend = valid_dst_max_y < original_images.shape[1] and crop_max_y > original_images.shape[1]
                        left_edge_needs_blend = valid_dst_min_x > 0 and crop_min_x < 0
                        right_edge_needs_blend = valid_dst_max_x < original_images.shape[2] and crop_max_x > original_images.shape[2]
                        
                        # 上边缘过渡（如果不是原图上边缘）
                        if not top_edge_needs_blend and valid_dst_min_y > 0:
                            for y in range(blend_y):
                                if y < weight_mask.shape[0]:
                                    weight = y / blend_y
                                    weight_mask[y, :] = weight
                        
                        # 下边缘过渡（如果不是原图下边缘）
                        if not bottom_edge_needs_blend and valid_dst_max_y < original_images.shape[1]:
                            for y in range(blend_y):
                                if y < weight_mask.shape[0]:
                                    weight = y / blend_y
                                    weight_mask[-(y+1), :] = weight
                        
                        # 左边缘过渡（如果不是原图左边缘）
                        if not left_edge_needs_blend and valid_dst_min_x > 0:
                            for x in range(blend_x):
                                if x < weight_mask.shape[1]:
                                    weight = x / blend_x
                                    weight_mask[:, x] = torch.min(weight_mask[:, x], torch.tensor(weight, device=img.device))
                        
                        # 右边缘过渡（如果不是原图右边缘）
                        if not right_edge_needs_blend and valid_dst_max_x < original_images.shape[2]:
                            for x in range(blend_x):
                                if x < weight_mask.shape[1]:
                                    weight = x / blend_x
                                    weight_mask[:, -(x+1)] = torch.min(weight_mask[:, -(x+1)], torch.tensor(weight, device=img.device))
                        
                        # 扩展权重掩码以匹配图像通道
                        weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, img.shape[2])
                        
                        # 应用过渡
                        src_img = img[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
                        dst_img = result_images[i, valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x]
                        
                        # 混合图像
                        blended_img = src_img * weight_mask + dst_img * (1 - weight_mask)
                        result_images[i, valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = blended_img
                        
                        # 如果提供了遮罩，也应用过渡
                        if masks is not None and i < masks.shape[0]:
                            mask = masks[i]
                            if mask.shape[0] >= valid_src_max_y and mask.shape[1] >= valid_src_max_x:
                                src_mask = mask[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
                                dst_mask = result_masks[i, valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x]
                                
                                # 混合遮罩
                                weight_mask_2d = weight_mask[:, :, 0]  # 取第一个通道作为2D权重掩码
                                blended_mask = src_mask * weight_mask_2d + dst_mask * (1 - weight_mask_2d)
                                result_masks[i, valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = blended_mask
                    else:
                        # 无过渡，直接回贴图像
                        result_images[i, 
                                    valid_dst_min_y:valid_dst_max_y, 
                                    valid_dst_min_x:valid_dst_max_x] = \
                            img[valid_src_min_y:valid_src_max_y, 
                                    valid_src_min_x:valid_src_max_x]
                        
                        # 如果提供了遮罩，也回贴遮罩
                        if masks is not None and i < masks.shape[0]:
                            mask = masks[i]
                            if mask.shape[0] >= valid_src_max_y and mask.shape[1] >= valid_src_max_x:
                                result_masks[i, valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = \
                                    mask[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
            
            return result_images, result_masks, crop_data
        
        
        # 处理mask反转
        if inversion_mask:
            masks = 1.0 - masks
            
        # 统一批次大小
        batch_size = min(images.shape[0], masks.shape[0])
        images = images[:batch_size]
        masks = masks[:batch_size]
        
        # 如果masks只有一个但images有多个，复制masks
        if masks.shape[0] == 1 and images.shape[0] > 1:
            masks = masks.repeat(images.shape[0], 1, 1)
        
        # 如果images只有一个但masks有多个，复制images
        if images.shape[0] == 1 and masks.shape[0] > 1:
            images = images.repeat(masks.shape[0], 1, 1, 1)
            
        # 存储裁剪结果
        cropped_images = []
        cropped_masks = []
        positions = []
        
        # 确定统一的裁剪大小
        final_crop_size = size if size > 0 else 0
        
        # 如果size为0，需要先计算所有遮罩的最大边界尺寸
        if final_crop_size == 0:
            for i in range(batch_size):
                mask = masks[i]
                thresholded_mask = (mask > mask_threshold).float()
                y_indices, x_indices = torch.nonzero(thresholded_mask, as_tuple=True)
                
                if len(y_indices) > 0:
                    min_y, max_y = y_indices.min().item(), y_indices.max().item()
                    min_x, max_x = x_indices.min().item(), x_indices.max().item()
                    mask_size = max(max_y - min_y, max_x - min_x) + 1
                    final_crop_size = max(final_crop_size, mask_size)
        
        # 如果仍然没有确定大小，使用默认值
        if final_crop_size == 0:
            final_crop_size = 64  # 默认大小
        
        for i in range(batch_size):
            img = images[i]
            mask = masks[i]
            
            # 应用阈值
            thresholded_mask = (mask > mask_threshold).float()
            
            # 获取非零索引
            y_indices, x_indices = torch.nonzero(thresholded_mask, as_tuple=True)
            
            if len(y_indices) == 0:
                # 如果没有前景像素，创建空白图像和遮罩
                padded_img = torch.ones((final_crop_size, final_crop_size, img.shape[2]), device=img.device) * fill_value
                padded_mask = torch.zeros((final_crop_size, final_crop_size), device=mask.device)
                cropped_images.append(padded_img.unsqueeze(0))
                cropped_masks.append(padded_mask.unsqueeze(0))
                positions.append((0, 0, final_crop_size, final_crop_size))
                continue
                
            # 计算边界框
            min_y, max_y = y_indices.min().item(), y_indices.max().item()
            min_x, max_x = x_indices.min().item(), x_indices.max().item()
            
            # 计算中心点
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            
            # 使用统一的裁剪大小
            crop_size = final_crop_size
                
            # 计算裁剪区域
            half_size = crop_size // 2
            crop_min_y = center_y - half_size
            crop_max_y = center_y + half_size + (crop_size % 2)  # 处理奇数大小
            crop_min_x = center_x - half_size
            crop_max_x = center_x + half_size + (crop_size % 2)  # 处理奇数大小
            
            # 创建填充后的图像和遮罩
            padded_img = torch.ones((crop_size, crop_size, img.shape[2]), device=img.device) * fill_value
            padded_mask = torch.zeros((crop_size, crop_size), device=mask.device)
            
            # 计算有效区域
            valid_src_min_y = max(0, crop_min_y)
            valid_src_max_y = min(img.shape[0], crop_max_y)
            valid_src_min_x = max(0, crop_min_x)
            valid_src_max_x = min(img.shape[1], crop_max_x)
            
            # 计算目标区域
            valid_dst_min_y = valid_src_min_y - crop_min_y
            valid_dst_max_y = valid_dst_min_y + (valid_src_max_y - valid_src_min_y)
            valid_dst_min_x = valid_src_min_x - crop_min_x
            valid_dst_max_x = valid_dst_min_x + (valid_src_max_x - valid_src_min_x)
            
            # 复制有效区域
            if valid_dst_max_y > valid_dst_min_y and valid_dst_max_x > valid_dst_min_x:
                padded_img[valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = \
                    img[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
                padded_mask[valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = \
                    mask[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
            
            cropped_images.append(padded_img.unsqueeze(0))
            cropped_masks.append(padded_mask.unsqueeze(0))
            positions.append((crop_min_y, crop_min_x, crop_max_y, crop_max_x))
        
        # 合并结果
        cropped_images = torch.cat(cropped_images, dim=0)
        cropped_masks = torch.cat(cropped_masks, dim=0)
        
        # 创建crop_data
        crop_data = {
            "images": images,
            "masks": masks,
            "positions": positions
        }
        
        return cropped_images, cropped_masks, crop_data

class crop_data_edit:
    DESCRIPTION = """
    说明：
        编辑crop_data数据，可以替换原图/原遮罩，设置回贴时的边缘过渡宽度比例
    输入：
        crop_data: 输入的crop_data数据
        replace_images: 是否替换原图像
        replace_masks: 是否替换原遮罩
        images: 用于替换的图像
        masks: 用于替换的遮罩
        edge_blend: 回贴时的边缘过渡宽度比例(0-1)，0表示无过渡，1表示全部过渡
    输出：
        crop_data: 编辑后的crop_data数据
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_data": ("crop_data",),
                "edge_blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("crop_data","IMAGE","MASK",)
    RETURN_NAMES = ("crop_data","images","masks",)
    FUNCTION = "edit_crop_data"
    
    def edit_crop_data(self, crop_data, edge_blend, images=None, masks=None):
        """编辑crop_data数据"""
        # 创建crop_data的副本
        new_crop_data = {
            "images": crop_data["images"],
            "masks": crop_data["masks"],
            "positions": crop_data["positions"]
        }
        
        # 如果需要替换原图像
        if images is not None:
            # 检查批次大小是否匹配
            if images.shape[0] == crop_data["images"].shape[0]:
                new_crop_data["images"] = images
            else:
                # 批次大小不匹配，调整大小
                batch_size = crop_data["images"].shape[0]
                if images.shape[0] > batch_size:
                    # 如果输入图像批次更大，截取
                    new_crop_data["images"] = images[:batch_size]
                else:
                    # 如果输入图像批次更小，重复最后一张
                    repeat_count = batch_size - images.shape[0]
                    repeated_images = images[-1].unsqueeze(0).repeat(repeat_count, 1, 1, 1)
                    new_crop_data["images"] = torch.cat([images, repeated_images], dim=0)
        
        # 如果需要替换原遮罩
        if masks is not None:
            # 检查批次大小是否匹配
            if masks.shape[0] == crop_data["masks"].shape[0]:
                new_crop_data["masks"] = masks
            else:
                # 批次大小不匹配，调整大小
                batch_size = crop_data["masks"].shape[0]
                if masks.shape[0] > batch_size:
                    # 如果输入遮罩批次更大，截取
                    new_crop_data["masks"] = masks[:batch_size]
                else:
                    # 如果输入遮罩批次更小，重复最后一张
                    repeat_count = batch_size - masks.shape[0]
                    repeated_masks = masks[-1].unsqueeze(0).repeat(repeat_count, 1, 1)
                    new_crop_data["masks"] = torch.cat([masks, repeated_masks], dim=0)
        
        # 添加边缘过渡宽度比例
        if edge_blend > 0:
            new_crop_data["edge_blend"] = edge_blend
        
        return (new_crop_data,new_crop_data["images"],new_crop_data["masks"])


NODE_CLASS_MAPPINGS = {
    "Accurate_mask_clipping": Accurate_mask_clipping,
    "mask_crop_square": mask_crop_square,
    "crop_data_edit": crop_data_edit
}