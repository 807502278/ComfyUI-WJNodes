import torch
import torch.nn.functional as F
import numpy as np

CATEGORY_NAME = "WJNode/mask_crop"

mask_crop_DefaultOption = {
    "inversion_mask":False,
    "mask_threshold":0.01,
    "size_pix":0,
    "Exceeding":"White",
    "enable_smooth_crop":False,
    "multi_frame_smooth":False,
    "smooth_strength":3,
    "smooth_threshold":0.5,
    "enable_blank_fill":False,
}


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
        功能1：输入图像和遮罩，按遮罩区域的中心裁剪图像和遮罩，输出为方形且固定大小的图像/遮罩/裁剪数据
            注意1：若图像为多张，遮罩为1张则按这个遮罩裁剪所有图像
            注意2：若图像为1张，遮罩为多张则按这些遮罩裁剪这个图像
            注意3：若图像与遮罩批次均大于1且数量不一样则丢弃多的批次
            注意4：若根据遮罩区域的中心计算的裁剪外框超出图像边界，则补充超出边界的部分为Exceeding指定的颜色(含图像和遮罩)
        功能2：输入裁剪数据，将输入的图像按crop_data数据贴回原图像或图像批次
            注意1：输入的任意图像都将拉伸缩放成原裁剪后的大小(无需再重缩放)
            注意2：当输入crop_data时与其无关的参数都将无效
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
            option：用于输入裁剪设置，可忽略，目前仅增加防抖设置
        输出：
            images：原输入图像或图像批次
            masks：原输入遮罩或遮罩批次
            crop_data：用于贴回原图像或图像批次，(包含图像大小/原图/位置信息)
                若crop_data输入不为空则直接返回原crop_data输入
        Function 1:
            Input images and masks, crop the images and masks based on the center of the mask region, and output square and fixed-sized images/masks/crop data.
                Note 1: If there are multiple images and only one mask, crop all images using this single mask.
                Note 2: If there is only one image but multiple masks, crop the image using these multiple masks.
                Note 3: If both the number of images and masks are greater than 1 but are not equal, discard the excess batches.
                Note 4: If the cropping box calculated from the center of the mask region exceeds the image boundary, 
                    fill the exceeded part with the color specified by "Exceeding" (for both image and mask).
        Function 2:
            Input crop data, and paste the input images back to the original image or batch of images based on the crop_data.
                Note 1: Any input image will be stretched and resized to the original cropped size (no need for further rescaling).
                Note 2: When inputting crop_data, all parameters unrelated to it will be invalid.
        Inputs:
            images: Input image or batch of images. If no masks are provided, use the A channel for cropping.
            inversion_mask: Whether to invert the input mask.
            mask_threshold: Threshold for cropping the mask, ignoring light-colored mask areas and fragmented regions.
            size: Specifies the output size (in pixels). If set to 0, use the maximum value of the input mask region.
            Exceeding: Method for filling pixels that exceed the cropping range.
            masks: Input masks used to obtain crop data. If no masks are provided, use the A channel of the images. 
                If the images do not have an A channel, output the original image directly.
            crop_data: If this input is empty, use the cropping function (Function 1). 
                If this input is not empty, paste the images back onto the original image contained in the crop_data. 
                In this case, cropping parameters will be ignored.
            option: Used for input cropping settings, which can be ignored for now. Currently, only anti-shake settings are added.
        Outputs:
            images: The original input image or batch of images.
            masks: The original input mask or batch of masks.
            crop_data: Used for pasting back to the original image or batch of images (contains image size/original image/position information). 
                If crop_data input is not empty, return the original crop_data input directly.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images":("IMAGE",),
                "size": ("FLOAT",{"default":1.0,"min":0.01,"max":300.0,"step":0.01}),
            },
            "optional": {
                "masks":("MASK",),
                "crop_data":("crop_data",),
                "option":("crop_option",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","crop_data")
    RETURN_NAMES = ("crop_images","crop_masks","crop_data")
    FUNCTION = "restore_mask"
    def restore_mask(self, size, images=None, masks=None, crop_data=None, option=None):
        #初始化设置
        print(option)
        print("测试....")
        if option is None:
            option = list(mask_crop_DefaultOption.values())
        else:
            option = list(option.values())
        print(option)
        inversion_mask, mask_threshold, size_pix, Exceeding, enable_smooth_crop, multi_frame_smooth, smooth_strength, smooth_threshold, enable_blank_fill = option

        # 如果没有提供masks，尝试使用图像的alpha通道
        if masks is None:
            if images.shape[3] == 4:  # 有alpha通道
                masks = images[:, :, :, 3]
            else:
                # 没有alpha通道，直接返回原图
                dummy_mask = torch.ones((images.shape[0], images.shape[1], images.shape[2]), device=images.device)
                if crop_data is not None:
                    masks = dummy_mask
                else:
                    print("Error: No alpha channel found in the input images.")
                    return images, dummy_mask, {"images": images, "masks": dummy_mask, "positions": []}

        # 如果提供了crop_data，执行回贴功能
        if crop_data is not None:
            return self._restore_with_crop_data(images, masks, crop_data, mask_threshold)
        
        # 处理填充颜色
        fill_value = 1.0 if Exceeding == "White" else (0.0 if Exceeding == "Black" else 0.5)
        
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
        
        # 计算基础裁剪大小
        final_crop_size = self._calculate_crop_size(masks, mask_threshold)
        
        # 根据size参数调整裁剪大小
        if size > 0.0 and size != 1.0:
            final_crop_size = int(final_crop_size * size)
            if final_crop_size == 0: final_crop_size = 1
        
        # 确保裁剪大小是8的倍数
        #final_crop_size = ((final_crop_size + 7) // 8) * 8
        
        # 获取所有帧的边界框信息
        bbox_info = self._get_all_bboxes(masks, mask_threshold)
        
        # 处理每一帧
        for i in range(batch_size):
            img = images[i]
            mask = masks[i]
            
            # 获取当前帧的裁剪信息
            crop_info = self._get_frame_crop_info(
                i, bbox_info, batch_size, final_crop_size,
                enable_smooth_crop, multi_frame_smooth, smooth_strength, smooth_threshold, enable_blank_fill
            )
            
            # 如果是空白帧且启用了填充
            if crop_info["is_blank"] and enable_blank_fill and batch_size >= 3:
                img, mask = self._fill_blank_frame(i, images, masks, bbox_info)
            
            # 创建填充后的图像和遮罩
            padded_img, padded_mask = self._create_padded_tensors(
                img, mask, crop_info, final_crop_size, fill_value
            )
            
            # 如果指定了size_pix且不为0，则调整大小
            if size_pix > 0 and size_pix != final_crop_size:
                # 调整图像大小
                padded_img = F.interpolate(
                    padded_img.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(size_pix, size_pix),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1).squeeze(0)
                
                # 调整遮罩大小
                padded_mask = F.interpolate(
                    padded_mask.unsqueeze(0).unsqueeze(0),
                    size=(size_pix, size_pix),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            cropped_images.append(padded_img.unsqueeze(0))
            cropped_masks.append(padded_mask.unsqueeze(0))
            positions.append((
                crop_info["crop_min_y"],
                crop_info["crop_min_x"],
                crop_info["crop_max_y"],
                crop_info["crop_max_x"]
            ))
        
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

    def _restore_with_crop_data(self, images, masks, crop_data, mask_threshold):
        """使用crop_data执行回贴功能"""
        # 获取原始图像和位置信息
        original_images = crop_data["images"]
        original_masks = crop_data["masks"]
        positions = crop_data["positions"]
        
        # 创建结果图像和遮罩的副本
        result_images = original_images.clone()
        result_masks = original_masks.clone()
        
        # 确定是否需要处理图像和遮罩
        process_images = images is not None
        process_masks = masks is not None
        
        # 确保批次大小一致
        if process_images:
            batch_size = min(images.shape[0], len(positions))
            images = images[:batch_size]
        elif process_masks:
            batch_size = min(masks.shape[0], len(positions))
            masks = masks[:batch_size]
        else:
            return (result_images, result_masks, crop_data)
        
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
        
        return (result_images, result_masks, crop_data,)

    def _calculate_crop_size(self, masks, mask_threshold):
        """计算裁剪大小"""
        final_crop_size = 0
        for mask in masks:
            thresholded_mask = (mask > mask_threshold).float()
            y_indices, x_indices = torch.nonzero(thresholded_mask, as_tuple=True)
            
            if len(y_indices) > 0:
                min_y, max_y = y_indices.min().item(), y_indices.max().item()
                min_x, max_x = x_indices.min().item(), x_indices.max().item()
                mask_size = max(max_y - min_y, max_x - min_x) + 1
                final_crop_size = max(final_crop_size, mask_size)
        
        return final_crop_size if final_crop_size > 0 else 64

    def _get_all_bboxes(self, masks, mask_threshold):
        """获取所有帧的边界框信息"""
        bbox_info = []
        for mask in masks:
            thresholded_mask = (mask > mask_threshold).float()
            y_indices, x_indices = torch.nonzero(thresholded_mask, as_tuple=True)
            
            if len(y_indices) == 0:
                bbox_info.append({
                    "is_blank": True,
                    "center_y": mask.shape[0] // 2,
                    "center_x": mask.shape[1] // 2
                })
            else:
                min_y, max_y = y_indices.min().item(), y_indices.max().item()
                min_x, max_x = x_indices.min().item(), x_indices.max().item()
                center_y = (min_y + max_y) // 2
                center_x = (min_x + max_x) // 2
                bbox_info.append({
                    "is_blank": False,
                    "center_y": center_y,
                    "center_x": center_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "min_x": min_x,
                    "max_x": max_x
                })
        return bbox_info

    def _get_frame_crop_info(self, idx, bbox_info, batch_size, crop_size, enable_smooth_crop, multi_frame_smooth, smooth_strength, smooth_threshold, enable_blank_fill):
        """获取帧的裁剪信息"""
        info = bbox_info[idx]
        center_y = info["center_y"]
        center_x = info["center_x"]
        
        # 如果启用平滑裁剪且帧数足够
        if enable_smooth_crop and batch_size >= multi_frame_smooth and not info["is_blank"]:
            # 计算需要考虑的帧数范围
            half_window = multi_frame_smooth // 2
            start_idx = max(0, idx - half_window)
            end_idx = min(batch_size - 1, idx + half_window)
            
            # 收集有效的中心点
            valid_centers_y = []
            valid_centers_x = []
            valid_weights = []
            
            for frame_idx in range(start_idx, end_idx + 1):
                frame_info = bbox_info[frame_idx]
                if not frame_info["is_blank"]:
                    # 计算与当前帧的距离
                    frame_diff_y = abs(frame_info["center_y"] - center_y)
                    frame_diff_x = abs(frame_info["center_x"] - center_x)
                    
                    # 只有当坐标差异超过阈值时才进行平滑处理
                    if frame_diff_y > smooth_threshold or frame_diff_x > smooth_threshold:
                        # 计算权重（距离当前帧越远，权重越小）
                        distance = abs(frame_idx - idx)
                        weight = (1 - distance / half_window) * smooth_strength
                        
                        valid_centers_y.append(frame_info["center_y"])
                        valid_centers_x.append(frame_info["center_x"])
                        valid_weights.append(weight)
            
            # 如果有有效的中心点，计算加权平均
            if valid_centers_y:
                total_weight = sum(valid_weights) + (1 - smooth_strength)
                weighted_y = sum(cy * w for cy, w in zip(valid_centers_y, valid_weights))
                weighted_x = sum(cx * w for cx, w in zip(valid_centers_x, valid_weights))
                
                # 将原始中心点也加入计算
                center_y = int((weighted_y + center_y * (1 - smooth_strength)) / total_weight)
                center_x = int((weighted_x + center_x * (1 - smooth_strength)) / total_weight)
        
        # 计算裁剪区域
        half_size = crop_size // 2
        crop_min_y = center_y - half_size
        crop_max_y = center_y + half_size + (crop_size % 2)
        crop_min_x = center_x - half_size
        crop_max_x = center_x + half_size + (crop_size % 2)
        
        return {
            "is_blank": info["is_blank"],
            "crop_min_y": crop_min_y,
            "crop_max_y": crop_max_y,
            "crop_min_x": crop_min_x,
            "crop_max_x": crop_max_x
        }

    def _fill_blank_frame(self, idx, images, masks, bbox_info):
        """填充空白帧"""
        # 查找最近的非空白帧
        prev_idx = idx - 1
        next_idx = idx + 1
        while prev_idx >= 0 and bbox_info[prev_idx]["is_blank"]:
            prev_idx -= 1
        while next_idx < len(bbox_info) and bbox_info[next_idx]["is_blank"]:
            next_idx += 1
        
        # 如果找到了有效帧，使用插值
        if prev_idx >= 0 and next_idx < len(bbox_info):
            weight = (idx - prev_idx) / (next_idx - prev_idx)
            img = images[prev_idx] * (1 - weight) + images[next_idx] * weight
            mask = masks[prev_idx] * (1 - weight) + masks[next_idx] * weight
        elif prev_idx >= 0:
            img = images[prev_idx]
            mask = masks[prev_idx]
        elif next_idx < len(bbox_info):
            img = images[next_idx]
            mask = masks[next_idx]
        else:
            img = images[idx]
            mask = masks[idx]
        
        return img, mask

    def _create_padded_tensors(self, img, mask, crop_info, crop_size, fill_value):
        """创建填充后的张量"""
        padded_img = torch.ones((crop_size, crop_size, img.shape[2]), device=img.device) * fill_value
        padded_mask = torch.zeros((crop_size, crop_size), device=mask.device)
        
        # 计算有效区域
        valid_src_min_y = max(0, crop_info["crop_min_y"])
        valid_src_max_y = min(img.shape[0], crop_info["crop_max_y"])
        valid_src_min_x = max(0, crop_info["crop_min_x"])
        valid_src_max_x = min(img.shape[1], crop_info["crop_max_x"])
        
        # 计算目标区域
        valid_dst_min_y = valid_src_min_y - crop_info["crop_min_y"]
        valid_dst_max_y = valid_dst_min_y + (valid_src_max_y - valid_src_min_y)
        valid_dst_min_x = valid_src_min_x - crop_info["crop_min_x"]
        valid_dst_max_x = valid_dst_min_x + (valid_src_max_x - valid_src_min_x)
        
        # 复制有效区域
        if valid_dst_max_y > valid_dst_min_y and valid_dst_max_x > valid_dst_min_x:
            padded_img[valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = \
                img[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
            padded_mask[valid_dst_min_y:valid_dst_max_y, valid_dst_min_x:valid_dst_max_x] = \
                mask[valid_src_min_y:valid_src_max_y, valid_src_min_x:valid_src_max_x]
        
        return padded_img, padded_mask

    def _paste_single_frame(self, idx, img, mask, pos, result_images, result_masks, mask_threshold, crop_data):
        """粘贴单个帧"""
        if mask is not None:
            # 应用阈值
            thresholded_mask = (mask > mask_threshold).float()
            
            # 获取裁剪区域的位置
            min_y, min_x, max_y, max_x = pos
            
            # 确保坐标在有效范围内
            min_y = max(0, min_y)
            min_x = max(0, min_x)
            max_y = min(result_images.shape[1], max_y)
            max_x = min(result_images.shape[2], max_x)
            
            # 计算源和目标区域的尺寸
            src_height = max_y - min_y
            src_width = max_x - min_x
            
            # 如果源区域有效
            if src_height > 0 and src_width > 0:
                # 计算目标区域在裁剪图像中的位置
                dst_min_y = 0
                dst_min_x = 0
                dst_max_y = src_height
                dst_max_x = src_width
                
                # 更新图像
                result_images[idx, min_y:max_y, min_x:max_x] = \
                    img[dst_min_y:dst_max_y, dst_min_x:dst_max_x]
                
                # 更新遮罩
                result_masks[idx, min_y:max_y, min_x:max_x] = \
                    thresholded_mask[dst_min_y:dst_max_y, dst_min_x:dst_max_x]

class mask_crop_option_SmoothCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_smooth_crop": ("BOOLEAN", {"default": False}),
                "multi_frame_smooth": ("INT", {"default": 3, "min": 3, "max": 51, "step": 2}),
                "smooth_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smooth_threshold": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "enable_blank_fill": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "add_option":("crop_option",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("crop_option",)
    RETURN_NAMES = ("crop_option",)
    FUNCTION = "smooth_crop"
    def smooth_crop(self, enable_smooth_crop, multi_frame_smooth, smooth_strength, smooth_threshold, enable_blank_fill, add_option=None):
        if add_option is None:
            add_option = mask_crop_DefaultOption.copy()
        add_option["enable_smooth_crop"] = enable_smooth_crop
        add_option["multi_frame_smooth"] = multi_frame_smooth
        add_option["smooth_strength"] = smooth_strength
        add_option["smooth_threshold"] = smooth_threshold
        add_option["enable_blank_fill"] = enable_blank_fill
        return (add_option,)

class mask_crop_option_Basic: 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inversion_mask":("BOOLEAN",{"default":False}),
                "mask_threshold": ("FLOAT",{"default":0.01,"min":0.0,"max":1.0,"step":0.001}),
                "size_pix":("INT",{"default":0,"min":0,"max":8192,"step":1}),
                "Exceeding": (["White","Black","Gray"],{"default":"White"}),
            },
            "optional": {
                "add_option":("crop_option",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("crop_option",)
    RETURN_NAMES = ("crop_option",)
    FUNCTION = "smooth_crop"
    def smooth_crop(self, inversion_mask, mask_threshold, size_pix, Exceeding, add_option=None):
        if add_option is None:
            add_option = mask_crop_DefaultOption.copy()
        add_option["inversion_mask"] = inversion_mask
        add_option["mask_threshold"] = mask_threshold
        add_option["size_pix"] = size_pix
        add_option["Exceeding"] = Exceeding
        return (add_option,)

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

class crop_data_CoordinateSmooth:
    DESCRIPTION = """
    功能：平滑crop_data中的坐标数据，使裁剪区域的移动更加平滑
    输入：
        crop_data：包含裁剪数据的字典
        frame_range：平滑窗口大小，用于计算相邻帧的范围
        smooth_strength：平滑强度，控制平滑效果的强度
        smooth_threshold：坐标变化阈值，只有当坐标变化超过阈值时才进行平滑
    输出：
        crop_data：平滑后的裁剪数据
        images：原始图像
        masks：原始遮罩
    
    Function: Smooth the coordinate data in crop_data to make the movement of the cropping area smoother
    Input:
        crop_data: Dictionary containing cropping data
        frame_range: Smoothing window size, used to calculate the range of adjacent frames
        smooth_strength: Smoothing strength, controls the intensity of smoothing effect
        smooth_threshold: Coordinate change threshold, smoothing is only applied when coordinate changes exceed the threshold
    Output:
        crop_data: Smoothed cropping data
        images: Original images
        masks: Original masks
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_data": ("crop_data",),
                "frame_range": ("INT", {"default": 3, "min": 3, "max": 49, "step": 2}),
                "smooth_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smooth_threshold": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "enable_blank_fill": ("BOOLEAN", {"default": False}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("crop_data",)
    RETURN_NAMES = ("crop_data",)
    FUNCTION = "edit_crop_data"
    def edit_crop_data(self, crop_data, frame_range, smooth_strength, smooth_threshold, enable_blank_fill):
        """平滑crop_data中的坐标数据"""
        # 创建crop_data的副本
        new_crop_data = {
            "images": crop_data["images"],
            "masks": crop_data["masks"],
            "positions": crop_data["positions"].copy()
        }
        
        # 获取坐标数据和遮罩数据
        positions = new_crop_data["positions"]
        masks = new_crop_data["masks"]
        batch_size = len(positions)
        
        # 如果帧数不足，直接返回
        if batch_size < frame_range:
            return (new_crop_data,)
        
        # 计算半窗口大小
        half_window = frame_range // 2
        
        # 标记空白帧
        blank_frames = []
        for i in range(batch_size):
            mask = masks[i]
            if mask.mean().item() < 0.01:  # 使用0.01作为阈值判断是否为空白帧
                blank_frames.append(i)
        
        # 如果启用空白帧填充，先处理空白帧
        if enable_blank_fill and blank_frames:
            # 先找到所有非空白帧的坐标
            valid_positions = []
            valid_indices = []
            for i in range(batch_size):
                if i not in blank_frames:
                    valid_positions.append(positions[i])
                    valid_indices.append(i)
            
            # 如果没有有效帧，直接返回
            if not valid_positions:
                return (new_crop_data,)
            
            # 处理每个空白帧
            for blank_idx in blank_frames:
                # 找到最近的两个有效帧
                prev_valid_idx = None
                next_valid_idx = None
                
                # 在有效帧列表中查找
                for idx in valid_indices:
                    if idx < blank_idx:
                        prev_valid_idx = idx
                    elif idx > blank_idx:
                        next_valid_idx = idx
                        break
                
                # 根据前后有效帧进行插值
                if prev_valid_idx is not None and next_valid_idx is not None:
                    # 使用线性插值
                    weight = (blank_idx - prev_valid_idx) / (next_valid_idx - prev_valid_idx)
                    prev_pos = positions[prev_valid_idx]
                    next_pos = positions[next_valid_idx]
                    
                    # 插值计算新位置
                    new_pos = tuple(int(p1 + (p2 - p1) * weight) for p1, p2 in zip(prev_pos, next_pos))
                    positions[blank_idx] = new_pos
                elif prev_valid_idx is not None:
                    # 只有前面有有效帧，使用前面的位置
                    positions[blank_idx] = positions[prev_valid_idx]
                elif next_valid_idx is not None:
                    # 只有后面有有效帧，使用后面的位置
                    positions[blank_idx] = positions[next_valid_idx]
                else:
                    # 如果没有有效帧，使用第一个有效位置
                    positions[blank_idx] = valid_positions[0]
        
        # 对每一帧进行平滑处理
        for i in range(batch_size):
            # 如果是空白帧且未启用填充，跳过平滑处理
            if not enable_blank_fill and i in blank_frames:
                continue
                
            # 获取当前帧的坐标
            min_y, min_x, max_y, max_x = positions[i]
            
            # 计算当前帧的中心点
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            
            # 计算需要考虑的帧数范围
            start_idx = max(0, i - half_window)
            end_idx = min(batch_size - 1, i + half_window)
            
            # 收集有效的中心点
            valid_centers_y = []
            valid_centers_x = []
            valid_weights = []
            
            # 遍历窗口内的帧
            for frame_idx in range(start_idx, end_idx + 1):
                if frame_idx != i:
                    # 如果是空白帧且未启用填充，跳过
                    if not enable_blank_fill and frame_idx in blank_frames:
                        continue
                        
                    # 获取相邻帧的坐标
                    frame_min_y, frame_min_x, frame_max_y, frame_max_x = positions[frame_idx]
                    frame_center_y = (frame_min_y + frame_max_y) // 2
                    frame_center_x = (frame_min_x + frame_max_x) // 2
                    
                    # 计算与当前帧的距离
                    frame_diff_y = abs(frame_center_y - center_y)
                    frame_diff_x = abs(frame_center_x - center_x)
                    
                    # 修改：当坐标差异小于阈值时进行平滑处理
                    if frame_diff_y <= smooth_threshold and frame_diff_x <= smooth_threshold:
                        # 计算权重（距离当前帧越远，权重越小）
                        distance = abs(frame_idx - i)
                        weight = (1 - distance / half_window) * smooth_strength
                        
                        valid_centers_y.append(frame_center_y)
                        valid_centers_x.append(frame_center_x)
                        valid_weights.append(weight)
            
            # 如果有有效的中心点，计算加权平均
            if valid_centers_y:
                total_weight = sum(valid_weights) + (1 - smooth_strength)
                weighted_y = sum(cy * w for cy, w in zip(valid_centers_y, valid_weights))
                weighted_x = sum(cx * w for cx, w in zip(valid_centers_x, valid_weights))
                
                # 将原始中心点也加入计算
                new_center_y = int((weighted_y + center_y * (1 - smooth_strength)) / total_weight)
                new_center_x = int((weighted_x + center_x * (1 - smooth_strength)) / total_weight)
                
                # 计算新的边界框
                half_height = (max_y - min_y) // 2
                half_width = (max_x - min_x) // 2
                
                # 更新坐标
                positions[i] = (
                    new_center_y - half_height,
                    new_center_x - half_width,
                    new_center_y + half_height + ((max_y - min_y) % 2),
                    new_center_x + half_width + ((max_x - min_x) % 2)
                )

        new_crop_data["positions"] = positions
        return (new_crop_data,)


NODE_CLASS_MAPPINGS = {
    "Accurate_mask_clipping": Accurate_mask_clipping,
    "mask_crop_square": mask_crop_square,
    "mask_crop_option_SmoothCrop": mask_crop_option_SmoothCrop,
    "mask_crop_option_Basic": mask_crop_option_Basic,
    "crop_data_edit": crop_data_edit,
    "crop_data_CoordinateSmooth": crop_data_CoordinateSmooth
}