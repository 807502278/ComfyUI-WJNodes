import torch
import numpy as np
import random
from PIL import Image, ImageOps

from comfy.utils import ProgressBar
import comfy.model_management as mm


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


CATEGORY_NAME = "WJNode/Other-functions"


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


class show_type:
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
                "data": (any,),
                "select_dim":("INT",{"default":0,"min":0,"max":64}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST","INT","INT","INT","INT","INT","INT")
    RETURN_NAMES = ("shape","image-N","image-H","image-W","image-C","sum_count","sel_count",)
    FUNCTION = "element_count"

    def element_count(self, data, select_dim):
        n, n1= 1, 1
        s = [0,0,0,0]
        try:
            s = list(data.shape)
        except:
            print("Warning: This object does not have a shape property, default output is 0")
        #try:
        shape = list(data.shape)
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
    #WJNode/Other-functions
    "any_data": any_data,
    "show_type": show_type,
    "array_count": array_count,
    "get_image_data": get_image_data,
    #WJNode/Other-plugins
    "WAS_Mask_Fill_Region_batch": WAS_Mask_Fill_Region_batch,
    "SegmDetectorCombined_batch": SegmDetectorCombined_batch,
    "bbox_restore_mask": bbox_restore_mask,
    "Sam2AutoSegmentation_data": Sam2AutoSegmentation_data,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #WJNode/Other-functions
    "any_data": "any data", 
    "show_type": "show type",
    "array_count": "array count",
    "get_image_data": "get image data",
    #WJNode/Other-plugins
    "WAS_Mask_Fill_Region_batch": "WAS Mask Fill Region batch",
    "SegmDetectorCombined_batch": "SegmDetectorCombined_batch",
    "bbox_restore_mask": "bbox restore mask",
    "Sam2AutoSegmentation_data": "Sam2AutoSegmentation_data",
}
