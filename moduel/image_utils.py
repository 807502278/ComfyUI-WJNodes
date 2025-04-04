import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..moduel.custom_class import any

def prepare_torch_img(img, size_H, size_W, device="cuda", keep_shape=False):
    # [N, H, W, C] -> [N, C, H, W]
    img_new = img.permute(0, 3, 1, 2).to(device)
    img_new = F.interpolate(img_new, (size_H, size_W), mode="bilinear", align_corners=False).contiguous()
    if keep_shape:
        img_new = img_new.permute(0, 2, 3, 1)
    return img_new

def torch_imgs_to_pils(images, masks=None):
    """
        images (torch): [N, H, W, C] or [H, W, C]
        masks (torch): [N, H, W] or [H, W]
    """
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    if masks is not None:
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)
        masks = masks.unsqueeze(3)

        images = torch.cat((images, masks), dim=3)
        mode="RGBA"
    else:
        mode="RGB"

    pil_image_list = [Image.fromarray((images[i].detach().cpu().numpy() * 255).astype(np.uint8), mode=mode) for i in range(images.shape[0])]

    return pil_image_list

def troch_image_dilate(img):
    """
    Remove thin seams on generated texture
        img (torch): [H, W, C]
    """
    import cv2
    img = np.asarray(img.cpu().numpy(), dtype=np.float32)
    img = img * 255
    img = img.clip(0, 255)
    mask = np.sum(img.astype(np.float32), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(np.float32)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = (img.clip(0, 255) / 255).astype(np.float32)
    return torch.from_numpy(img)

def pils_to_torch_imgs(pils: Union[Image.Image, List[Image.Image]], device="cuda"):
    if isinstance(pils, Image.Image):
        pils = [pils]
    
    images = []
    for pil in pils:
        if pil.mode == "RGBA":
            pil = pil.convert('RGB')

        images.append(TF.to_tensor(pil).permute(1, 2, 0))

    images = torch.stack(images, dim=0)

    return images

def pils_rgba_to_rgb(pils: Union[Image.Image, List[Image.Image]], bkgd="WHITE"):
    if isinstance(pils, Image.Image):
        pils = [pils]
    
    rgbs = []
    for pil in pils:
        if pil.mode == 'RGBA':
            new_image = Image.new("RGBA", pil.size, bkgd)
            new_image.paste(pil, (0, 0), pil)
            rgbs.append(new_image.convert('RGB'))
        else:
            rgbs.append(pil)

    return rgbs

def pil_split_image(image, rows=None, cols=None):
    """
        inverse function of make_image_grid
    """
    # image is in square
    if rows is None and cols is None:
        # image.size [W, H]
        rows = 1
        cols = image.size[0] // image.size[1]
        assert cols * image.size[1] == image.size[0]
        subimg_size = image.size[1]
    elif rows is None:
        subimg_size = image.size[0] // cols
        rows = image.size[1] // subimg_size
        assert rows * subimg_size == image.size[1]
    elif cols is None:
        subimg_size = image.size[1] // rows
        cols = image.size[0] // subimg_size
        assert cols * subimg_size == image.size[0]
    else:
        subimg_size = image.size[1] // rows
        assert cols * subimg_size == image.size[0]
    subimgs = []
    for i in range(rows):
        for j in range(cols):
            subimg = image.crop((j*subimg_size, i*subimg_size, (j+1)*subimg_size, (i+1)*subimg_size))
            subimgs.append(subimg)
    return subimgs

def pil_make_image_grid(images, rows=None, cols=None):
    if rows is None and cols is None:
        rows = 1
        cols = len(images)
    if rows is None:
        rows = len(images) // cols
        if len(images) % cols != 0:
            rows += 1
    if cols is None:
        cols = len(images) // rows
        if len(images) % rows != 0:
            cols += 1
    total_imgs = rows * cols
    if total_imgs > len(images):
        images += [Image.new(images[0].mode, images[0].size) for _ in range(total_imgs - len(images))]

    w, h = images[0].size
    grid = Image.new(images[0].mode, size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def pils_erode_masks(mask_list):
    out_mask_list = []
    for idx, mask in enumerate(mask_list):
        arr = np.array(mask)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 1px
        import cv2
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
        alpha = (alpha * 255).astype(np.uint8)
        out_mask_list.append(Image.fromarray(alpha[:, :, None]))

    return out_mask_list

# Retrieve the list of devices recognized by Torch and default devices 
# 获取torch识别到的设备列表和默认设备
def get_device_list():
    device_str = ["default", "cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_str.append(f"cuda:{i}")
    if torch.backends.mps.is_available():
        device_str.append("mps:0")
    n = len(device_str)
    if n > 2:  # default device 默认设备
        device_default = torch.device(device_str[2])
    else:
        device_default = torch.device(device_str[1])
    # Establish a device list dictionary 建立设备列表字典
    device_list = {device_str[0]: device_default, }
    for i in range(n-1):
        device_list[device_str[i+1]] = torch.device(device_str[i+1])

    return [device_list, device_default]


device_list, device_default = get_device_list() #设备列表(字典)，和默认设备(cuda设备类)
device_input  = (list(device_list.keys()), 
                 {"default": list(device_list.keys())[0]}) #节点输入元组，默认default


def tensor_to_pil(image):  # Tensor to PIL
    return Image.fromarray(
        np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil_to_tensor(image):  # PIL to Tensor
    return torch.from_numpy(
        np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil_to_mask(image):  # PIL to Mask
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

def mask_to_pil(mask):  # Mask to PIL
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil

def clean_data(t, type_name="float"): #清理数据中的特殊值nan/posinf/neginf
    # 处理None值
    if t is None:
        return None
    
    # 设置替换值
    replace_values = {
        "float": {"nan": 0.0, "posinf": 1.0, "neginf": 0.0},
        "int": {"nan": 0, "posinf": 255, "neginf": 0},
        "bool": {"nan": False, "posinf": True, "neginf": False},
        "boolean": {"nan": False, "posinf": True, "neginf": False}
    }
    values = replace_values.get(type_name, replace_values["float"])
    
    # 递归处理各种数据类型
    if isinstance(t, (list, tuple, set)):
        # 处理列表和元组
        cleaned = [clean_data(item, type_name) for item in t]
        return type(t)(cleaned)  # 保持原始类型
    
    elif isinstance(t, dict):
        # 处理字典
        return {k: clean_data(v, type_name) for k, v in t.items()}
    
    elif isinstance(t, torch.Tensor):
        # 处理张量
        return torch.nan_to_num(t, 
                              nan=values["nan"], 
                              posinf=values["posinf"], 
                              neginf=values["neginf"])
    
    elif isinstance(t, (int, float, bool)):
        # 处理基本数据类型
        if isinstance(t, bool):
            return bool(values["nan"]) if t != t else t  # 处理nan
        try:
            if np.isnan(t):
                return values["nan"]
            elif np.isinf(t):
                return values["posinf"] if t > 0 else values["neginf"]
            return t
        except:
            return t
    
    # 其他类型直接返回
    return t