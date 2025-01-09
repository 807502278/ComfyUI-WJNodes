from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
# from xml.dom import minidom
import copy
import torch
import numpy as np
import os
import requests
import math
from PIL import Image, ImageOps, ImageSequence
import json

import folder_paths
import node_helpers

from ..moduel.str_edit import str_edit


# Retrieve the list of devices recognized by Torch and default devices 获取torch识别到的设备列表和默认设备
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


device_list, device_default = get_device_list()


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


# ------------------image load/save nodes--------------------
CATEGORY_NAME = "WJNode/Image"


class LoadImageFromPath:
    """
    按路径加载图片
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "PathFileName": ("STRING", {"default": ""})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image1", "mask1")
    FUNCTION = "load_image"

    def load_image(self, PathFileName):
        # Removes any quotes from Explorer
        image_path = PathFileName.replace('"', "")
        i = None
        if image_path.startswith("http"):
            response = requests.get(image_path)
            i = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32,
                               device=device_default)
        return (image, mask)


class LoadImageAdv:
    DESCRIPTION = """
        A load - image node with image - path output and flipped mask.
        带图片路径输出和翻转遮罩的加载图片节点
    """

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "invert": ("BOOLEAN", {"default": False})
            },
        }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load_image"

    def load_image(self, image, invert):
        """
        def to_tensor(image_path):
            from PIL import Image
            image = Image.open(image_path)
            img_np = np.array(image).transpose(2, 0, 1)
            tensor = torch.from_numpy(img_np).float() / 255.0
            return tensor.repeat(1, 1, 1, 1)
        # 简单的加载图片到torch张量
        """

        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros(
                    (64, 64), dtype=torch.float32, device=device_default)
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        if invert:
            output_mask = 1.0 - output_mask

        return (output_image, output_mask, image_path)


class SaveImageToPath:
    """

    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"images": ("IMAGE",),
                 "file_path": ("STRING", {"multiline": True,
                                          "default": "",
                                          "dynamicPrompts": False}),
                 },
                "hidden": {"prompt": "PROMPT",
                           "extra_pnginfo": "EXTRA_PNGINFO"},

                }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE1", "PATH1")
    # OUTPUT_IS_LIST = (True,)
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, file_path, prompt=None, extra_pnginfo=None):
        file_path1 = copy.deepcopy(file_path)
        filename_prefix = os.path.basename(file_path)
        if file_path == '':
            filename_prefix = "ComfyUI"
        filename_prefix, _ = os.path.splitext(filename_prefix)
        _, extension = os.path.splitext(file_path)
        if extension:
            # 是文件名，需要处理
            file_path = os.path.dirname(file_path)
        # filename_prefix=
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        if not os.path.exists(file_path):
            # 使用os.makedirs函数创建新目录
            os.makedirs(file_path)
            print("目录已创建")
        else:
            print("目录已存在")

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # file = f"{filename}_{counter:05}_.png"
            file = f"{filename}.png"

            fp = os.path.join(file_path, file)
            if os.path.exists(fp):
                # file = f"{filename}_{counter:05}_{generate_random_string(8)}.png"
                os.remove(fp)
                fp = os.path.join(file_path, file)
            img.save(os.path.join(file_path, file),
                     compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": file_path,
                "type": self.type
            })
            counter += 1
        return (images, file_path1)


class SaveImageOut:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"images": ("IMAGE", ),
                 "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE1", "PATH1")
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        # images1=copy.deepcopy(images)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        file_path = full_output_folder
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.png"
            fp = os.path.join(file_path, file)
            if os.path.exists(fp):
                # file = f"{filename}_{counter:05}_{generate_random_string(8)}.png"
                os.remove(fp)
                fp = os.path.join(file_path, file)
            img.save(os.path.join(file_path, file),
                     compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": file_path,
                "type": self.type
            })
            counter += 1
        file = f"{file_path}{filename}_{counter:05}_.png"
        return (images, file,)
        # return { "ui": { "images": results }, "IMAGE":image, "PATH":full_output_folder,}


# ------------------image GetData nodes------------------
CATEGORY_NAME = "WJNode/ImageGetData"


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


class SelectImagesBatch:
    DESCRIPTION = """
        返回指定批次编号处的图像(第1张编号为1),
        若输入的编号全部不在范围内则返回原输入，
        超出范围则仅输出符合编号的图像，
        可识别中文逗号。\n

        Returns the image at the specified batch number (the first number is 1). 
        If all input numbers are not within the range, the original input will be returned. 
        If they are outside the range, only images that match the number will be output, 
        and Chinese commas can be recognized.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "1,2,"}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("select_img", "exclude_img")
    FUNCTION = "SelectImages"

    def SelectImages(self, images, indexes):
        select_list = np.array(str_edit.tolist_v2(
            indexes, to_oneDim=True, to_int=True, positive=True))
        select_list1 = select_list[(select_list >= 1) & (
            select_list <= len(images))]-1
        if len(select_list1) < 1:  # 若输入的编号全部不在范围内则返回原输入
            print(
                "Warning:The input value is out of range, return to the original input.")
            return (images, [])
        else:
            exclude_list = np.arange(1, len(images) + 1)-1
            exclude_list = np.setdiff1d(exclude_list, select_list1)  # 排除的图像
            if len(select_list1) < len(select_list):  # 若输入的编号超出范围则仅输出符合编号的图像
                n = abs(len(select_list)-len(select_list1))
                print(
                    f"Warning:The maximum value entered is greater than the batch number range, {n} maximum values have been removed.")
            print(f"Selected the first {select_list1} image")
            return (images[torch.tensor(select_list1, dtype=torch.long)], images[torch.tensor(exclude_list, dtype=torch.long)])


# ------------------image edit nodes------------------
CATEGORY_NAME = "WJNode/ImageEdit"


class image_cutter:  # 开发中
    """
    xy分割图片
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "x": ("INT", {"default": 1, min: 1, "step": 1}),
                "y": ("INT", {"default": 1, min: 1, "step": 1}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "LIST")
    RETURN_NAMES = ("image_batch", "cutter_data")
    FUNCTION = "cutter_image"

    def cutter_image(self, image, x, y):
        image_batch = np.array([])
        n, h, w, c = image.shape
        height = math.ceil(h / y) * y
        width = math.ceil(w / x) * x  # 缩放到xy整倍数以便切割
        image = torch.nn.functional.interpolate(
            image, size=(height, width), mode="bicubic")
        image = np.array(tuple(image))
        if n == 1:
            image_batch[0] = self.cut_image(image[0], x, y)
        else:
            for i in range(n):
                image_batch = image_batch.append(
                    self.cut_image(image[i], x, y))
        print(f"类型：{type(image_batch.shape)}")
        print(f"形状：{image_batch.shape}")
        print(f"形状0:{image_batch[0].shape}")
        image_batch = torch.Size(tuple(image_batch))
        return (image_batch, (n, x, y))

    def cut_image(image, x, y):  # 开发中
        h, w, c = image.shape
        print(f"h:{h},w:{w},c:{c}")
        image_i = np.array([])
        y1 = int(h / y)
        x1 = int(w / x)
        image_batch = []
        print(f"y1:{y1},x1:{x1}----------------")
        for i in range(y):
            print(np.array_split(image[i], x))
        return image_batch


class adv_crop:
    DESCRIPTION = """
    Advanced Crop Node-20241009
        1. Expand and crop (cropping when negative numbers are input) the image or mask up, down, left, and right (can be processed separately or simultaneously).
        2. Synchronously crop the input mask (can be inconsistent with the image resolution).
        3. Adjustable fill color (black/white, white by default), or fill the original image with reflect, circular, or replicate for edge extension.
        4. Output the background mask after the image is expanded (if the background mask of the output mask is required, no image can be input).

        5. If you want to customize the background color or texture, please use the background mask to blend by yourself.
        6. Method for moving an image: Simultaneously expand and crop in one direction (input negative number + positive number).
        7. Note that cropping should not exceed the boundary (especially when the resolution of the mask is different from that of the image), otherwise an error will be reported.

    高级裁剪节点-20241009
        1:上下左右扩展&裁剪(输入负数时为裁剪)图像或遮罩(可单独或同时处理)
        2:同步裁剪输入遮罩(与图像分辨率可以不一致)
        3:可调整填充色(黑/白，默认白色),或填充原图reflect,circular,或边缘扩展replicate
        4:输出图像扩展后的背景遮罩(若需输出遮罩的背景遮罩，则不能输入图像)

        5:自定义背景/纹理：请使用背景遮罩自行混合
        6:移动图片的方法：在一个方向上同时扩展和裁剪(输入负数+正数)
        7:注意裁剪不要超过边界(特别是遮罩与图像的分辨率不一样时)，否则会报错
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "up": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "down": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "left": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "right": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "Background": (["White", "Black", "Mirror", "Tile", "Extend"],),
                "InvertValue": ("BOOLEAN", {"default": False}),
                "InvertMask": ("BOOLEAN", {"default": False}),
                "InvertBackMask": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "back_mask")
    FUNCTION = "adv_crop"

    def adv_crop(self, up, down, left, right, Background, InvertMask, InvertBackMask, InvertValue, image=None, mask=None):
        Background_mapping = {
            "White": "White",
            "Black": "Black",
            "Mirror": "reflect",
            "Tile": "circular",
            "Extend": "replicate"
        }
        # Map fill method names to function parameters 将填充方式名称映射到函数参数
        Background = Background_mapping[Background]

        if InvertValue:
            up = -up
            down = -down
            left = -left
            right = -right

        crop_data = np.array([left, right, up, down])
        back_mask = None

        if image is not None:
            image_data = self.get_image_data(image)
            n, h, w, c, *p = image_data
            extend_separate, crop_separate = self.process_crop_data(
                crop_data, h, w)
            image, back_mask = self.data_processing(
                crop_separate, extend_separate, image_data, back_mask, Background)
            if len(list(image.shape)) == 4 and InvertMask:
                image[..., 3] = 1-image[..., 3]

        if mask is not None:  # 单独处理遮罩与图像，可同时裁剪不同分辨率不同批次的遮罩与图像
            image_data = self.get_image_data(mask)
            n, h, w, *p = image_data
            extend_separate, crop_separate = self.process_crop_data(
                crop_data, h, w)
            mask, back_mask = self.data_processing(
                crop_separate, extend_separate, image_data, back_mask, Background)
            if InvertMask:
                mask = 1.0 - mask
        if InvertBackMask and back_mask is not None:
            back_mask = 1.0 - back_mask
        return (image, mask, back_mask)

    def data_processing(self, crop_data, extend_data, image_data, back_mask, Background):
        # Obtain image data 获取图像数据
        n, h, w, c, dim, image = image_data

        # Expand the image and mask 扩展背景遮罩
        back_mask_run = False
        if back_mask is None:
            back_mask_run = True
            back_mask = torch.ones(
                (n, h, w), dtype=torch.float32, device=image.device)
            back_mask = torch.nn.functional.pad(
                back_mask, tuple(extend_data), mode='constant', value=0.0)

        # Expand the image and mask 扩展图像和背景遮罩
            # Filling method during expansion 扩展时的图像填充方式
        fill_color = 1.0
        if Background == "White":
            Background = "constant"
        elif Background == "Black":
            Background = "constant"
            fill_color = 0.0

            # Extended data varies depending on the image or mask
            # 扩展数据因图像或遮罩而异
        if dim == 4:
            extend_data = tuple(np.concatenate(
                (np.array([0, 0]), extend_data)))
        else:
            extend_data = tuple(extend_data)

            # run Expand the image and mask 运行扩展图像和背景遮罩
        if Background == "constant":
            image = torch.nn.functional.pad(
                image, extend_data, mode=Background, value=fill_color)
            print(f"avd_crop:expand image {extend_data} to color")
        else:
            image = torch.nn.functional.pad(
                image, extend_data, mode=Background)
            print(f"avd_crop:expand image {extend_data} to fill")

        # Crop the image and mask 裁剪图像和背景遮罩
        if dim == 4:
            print("avd_crop:crop image")
            n, h, w, c = image.shape
            image = image[:,
                          crop_data[2]:h-crop_data[3],
                          crop_data[0]:w-crop_data[1],
                          :]
        else:
            print("avd_crop:crop mask")
            n, h, w = image.shape
            image = image[:,
                          crop_data[2]:h-crop_data[3],
                          crop_data[0]:w-crop_data[1]
                          ]
        if back_mask_run:
            print("avd_crop:crop back_mask")
            back_mask = back_mask[:,
                                  crop_data[2]:h-crop_data[3],
                                  crop_data[0]:w-crop_data[1]
                                  ]
        return [image, back_mask]

    # Obtaining and standardizing image data 获取并标准化图像数据
    def get_image_data(self, image):
        shape = image.shape
        dim = image.dim()
        n, h, w, c = 1, 1, 1, 1
        if dim == 4:
            n, h, w, c = shape
            if c == 1:  # When the last dimension is a single channel, it should be a mask 最后一维为单通道时应为遮罩
                image = image.squeeze(3)
                dim = 3
                print(f"""avd_crop warning: Due to the input not being a standard image tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
            elif h == 1 and (w != 1 or c != 1):  # When the second dimension is a single channel, it should be a mask 第2维为单通道时应为遮罩
                image = image.squeeze(1)
                dim = 3
                print(f"""avd_crop warning: Due to the input not being a standard image/mask tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
            else:
                print(f"avd_crop:Processing standard images:{shape}")
        elif dim == 3:
            n, h, w = shape
            print(f"avd_crop:Processing standard mask:{shape}")
        elif dim == 5:
            n, c, c1, h, w = shape
            if c == 1 and c1 == 1:  # The mask batch generated by the was plugin may have this issue WAS插件生成的mask批次可能会有此问题
                image = image.squeeze(1)
                # Remove unnecessary dimensions from mask batch 移除mask批次多余的维度
                image = image.squeeze(1)
                dim = 3
                print(f"""avd_crop warning: Due to the input not being a standard mask tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
        else:  # The image dimension is incorrect 图像维度不正确
            raise ValueError(
                f"avd_crop Error: The shape of the input image or mask data is incorrect, requiring image n, h, w, c mask n, h, w \nWhat was obtained is{shape}")
        return [n, h, w, c, dim, image]

    # Separate cropped data into cropped and expanded data 将裁剪数据分离为裁剪和扩展数据
    def process_crop_data(self, crop_data, h, w):
        shape_hw = np.array([h, h, w, w])
        crop_n = crop_data.shape[0]

        # Set the crops_data value that exceeds the boundary on one side to the boundary value of -1
        # 将单边超出边界的crop_data值设为边界值-1
        for i in range(crop_n):
            if crop_data[i] >= h:
                crop_data[i] = shape_hw[i]-1

        # Determine whether the total height exceeds the boundary 判断总高度是否超出边界
        if crop_data[0]+crop_data[1] >= h:
            raise ValueError(
                f"avd_crop Error:The height {crop_data[0]+crop_data[1]} of the cropped area exceeds the size of image {h}")
        # Determine whether the total width exceeds the boundary 判断总宽度是否超出边界
        elif crop_data[2]+crop_data[3] >= w:
            raise ValueError(
                f"avd_crop Error:The width {crop_data[2]+crop_data[3]} of the cropped area exceeds the size of image {w}")

        # Separate into cropped and expanded data 分离为裁剪和扩展数据
        extend_separate = np.array([0, 0, 0, 0])
        corp_separate = np.copy(extend_separate)
        for i in range(crop_n):
            if crop_data[i] < 0:
                extend_separate[i] = abs(crop_data[i])
            else:
                corp_separate[i] = crop_data[i]
        return [extend_separate, corp_separate]


class mask_detection:
    DESCRIPTION = """
    Input mask and perform duplicate detection:
        1: Whether it exists.
        2: Whether it is a hard edge (binary value).
        3: Whether it is an all-white mask.
        4: Whether it is an all-black mask.
        5: Is it a grayscale mask
        6: Output color value (output 0 when mask is not monochrome)

    输入mask,使用去重检测：
        1:是否存在
        2:是否为硬边缘(二值)
        3:是否为全白遮罩
        4:是否为全黑遮罩
        5:是否为灰度遮罩
        6:输出色值(当mask不为单色时输出0)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN",
                    "BOOLEAN", "BOOLEAN", "int")
    RETURN_NAMES = ("Exist?", "HardEdge", "PureWhite",
                    "PureBlack", "PureGray", "PureColorValue")
    FUNCTION = "mask_detection"

    def mask_detection(self, mask):
        Exist = True
        binary = False
        PureWhite = False
        PureBlack = False
        PureGray = False
        PureColorValue = int(0)
        data = torch.unique(mask).tolist()
        n = len(data)
        if n == 1:
            Exist = False
            PureColorValue = int(round(data[0] * 255))
            if data[0] == 0:
                PureBlack = True
            elif data[0] == 1:
                PureWhite = True
            else:
                PureGray = True
        if n <= 2:
            binary = True
        return (Exist, binary, PureWhite, PureBlack, PureGray, PureColorValue)


class Accurate_mask_clipping:  # 精确查找遮罩bbox边界 (待开发)
    DESCRIPTION = """
    Clip the input mask to the specified range of values
    裁剪输入mask到指定范围的值
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "offset": ("INT", {"default": 0, "min": -8192, "max": 8192}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "accurate_mask_clipping"

    def accurate_mask_clipping(self, mask, offset):
        pass


class invert_channel_adv:
    DESCRIPTION = """
    Reverse the channel of the input image
    反转输入图像的通道。
    """

    @classmethod
    def INPUT_TYPES(s):
        device_select = list(device_list.keys())
        device_select[0] = "Auto"
        device_select.insert(0, "Original")
        return {
            "required": {
                "invert_R": ("BOOLEAN", {"default": False}),
                "invert_G": ("BOOLEAN", {"default": False}),
                "invert_B": ("BOOLEAN", {"default": False}),
                "invert_A": ("BOOLEAN", {"default": False}),
                "device": (device_select, {"default": "Original"}),
            },
            "optional": {
                "RGBA_or_RGB": ("IMAGE",),
                "R": ("MASK",),
                "G": ("MASK",),
                "B": ("MASK",),
                "A": ("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("RGBA", "RGB", "R", "G", "B", "A", "RGBA_Bath")
    FUNCTION = "invert_channel"

    def invert_channel(self, invert_R, invert_G, invert_B, invert_A, device, RGBA_or_RGB=None, R=None, G=None, B=None, A=None):
        # print(f"invert_channel:device:{device}")
        channel_dirt = {"R":R, "G":G, "B":B, "A":A}
        channel_list = list(channel_dirt.values())
        channel_name = list(channel_dirt.keys())
        n_none=channel_list.count(None)
        device_image = None

        # If the input RGBA is not empty, replace the channel 如果输入的RGBA不为空，则替换通道
        if RGBA_or_RGB is not None: 
            _, device_image = self.image_device(RGBA_or_RGB, device)# Device selection 设备选择
            image = RGBA_or_RGB.clone()
            # When the input image has only 3 channels, add an alpha channel 
            # 输入图像只有3通道时，添加一个全1的alpha通道
            if image.shape[3] == 3:
                n, h, w, c = image.shape
                image_A = torch.ones((n, h, w, 1), dtype=torch.float32, device=device_image)
                image = torch.cat((image, image_A), dim=-1)

            # 如果输入的RGBA不为空，则替换通道
            if n_none != 4: 
                for i in range(len(channel_list)):
                    if channel_list[i] is not None:
                        if channel_list[i].shape == image[...,i].shape:
                            image[...,i] = channel_list[i]
                        else:
                            print(f"invert_channel_Warning: The input channel {channel_name [i]} does not match the size of the image. The channel replacement has been skipped!")
                            print(f"invert_channel_警告(CH): 输入的通道 {channel_name[i]} 与image大小不匹配,已跳过该通道替换!")

        # If the input RGBA is empty, combine RGBA into an image 如果输入的RGBA为空，则组合RGBA为图像
        else:
            if n_none == 4: # If both the input image and RGBA are empty, an error will be reported 如果输入image和RGBA都为空，则报错
                raise ValueError(f"invert_channel_Error: No input image was provided!")
            # If the image is empty and RGBA is not completely empty, replace the empty channel with all 0s 
            # 如果image为空,RGBA不全为空，则将空通道替换为全0
            elif n_none != 0: 
                channel_0 = None
                channel_1 = None
                for i in range(len(channel_list)): # Traverse the channel list and find non empty channels 遍历通道列表，找到不为空的通道
                    if channel_list[i] is not None:
                        _, device_image = self.image_device(channel_list[i], device)# Device selection 设备选择
                        channel_0 = torch.zeros(channel_list[i].shape, device=device_image)
                        channel_1 = torch.ones(channel_list[i].shape, device=device_image)
                        break
                # Traverse the channel list and replace empty channels with all 0s or all 1s 
                # 遍历通道列表，将空通道替换为全0或全1
                for i in range(len(channel_list)): 
                    if channel_list[i] is None:
                        if i != 3:
                            channel_list[i] = channel_0
                        else: # If channel A is empty, replace A with all 1s 如果A通道为空，则将A替换为全1
                            channel_list[i] = channel_1
                # Check if the batch quantity is consistent 检测批次数量是否一致
                batch_n = [channel_list[i].shape[0] for i in range(len(channel_list))] 
                if max(batch_n) != min(batch_n): 
                    # Repeat the channel with fewer batches to the channel with more batches 
                    # 将批次数量少的通道重复到批次数量多的通道
                    for i in range(len(channel_list)): 
                        channel_list[i] = channel_list[i].repeat(max(batch_n), 1, 1)
                        print(f"invert_channel_Warning: The input channel batch does not match. The channel {channel_name [i]} batch {channel_name [i]. shape [0]} has been automatically matched to {max (batch_n)}")
                        print(f"invert_channel_警告(CH): 输入的通道批次不匹配，已将通道{channel_name[i]}批次{channel_name[i].shape[0]}自动匹配到{max(batch_n)}")
            # 将通道列表中的每个通道添加一个维度后合成RGBA图像
            try:
                channel_list = [i.unsqueeze(3) for i in channel_list]
                image = torch.cat(channel_list, dim=-1)
            except:
                raise ValueError(f"invert_channel_Error: The input channels do not match the size of the image!")
        
        # Device selection 设备选择
        _, device_image = self.image_device(image, device)

        # Invert the channel 反转通道
        invert = [invert_R, invert_G, invert_B, invert_A]
        if image.shape[3] == 4:
            for i in range(len(invert)):
                if invert[i]:
                    image[..., i] = 1.0 - image[..., i]
        else:
            raise ValueError(
                f"avd_crop Error:The input image should have 3 or 4 dimensions, but got {image.shape}")

        # Separate RGBA images into R, G, B, A 将RGBA图像分离为R, G, B, A
        image_RGBA = [image[...,i] for i in range(int(image.shape[-1]))]
        
        return (image, #RGBA
                image[..., :3], #RGB
                *image_RGBA, #R,G,B,A
                torch.stack(image_RGBA, dim=0) #RGBA_Bath
                )
    def image_device(self, image, device):
        # Device selection 设备选择
        if device == "Auto":
            device_image == device_list["default"]
            image = image.to(device_image)
        elif device == "Original":
            device_image = image.device
        else:
            device_image = device_list[device]
            image = image.to(device_image)
        return [image, device_image]


class image_channel_bus:  # 待开发
    DESCRIPTION = """
    Combine the input images into a single image with the specified channels
    将输入图像合并为具有指定通道的单个图像。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "r": ("MASK",),
                "g": ("MASK",),
                "b": ("MASK",),
                "a": ("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("RGBA", )
    FUNCTION = "image_channel_bus"

    def image_channel_bus(self, image, r, g, b, a):
        image = torch.cat(
            (r, g, b, a), dim=3)
        return (image, )


class RGBABatch_to_image:  # 待开发
    DESCRIPTION = """
    Convert a batch of RGBA images to a single image
    将RGBA遮罩批次转换为单个图像。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_batch": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("RGBA", )
    FUNCTION = "RGBABatch_to_image"

    def RGBABatch_to_image(self, images):
        image_RGBA = torch.cat(
            (images[0], images[1], images[2], images[3]), dim=3)
        return (image_RGBA, )


class to_image_list_data:
    def __init__(self):
        self.image_list = []
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE_LIST_DATA",)
    RETURN_NAMES = ("image_list_data",)
    FUNCTION = "image_list"

    def image_list(self, image1, image2=None):
        if self.image_list == []:
            self.image_list = [image1]
        else:
            self.image_list.append(image1)
        return (self.image_list,)


class merge_image_list:  # 待开发
    DESCRIPTION = """
    Support packaging images in batches/images lists/single images/packaging into image batches (ignoring null objects, used in loops).
    支持将图像批次或图像打包成图像列表，若输入图像列表则(可忽略空对象，用于循环中)
    """

    def __init__(self):
        self.image_list = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
            "optional": {
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "image_list"

    def image_list(self, image1, image2):
        #未完成
        return (self.image_list,)

    #    image_list = [image1, image2, image3, image4]
    #    image_list = self.flatten_array(image_list)
    #    image_list = list(filter(lambda x: x is not None, image_list))
    #    return (image_list,)

    # def flatten_array(self, arr):
    #    flat_arr = []
    #    for i in arr:
    #        if isinstance(i, list):
    #            flat_arr.extend(self.flatten_array(i))
    #        else:
    #            flat_arr.append(i)
    #    return flat_arr


class BilateralFilter:
    DESCRIPTION = """
    Image/Mask Bilateral Filtering: Can repair layered distortion caused by color or brightness scaling in images
    CV2 module is required during runtime
    图像/遮罩双边滤波：可修复图像因颜色或亮度缩放造成的分层失真
    运行时须cv2模块
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diameter":("INT",{"default":30,"min":1,"max":2048}),
                "sigma_color":("FLOAT",{"default":75.0,"min":0.01,"max":256.0}),
                "sigma_space":("FLOAT",{"default":75.0,"min":0.01,"max":1024.0}),
            },
            "optional": {
                "image":("IMAGE",),
                "mask":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "bilateral"
    def bilateral(self, diameter, sigma_color, sigma_space, image = None, mask = None):
        if image is None:
            if mask is None:
                print("Error: Enter at least one of image and mask !")
            else:
                mask = self.Filter_batch(mask,diameter,sigma_color,sigma_space)
                image = mask.unsqueeze(-1).repeat(1,1,1,3)
        else:
            if mask is None:
                image = self.Filter_batch(image,diameter,sigma_color,sigma_space)
                if image.dim == 4:
                    mask = image[...,-1:]
                else:
                    mask = torch.mean(image, dim=-1, keepdim=False)
            else:
                mask = self.Filter_batch(mask,diameter,sigma_color,sigma_space)
                image = self.Filter_batch(image,diameter,sigma_color,sigma_space)
        return (image, mask)

    def Filter_batch(self,image,diameter,sigma_color,sigma_space):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        shape = image.shape
        if shape[-1] == 3 or len(shape) == 3:
            if shape[0] > 1 :
                image_batch = []
                for i in image:
                    image_batch.append(self.bilateralFilter(i,diameter,sigma_color,sigma_space).unsqueeze(0))
                image = torch.cat(image_batch,dim=0)
            else:
                image = self.bilateralFilter(image[0],diameter,sigma_color,sigma_space).unsqueeze(0)
        elif shape[-1] == 4 :
            if shape[0] > 1 :
                image_batch = []
                for i in image:
                    image_batch.append(self.bilateralFilter_RGBA(i,diameter,sigma_color,sigma_space).unsqueeze(0))
                image = torch.cat(image_batch,dim=0)
            else:
                image = self.bilateralFilter_RGBA(image[0],diameter,sigma_color,sigma_space).unsqueeze(0)
        else:
            print("Error: The input is not standard image data, and the original data will be returned !")
        return image
    
    def bilateralFilter(self,image,diameter,sigma_color,sigma_space):
        import cv2
        image = np.array(image * 255).astype('uint8')
        image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        image = torch.tensor(image).float()/255
        return image
    def bilateralFilter_RGBA(self,image,diameter,sigma_color,sigma_space):
        import cv2
        image = np.array(image[...,:-1] * 255).astype('uint8')
        image_A = np.array(image[...,-1:].squeeze(-1) * 255).astype('uint8')
        image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        image_A = cv2.bilateralFilter(image_A, diameter, sigma_color, sigma_space)
        image = torch.cat((torch.tensor(image),torch.tensor(image_A).unsqueeze(-1)),dim=-1).float()/255
        return image

# ------------------video nodes--------------------
CATEGORY_NAME = "WJNode/video"

class Video_fade:
    DESCRIPTION = """
    Support video fade in and fade out
        mask: Local gradient is currently under development
        Exponential: Index gradient development in progress
    支持视频渐入和渐出
        mask:局部渐变正在开发中
        Exponential:指数渐变开发中
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video1": ("IMAGE",),
                "video2": ("IMAGE",),
                "OverlappingFrame": ("INT", {"default": 8, "min": 2, "max": 99999}),
                "method":(["Linear","Cosine","Exponential"],{"default":"Cosine"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","INT")
    RETURN_NAMES = ("video","frames")
    FUNCTION = "fade"
    def fade(self, video1, video2, OverlappingFrame, method="Exponential", mask=None):
        frame = video1.shape[0]+video2.shape[0]-OverlappingFrame
        video = None

        # 如果输入视频有alpha通道，则去除alpha通道
        if video1.shape[3] == 4:
            video1 = video1[...,:3]
        if video2.shape[3] == 4:
            video2 = video2[...,:3]

        # 如果OverlappingFrame为0，则直接合并视频
        if OverlappingFrame == 2:
            video = torch.cat((video1, video2), dim=0)
        elif OverlappingFrame == 3:
            video_temp = video1[-1]*0.5+video2[0]*0.5
            video = torch.cat((video1[:-1], video_temp.unsqueeze(0), video2[1:]), dim=0)
        else:
            Gradient = []
            if method == "Linear":
                Gradient = [i/OverlappingFrame for i in range(OverlappingFrame)]
            elif method == "Cosine":
                Gradient = [0.5+0.5*math.cos(i*math.pi/OverlappingFrame) for i in range(OverlappingFrame)]
            elif method == "Exponential":
                Gradient = [math.exp(-i/OverlappingFrame) for i in range(OverlappingFrame)]
            video_temp = torch.zeros((0,*video1.shape[1:]), device=video1.device)
            for i in range(OverlappingFrame):
                video_temp_0 = video1[i-OverlappingFrame] * Gradient[i] + video2[i] * (1-Gradient[i])
                video_temp = torch.cat((video_temp, video_temp_0.unsqueeze(0)), dim=0)
            video = torch.cat((video1[:-OverlappingFrame], video_temp, video2[OverlappingFrame:]), dim=0)
        return (video, frame)

# The following is a test and has not been imported yet 以下为测试，暂未导入

class SaveImage1:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"images": ("IMAGE", ),
                 "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    CATEGORY = "WJNode/Image"
    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("IMAGE", )
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace(
                "%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file),
                     compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return {"ui": {"images": results}, "IMAGE": images}


NODE_CLASS_MAPPINGS = {
    "LoadImageFromPath": LoadImageFromPath,
    "SaveImageToPath": SaveImageToPath,
    "SaveImageOut": SaveImageOut,
    "SelectImagesBatch": SelectImagesBatch,
    "LoadImageAdv": LoadImageAdv,
    "AdvCrop": adv_crop,
    "MaskDetection": mask_detection,
    "InvertChannelAdv": invert_channel_adv,
    "ToImageListData": to_image_list_data,
    "MergeImageList": merge_image_list,
    # "ImageChannelBus": image_channel_bus,
    # "RGBABatchToImage": RGBABatch_to_image,
    "BilateralFilter": BilateralFilter,
    "VideoFade": Video_fade,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromPath": "Load Image From Path",
    "SaveImageToPath": "Save Image To Path",
    "SaveImageOut": "Save Image Out",
    "SelectImagesBatch": "Select Images Batch",
    "LoadImageAdv": "Load Image Adv",
    "AdvCrop": "Adv Crop",
    "MaskDetection": "Mask Detection",
    "InvertChannelAdv": "Invert Channel Adv",
    "ToImageListData": "To Image List Data",
    "MergeImageList": "Merge Image List",
    # "ImageChannelBus": "Image Channel Bus",
    # "RGBABatchToImage": "RGBA Batch To Image",
    "BilateralFilter": "Bilateral Filter",
    "VideoFade": "Video Fade",
}
