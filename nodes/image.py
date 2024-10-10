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

import folder_paths
import node_helpers

from ..moduel.str_edit import str_edit

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


CATEGORY_NAME_WJnode = "WJNode/Image"

# ------------------image load/save nodes--------------------


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
    CATEGORY = CATEGORY_NAME_WJnode
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image1", "mask1")
    FUNCTION = "load_image"

    def load_image(self, PathFileName):
        image = PathFileName
        # Removes any quotes from Explorer
        image_path = str(image)
        image_path = image_path.replace('"', "")
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
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
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

    CATEGORY = CATEGORY_NAME_WJnode
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
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
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

    CATEGORY = CATEGORY_NAME_WJnode
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

    CATEGORY = CATEGORY_NAME_WJnode
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
                "indexes": ("STRING", {"default": "1,2,", "multiline": True}),
            },
        }
    CATEGORY = CATEGORY_NAME_WJnode
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
CATEGORY_NAME_WJnode = "WJNode/ImageEdit"


class image_cutter:
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
    CATEGORY = CATEGORY_NAME_WJnode
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
                "InvertMask": ("BOOLEAN", {"default": False}),
                "InvertBackMask": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }
    CATEGORY = CATEGORY_NAME_WJnode
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "back_mask")
    FUNCTION = "adv_crop"

    def adv_crop(self, up, down, left, right, Background, InvertMask, InvertBackMask, image=None, mask=None):
        Background_mapping = {
            "White": "White",
            "Black": "Black",
            "Mirror": "reflect",
            "Tile": "circular",
            "Extend": "replicate"
        }
        Background = Background_mapping[Background]

        back_mask = None
        crop_data = np.array([left, right, up, down])
        if image is not None:
            image, back_mask = self.data_processing(
                image, crop_data, back_mask, Background)
        if mask is not None:
            mask, back_mask = self.data_processing(
                mask, crop_data, back_mask, Background)
            if InvertMask:
                mask = 1.0 - mask
        if InvertBackMask and back_mask is not None:
            back_mask = 1.0 - back_mask
        return (image, mask, back_mask)

    def data_processing(self, image, crop_data, back_mask, Background):
        # Obtain image data 获取图像数据
        n, h, w, c, dim, image = self.get_image_data(image)
        shape_hw = np.array([h, h, w, w])

        # Set the crop data value that exceeds the boundary to the boundary value of -1
        # 将超出边界的crop_data值设为边界值-1
        for i in range(crop_data.shape[0]):
            if crop_data[i] >= h:
                crop_data[i] = shape_hw[i]-1

        # Determine whether the height exceeds the boundary 判断高是否超出边界
        if crop_data[0]+crop_data[1] >= h:
            raise ValueError(
                f"The height {crop_data[0]+crop_data[1]} of the cropped area exceeds the size of image {shape[2]}")
        # Determine if the width exceeds the boundary 判断宽是否超出边界
        elif crop_data[2]+crop_data[3] >= w:
            raise ValueError(
                f"The width {crop_data[2]+crop_data[3]} of the cropped area exceeds the size of image {shape[3]}")

        # Separate into cropped and expanded data 分离为裁剪和扩展数据
        extend_data = np.array([0, 0, 0, 0])
        for i in range(crop_data.shape[0]):
            if crop_data[i] < 0:
                extend_data[i] = abs(crop_data[i])
                crop_data[i] = 0

        # Expand the image and mask 扩展背景遮罩
        back_mask_run = False
        if back_mask is None:
            back_mask_run = True
            back_mask = torch.ones(
                (n, h, w), dtype=torch.float32, device=device)
            back_mask = torch.nn.functional.pad(
                back_mask, tuple(extend_data), mode='constant', value=0.0)

        # Expand the image and mask 扩展图像和背景遮罩
            # Filling method during expansion
            # 扩展时的图像填充方式
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
        else:
            image = torch.nn.functional.pad(
                image, extend_data, mode=Background)

        # Crop the image and mask 裁剪图像和背景遮罩
        if dim == 4:
            n, h, w, c = image.shape
            image = image[:,
                          crop_data[2]:h-crop_data[3],
                          crop_data[0]:w-crop_data[1],
                          :]
        else:
            n, h, w = image.shape
            image = image[:,
                          crop_data[2]:h-crop_data[3],
                          crop_data[0]:w-crop_data[1]
                          ]
        if back_mask_run:
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
            if c == 1:  # 最后一维为单通道时应为遮罩
                image = image.squeeze(3)
                dim = 3
                print(f"""warning: Due to the input not being a standard image tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
            elif h == 1 and (w != 1 or c != 1):  # 第2维为单通道时应为遮罩
                image = image.squeeze(1)
                dim = 3
                print(f"""warning: Due to the input not being a standard image/mask tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
            else:
                print(f"Processing standard images:{shape}")
        elif dim == 3:
            n, h, w = shape
            print(f"Processing standard mask:{shape}")
        elif dim == 5:
            n, c, c1, h, w = shape
            if c == 1 and c1 == 1:  # was插件生成的mask批次可能会有此问题
                image = image.squeeze(1)
                image = image.squeeze(1)  # 移除mask批次多余的维度
                dim = 3
                print(f"""warning: Due to the input not being a standard mask tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
        else:  # The image dimension is incorrect 图像维度不正确
            raise ValueError(
                f"The shape of the input image or mask data is incorrect, requiring image n, h, w, c mask n, h, w \nWhat was obtained is{shape}")
        return [n, h, w, c, dim,image]

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
    CATEGORY = CATEGORY_NAME_WJnode
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
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromPath": "Load Image From Path",
    "SaveImageToPath": "Save Image To Path",
    "SaveImageOut": "Save Image Out",
    "SelectImagesBatch": "Select Images Batch",
    "LoadImageAdv": "Load Image Adv",
    "AdvCrop": "Adv Crop",
    "MaskDetection": "Mask Detection",
}
