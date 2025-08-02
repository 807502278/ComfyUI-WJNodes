from io import BytesIO
from copy import copy
import os
from math import gcd
from functools import reduce

import requests
from tqdm import tqdm
# from xml.dom import minidom

import numpy as np
from PIL import Image, ImageOps, ImageSequence, ImageFilter
import torch
import torch.nn.functional as F

import folder_paths
import node_helpers
from ..moduel.image_utils import device_list, device_default, clean_data
from ..moduel.custom_class import any
from ..moduel.str_edit import is_safe_eval


# ------------------image load/save nodes--------------------
CATEGORY_NAME = "WJNode/ImageFile"

class Load_Image_From_Path:
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

class Load_Image_Adv:
    DESCRIPTION = """
        A load - image node with image - path output and flipped mask.
        Support jpg, png, jpeg, webp, tiff, bmp, gif, ico, svg format
        Can obtain images inside subfolders and individual images within the input directory.

        带图片路径输出和翻转遮罩的加载图片节点
        默认扫描 jpg,png,jpeg,webp,tiff,bmp,gif,ico,svg 格式
        可获取input内子文件夹内的图像和单个图像
        
    """
    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
        self.files = []
        self.allowed_extensions = ['.jpg', '.png', '.jpeg', '.webp', '.tiff', '.bmp', '.gif', '.ico', '.svg']

    @classmethod
    def INPUT_TYPES(cls):
        instance = cls()
        instance.traverse_directory(instance.input_dir)
        return {
            "required": {
                "image": (sorted(instance.files), 
                          {"image_upload": True}),
                "invert_mask": ("BOOLEAN", {"default": False})
            },
        }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load_image"

    def load_image(self, image, invert_mask):
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
                try:
                    w = image.size[0]
                    h = image.size[1]
                except:
                    w = 64; h = 64
                mask = torch.zeros(
                    (h, w), dtype=torch.float32, device=device_default)
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        if invert_mask:
            output_mask = 1.0 - output_mask

        return (output_image, output_mask, image_path)

    def traverse_directory(self, directory):
        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isdir(full_path):
                self.traverse_directory(full_path)  # 递归遍历子文件夹
            elif os.path.isfile(full_path):
                ext = os.path.splitext(f)[-1].lower()
                if ext in [e.lower() for e in self.allowed_extensions]:
                    relative_path = os.path.relpath(full_path, self.input_dir)
                    self.files.append(relative_path)

class Save_Image_To_Path:
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
                 "file_path": ("STRING", {"multiline": True,"default": "","dynamicPrompts": False}),
                 },
                }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "PATH")
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, file_path):
        file_path1 = copy.deepcopy(file_path)
        filename_prefix = os.path.basename(file_path)
        if file_path == '':
            filename_prefix = "ComfyUI"
        filename_prefix, _ = os.path.splitext(filename_prefix)
        _, extension = os.path.splitext(file_path)

        # 是文件名，需要处理
        if extension: file_path = os.path.dirname(file_path)

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        if not os.path.exists(file_path): os.makedirs(file_path)

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

class Save_Image_Out:
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
                }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "PATH")
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, filename_prefix="ComfyUI"):
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

class image_url_download:
    DESCRIPTION = """图像URL下载节点
    输入说明：
        image_urls:图像URL列表，或单个图像URL字符串
        timeout_single:单个图像下载超时时间(秒)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_urls": (any,),
                "timeout_single": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 3600.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "download_images"
    CATEGORY = CATEGORY_NAME

    def download_images(self, image_urls, timeout_single):
        empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        """下载图像并转换为ComfyUI格式的张量"""
        if not image_urls:
            print("Error: image URLs input is none !")
            return (empty_image,)

        if isinstance(image_urls, str):
            image_urls = [image_urls]
        if not isinstance(image_urls, list):
            print("Error: image_urls input is not a list or string !")
            return (empty_image,)
        print(f"📥 Downloading {len(image_urls)} image(s) with {timeout_single}s timeout per image...")

        images = []
        Error_n = 0
        for i, url in tqdm(enumerate(image_urls),):
            try:
                # 下载图像，使用用户指定的超时时间
                response = requests.get(url, timeout=timeout_single)
                response.raise_for_status()
                image_bytes = BytesIO(response.content)
                pil_image = Image.open(image_bytes)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image_array = np.array(pil_image, dtype=np.float32) / 255.0
                images.append(image_array)
                
            except Exception as e:
                #print(f"❌ Error downloading image {i+1}: {e}")
                Error_n += 1
                continue

        if not images:
            print("No images were successfully downloaded")
            # 返回空的图像张量
            return (empty_image,)

        # 检查所有图像的尺寸，如果不同则调整为统一尺寸
        if len(images) > 1:
            print(f"✅ Successfully downloaded {len(images)} image")
            if Error_n > 0:
                print(f"❌ {Error_n} image(s) failed to download")
            # 找到最大的高度和宽度
            max_height = max(img.shape[0] for img in images)
            max_width = max(img.shape[1] for img in images)

            # 调整所有图像到相同尺寸
            resized_images = []
            for i, img in enumerate(images):
                if img.shape[0] != max_height or img.shape[1] != max_width:
                    # 使用PIL进行resize
                    pil_img = Image.fromarray((img * 255).astype(np.uint8))
                    pil_img = pil_img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                    resized_img = np.array(pil_img, dtype=np.float32) / 255.0
                    resized_images.append(resized_img)
                    print(f"Resized image {i+1} from {img.shape} to {resized_img.shape}")
                else:
                    resized_images.append(img)
            images = resized_images

        # 转换为torch张量，维度顺序为(批次，高，宽，通道)
        images_tensor = torch.stack([torch.from_numpy(img) for img in images])
        #print(f"Final tensor shape: {images_tensor.shape}, dtype: {images_tensor.dtype}")
        #print(f"Batch size: {images_tensor.shape[0]} images")

        return (images_tensor,)


# ------------------image crop nodes------------------
CATEGORY_NAME = "WJNode/ImageCrop"

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

# ------------------image edit nodes------------------
CATEGORY_NAME = "WJNode/ImageEdit"

class invert_channel_adv: #
    DESCRIPTION = """
    Functionality:
    Channel Reorganization: Includes inversion, creation, and replacement of channels, with batch support.
    Channel Operations: Channel separation, channel batching (can be used for calculations that only support masks).
    Batch Matching: If different batch sizes are input, it will attempt to automatically match batches (often fails).

    Input: (Resolution must be the same, batch support is available, try to keep batch sizes consistent)
    RGBA_or_RGB: Input image. If channels are also input, the corresponding channels of this image will be replaced.
    RGBA_Bath: Input RGBA channel batch, used to recombine RGBA channel batches into an image. 
                If this is input, it will ignore the input of RGBA_or_RGB.
    RGB_Bath: Input RGB channel batch, used to recombine RGB channel batches into an image. 
                If RGBA_or_RGB is input, it will replace the RGB channels of the input image.
    R/G/B/A: Input channels (all channels must be of the same size). 
                If only channels are input without an image, an image will be created based on the channels.
    (Replacement Priority: Later channel data will replace earlier ones. 
                R/G/B/A > RGB_Bath > RGBA_Bath > RGBA_or_RGB)

    Output: (All outputs are after replacement, inversion, and other operations)
    RGBA: Output RGBA image, batch support is available. Channels without data will be black.
    RGB: Output RGB image, batch support is available.
    R/G/B/A: Output individual RGBA channels.
    RGB_Bath: Output RGB channel batch, can be used for calculations that only support masks.
    RGBA_Bath: Output RGBA channel batch, can be used for calculations that only support masks.

    功能：
    通道重组：含反转/新建/替换通道，支持批次
    通道操作：分离通道，转通道批次(可用于将图片进行仅支持遮罩的计算)
    批次匹配：若输入了不同批次大小，会简单尝试自动匹配批次(多半会失败)

    输入：(分辨率必须一样，支持批次，批次大小尽量一样)
    RGBA_or_RGB：输入图像，若同时输入通道，则该图像的对应通道会被替换，
    RGBA_Bath：输入RGBA通道批次，用于将RGBA通道批次重新组合为图像，
            若此处有输入则会忽略RGBA_or_RGB输入
    RGB_Bath：输入RGB通道批次，用于将RGB通道批次重新组合为图像，
            若RGBA_or_RGB有输入则替换输入的图像rgb通道
    R/G/B/A：输入通道(所有通道须大小一样)，
            若仅输入通道不输入图像则根据通道新建图像
    (替换优先级：后面的通道数据会被前面的替换
            R/G/B/A > RGB_Bath > RGBA_Bath > RGBA_or_RGB )

    输出：(所有输出均为替换/反转等操作后的)
    RGBA：输出RGBA图像，支持批次，若某通道无数据将为黑色
    RGB：输出RGB图像，支持批次
    R/G/B/A：输出RGBA单独的通道
    RGB_Bath：输出RGB通道批次，可用于将图片进行仅支持遮罩的计算
    RGBA_Bath：输出RGBA通道批次，可用于将图片进行仅支持遮罩的计算\
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
                "RGBA_Bath": ("MASK",),
                "RGB_Bath": ("MASK",),
                "R": ("MASK",),
                "G": ("MASK",),
                "B": ("MASK",),
                "A": ("MASK",),
            },
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("RGBA", "RGB", "R", "G", "B", "A", "RGB_Bath", "RGBA_Bath")
    FUNCTION = "invert_channel"

    def invert_channel(self, 
                       invert_R, invert_G, invert_B, invert_A, 
                       device, 
                       RGBA_or_RGB=None, RGBA_Bath=None, RGB_Bath=None, 
                       R=None, G=None, B=None, A=None):
        #初始化通道数据
        channel_dirt = {"R":None, "G":None, "B":None, "A":None}
        channel_name = list(channel_dirt.keys())

        #要替换的通道-RGBA_Bath
        if RGBA_Bath is not None:
            n = RGBA_Bath.shape[0]
            if n % 4 == 0:
                m = int(n / 4)
                for i in range(4):
                    channel_dirt[channel_name[i]] = RGBA_Bath[i*m:(i+1)*m,...]
            else:
                print("Warning: RGBA_Cath mask batch input not RGBA detected, this input will be skipped !")
                print("警告：检测到RGBA_Bath遮罩批次输入不为RGBA，将跳过此输入!")

        #要替换的通道-RGB_Bath
        if RGB_Bath is not None:
            n = RGB_Bath.shape[0]
            if n % 3 == 0:
                m = int(n / 3)
                for i in range(3):
                    channel_dirt[channel_name[i]] = RGB_Bath[i*m:(i+1)*m,...]
            else:
                print("Warning: RGBA_Cath mask batch input not RGBA detected, this input will be skipped !")
                print("警告：检测到RGB_Bath遮罩批次输入不为RGB，将跳过此输入!")

        #要替换的单通道-RGBA
        channel_dirt_temp = {"R":R, "G":G, "B":B, "A":A}
        channel_list_temp = list(channel_dirt_temp.values())
        for i in range(4):
            if channel_list_temp[i] is not None:
                channel_dirt[channel_name[i]] = channel_list_temp[i]

        #要替换最终通道
        channel_list = list(channel_dirt.values())
        n_none=channel_list.count(None)
        device_image = None

        # If the input RGBA is not empty, replace the channel 
        # 如果输入的RGBA不为空，则替换通道
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
                if RGBA_Bath is not None:
                    raise ValueError(f"invert_channel_Error: Input RGBA_Cath is not an RGBA batch data !\ninvert_channel_错误:输入RGBA_Bath不是RGBA批次数据！")
                elif RGB_Bath is not None:
                    raise ValueError(f"invert_channel_Error: Input RGB_Cath is not an RGB batch data !\ninvert_channel_错误:输入RGB_Bath不是RGB批次数据！")
                else:
                    raise ValueError(f"invert_channel_Error: No input image was provided !\ninvert_channel_错误:未输入任何图像数据！")
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
                        print(f"invert_channel_Warning: The input channel batch does not match. The channel {channel_name [i]} batch {image.shape [0]} has been automatically matched to {max (batch_n)}")
                        print(f"invert_channel_警告(CH): 输入的通道批次不匹配，已将通道{channel_name[i]}批次{image.shape[0]}自动匹配到{max(batch_n)}")
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
        image_RGB = image_RGBA[0:int(len(image_RGBA)/4*3)]

        return (image, #RGBA
                image[..., :3], #RGB
                *image_RGBA, #R,G,B,A
                torch.cat(image_RGB, dim=0), #RGB_Bath
                torch.cat(image_RGBA, dim=0) #RGBA_Bath
                )
    
    def image_device(self, image, device):
        # Device selection 设备选择
        if device == "Auto":
            device_image = device_list["default"]
            image = image.to(device_image)
        elif device == "Original":
            device_image = image.device
        else:
            device_image = device_list[device]
            image = image.to(device_image)
        return [image, device_image]

class ListMerger:
    def __init__(self):
        self.list1_buffer = []
        self.list2_buffer = []
        self.current_index = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list1": ("*",),
                "list2": ("*",),
                "is_last": ("BOOLEAN", {"default": False}),  # 标记是否是最后一个元素
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("merged_list",)
    OUTPUT_IS_LIST = False
    FUNCTION = "merge_lists"
    CATEGORY = CATEGORY_NAME

    def merge_lists(self, list1, list2, is_last):
        # 将当前元素添加到缓冲区
        self.list1_buffer.append(list1)
        self.list2_buffer.append(list2)
        
        if is_last:
            # 最后一个元素时，返回完整的合并列表
            result = self.list1_buffer + self.list2_buffer
            # 清空缓冲区，为下一次操作做准备
            self.list1_buffer = []
            self.list2_buffer = []
            return (result,)
        else:
            # 不是最后一个元素时，返回None或空列表
            return ([],)

class Bilateral_Filter:
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
                    image_batch.append(self.Bilateral_Filter(i,diameter,sigma_color,sigma_space).unsqueeze(0))
                image = torch.cat(image_batch,dim=0)
            else:
                image = self.Bilateral_Filter(image[0],diameter,sigma_color,sigma_space).unsqueeze(0)
        elif shape[-1] == 4 :
            if shape[0] > 1 :
                image_batch = []
                for i in image:
                    image_batch.append(self.Bilateral_Filter_RGBA(i,diameter,sigma_color,sigma_space).unsqueeze(0))
                image = torch.cat(image_batch,dim=0)
            else:
                image = self.Bilateral_Filter_RGBA(image[0],diameter,sigma_color,sigma_space).unsqueeze(0)
        else:
            print("Error: The input is not standard image data, and the original data will be returned !")
        return image
    
    def Bilateral_Filter(self,image,diameter,sigma_color,sigma_space):
        import cv2
        image = np.array(image * 255).astype('uint8')
        image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        image = torch.tensor(image).float()/255
        return image
    def Bilateral_Filter_RGBA(self,image,diameter,sigma_color,sigma_space):
        import cv2
        image = np.array(image[...,:-1] * 255).astype('uint8')
        image_A = np.array(image[...,-1:].squeeze(-1) * 255).astype('uint8')
        image = cv2.Bilateral_Filter(image, diameter, sigma_color, sigma_space)
        image_A = cv2.Bilateral_Filter(image_A, diameter, sigma_color, sigma_space)
        image = torch.cat((torch.tensor(image),torch.tensor(image_A).unsqueeze(-1)),dim=-1).float()/255
        return image

class image_math:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation":(["-","+","*","&"],{"default":"-"}),
                "algorithm":(["cv2","torch"],{"default":"cv2"}),
                "invert1":("BOOLEAN",{"default":False}),
                "invert2":("BOOLEAN",{"default":False}),
            },
            "optional": {
                "image1":("IMAGE",),
                "image2":("IMAGE",),
                "mask1":("MASK",),
                "mask2":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK")
    FUNCTION = "image_math"
    def image_math(self, operation, algorithm, invert1, invert2,
                    mask1=None, mask2=None, image1=None, image2=None):
        mask,image = None,None

        #invert mask
        mask1, image1 = self.invert([mask1, image1],invert1)
        mask2, image2 = self.invert([mask2, image2],invert2)

        #统一数量
        mask1, mask2 = self.repeat_mask(mask1, mask2)
        image1, image2 = self.repeat_mask(image1, image2)

        #check cv2
        if algorithm == "cv2":
            try: import cv2
            except:
                print("prompt-mask_and_mask_math: cv2 is not installed, Using Torch")
                print("prompt-mask_and_mask_math: cv2 未安装, 使用torch")
                algorithm = "torch"

        #如果mask只输入一个，则输出这个
        if mask1 is None:
            if mask2 is None: pass
            else: mask = mask2
        else :
            if mask2 is None: mask = mask1
            else: mask = self.math(algorithm,operation,mask1,mask2)[0]

        #如果image只输入一个，则输出这个
        if image1 is None:
            if image2 is None: pass
            else: image = image2
        else:
            if image2 is None: image = image1
            else: image = self.math(algorithm,operation,image1,image2)[0]

        return (mask,image)
    
    #批量翻转
    def invert(self,data,inv):
        if isinstance(data,list):
            data = [1-i if inv and i is not None else i for i in data]
        else:
            if inv and data is not None:
                data = 1-data
        return data

    #某个输入为单张时统一数量
    def repeat_mask(self,mask1,mask2):
        if mask1 is None:
            if mask2 is None: return None,None
            else: return mask2,mask2
        if mask1.shape[0] == 1 and mask2.shape[0] != 1:
            mask1 = mask1.repeat(mask2.shape[0],1,1)
        elif mask1.shape[0] != 1 and mask2.shape[0] == 1:
            mask2 = mask2.repeat(mask1.shape[0],1,1)
        return mask1,mask2

    #按模式调用计算
    def math(self,algorithm,operation,mask1,mask2):
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

    #4种计算
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

class image_math_value:
    DESCRIPTION = """
    expression: expression
    clamp: If you want to continue with the next image_math_ralue, 
            it is recommended not to open it
    Explanation: The A channel of the image will be automatically removed, 
            and the shape will be the data shape
    Note: This node has been deprecated, please use image_math-value-v1

    expression:表达式
    clamp:如果要继续进行下一次image_math_value建议不打开
    说明：会自动去掉image的A通道，shape为数据形状
    注意：此节点已被弃用，请使用image_math_value_v1
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression":("STRING",{"default":"a+b","multiline": True}),
                "clamp":("BOOLEAN",{"default":True}),
            },
            "optional": {
                "a":("IMAGE",),
                "b":("IMAGE",),
                "c":("MASK",),
                "d":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","LIST")
    RETURN_NAMES = ("image","mask","shape")
    FUNCTION = "image_math"
    def image_math(self,expression,clamp,
                   a=None, b=None, c=None, d=None):
        image = None
        mask = None
        s = 0

        # 获取所有输入的尺寸并计算目标尺寸
        shapes = []
        inputs = {'a': a, 'b': b, 'c': c, 'd': d}
        for v in inputs.values():
            if v is not None:
                shapes.append(v.shape[1:3])  # 只取高度和宽度
        
        if shapes:
            target_height = min(s[0] for s in shapes)
            target_width = min(s[1] for s in shapes)

            # 处理每个输入
            for k, v in inputs.items():
                if v is not None:
                    # 去掉A通道
                    if k in ['a', 'b'] and v.shape[-1] == 4:
                        v = v[..., 0:-1]
                    
                    # 调整尺寸
                    if v.shape[1] != target_height or v.shape[2] != target_width:
                        v = self._resize_tensor(v, target_height, target_width)
                    
                    # 遮罩转3通道
                    if k in ['c', 'd']:
                        v = v.unsqueeze(-1).expand(-1, -1, -1, 3)
                    
                    inputs[k] = v

        # 更新局部变量
        a, b, c, d = inputs['a'], inputs['b'], inputs['c'], inputs['d']

        try:
            local_vars = locals().copy()
            exec(f"image = {expression}", {}, local_vars)
            image = local_vars.get("image")
        except:
            print("Warning: Invalid expression !, will output null value.")

        if image is not None:
            if clamp:
                image = torch.clamp(image, 0.0, 1.0)
            s = list(image.shape)
            mask = torch.mean(image, dim=3, keepdim=False)
        return (image,mask,s)

    def _resize_tensor(self, tensor, target_height, target_width):
        """调整张量尺寸"""
        if tensor.dim() == 3:  # 处理mask
            tensor = tensor.unsqueeze(-1)
            tensor = F.interpolate(
                tensor.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).squeeze(-1)
        else:  # 处理image
            tensor = F.interpolate(
                tensor.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        return tensor

class Robust_Imager_Merge:
    """具有一定鲁棒性的图像和遮罩拼接节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_direction": (["right", "left", "up", "down"], {"default": "right"}),
                "auto_scale": ("BOOLEAN", {"default": True}),
                "output_mode": (["merge", "input1", "input2"], {"default": "merge"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "merge_images"
    CATEGORY = CATEGORY_NAME

    def merge_images(self, merge_direction, auto_scale, output_mode,
                    image1=None, image2=None, mask1=None, mask2=None):
        """拼接图像和遮罩"""

        # 检查输入
        inputs = [image1, image2, mask1, mask2]
        input_count = sum(1 for x in inputs if x is not None)

        if input_count == 0:
            raise ValueError("至少需要一个输入（image或mask）")

        # 处理output_mode为input1或input2的情况
        if output_mode == "input1":
            if image1 is not None:
                image_out = image1
                mask_out = self._create_white_mask_like_image(image1)
            elif mask1 is not None:
                image_out = self._mask_to_image(mask1)
                mask_out = mask1
            else:
                raise ValueError("output_mode为input1时，必须有image1或mask1输入")
            return (image_out, mask_out)

        elif output_mode == "input2":
            if image2 is not None:
                image_out = image2
                mask_out = self._create_white_mask_like_image(image2)
            elif mask2 is not None:
                image_out = self._mask_to_image(mask2)
                mask_out = mask2
            else:
                raise ValueError("output_mode为input2时，必须有image2或mask2输入")
            return (image_out, mask_out)

        # merge模式的处理逻辑
        return self._process_merge_mode(merge_direction, auto_scale,
                                      image1, image2, mask1, mask2)

    def _process_merge_mode(self, merge_direction, auto_scale,
                           image1, image2, mask1, mask2):
        """处理merge模式的逻辑"""

        # 统计输入类型
        has_image1 = image1 is not None
        has_image2 = image2 is not None
        has_mask1 = mask1 is not None
        has_mask2 = mask2 is not None

        image_count = sum([has_image1, has_image2])
        mask_count = sum([has_mask1, has_mask2])

        # 单独输入逻辑
        if image_count == 1 and mask_count == 0:
            # 只有一个image输入
            img = image1 if has_image1 else image2
            return (img, self._create_white_mask_like_image(img))

        elif image_count == 0 and mask_count == 1:
            # 只有一个mask输入
            mask = mask1 if has_mask1 else mask2
            return (self._mask_to_image(mask), mask)

        # 混合输入逻辑
        elif image_count == 1 and mask_count == 1:
            # 一个image和一个mask
            img = image1 if has_image1 else image2
            mask = mask1 if has_mask1 else mask2

            # 将mask转为黑白图像与image拼接
            mask_as_image = self._mask_to_image(mask)
            merged_image = self._merge_tensors(img, mask_as_image, merge_direction, auto_scale)

            # 将image转为mask与原mask拼接
            image_as_mask = self._image_to_mask(img)
            merged_mask = self._merge_tensors(image_as_mask, mask, merge_direction, auto_scale)

            return (merged_image, merged_mask)

        elif image_count == 2 and mask_count == 0:
            # 两个image
            merged_image = self._merge_tensors(image1, image2, merge_direction, auto_scale)

            # 检查Alpha通道逻辑
            has_alpha1 = image1.shape[-1] == 4
            has_alpha2 = image2.shape[-1] == 4

            if has_alpha1 and has_alpha2:
                # 都有Alpha通道，mask输出为拼接后的Alpha通道
                alpha1 = image1[..., 3:4]  # 保持维度
                alpha2 = image2[..., 3:4]
                merged_alpha = self._merge_tensors(alpha1, alpha2, merge_direction, auto_scale)
                merged_mask = merged_alpha.squeeze(-1)  # 移除最后一个维度变成mask格式
            else:
                # 没有Alpha通道或只有一个有，输出白色mask
                merged_mask = self._create_white_mask_like_image(merged_image)

            return (merged_image, merged_mask)

        elif image_count == 0 and mask_count == 2:
            # 两个mask
            merged_mask = self._merge_tensors(mask1, mask2, merge_direction, auto_scale)
            merged_image = self._mask_to_image(merged_mask)
            return (merged_image, merged_mask)

        elif image_count == 2 and mask_count == 1:
            # 两个image和一个mask
            merged_image = self._merge_tensors(image1, image2, merge_direction, auto_scale)
            mask = mask1 if has_mask1 else mask2
            return (merged_image, mask)

        elif image_count == 1 and mask_count == 2:
            # 一个image和两个mask
            img = image1 if has_image1 else image2
            merged_mask = self._merge_tensors(mask1, mask2, merge_direction, auto_scale)
            return (img, merged_mask)

        elif image_count == 2 and mask_count == 2:
            # 全部输入
            merged_image = self._merge_tensors(image1, image2, merge_direction, auto_scale)
            merged_mask = self._merge_tensors(mask1, mask2, merge_direction, auto_scale)
            return (merged_image, merged_mask)

        else:
            raise ValueError("未处理的输入组合")

    def _create_white_mask_like_image(self, image):
        """创建与图像相同大小的白色遮罩"""
        batch, height, width = image.shape[:3]
        return torch.ones((batch, height, width), dtype=torch.float32)

    def _mask_to_image(self, mask):
        """将遮罩转换为黑白图像"""
        # mask格式: (batch, height, width)
        # image格式: (batch, height, width, channels)
        # 创建RGB图像，所有通道都使用mask的值
        image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        return image

    def _image_to_mask(self, image):
        """将图像转换为遮罩（灰度）"""
        # image格式: (batch, height, width, channels)
        # mask格式: (batch, height, width)
        if image.shape[-1] == 1:
            # 灰度图像
            return image.squeeze(-1)
        elif image.shape[-1] >= 3:
            # RGB或RGBA图像，转换为灰度
            # 使用标准的灰度转换权重: 0.299*R + 0.587*G + 0.114*B
            weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
            gray = torch.sum(image[..., :3] * weights, dim=-1)
            return gray
        else:
            raise ValueError(f"不支持的图像通道数: {image.shape[-1]}")

    def _merge_tensors(self, tensor1, tensor2, direction, auto_scale):
        """拼接两个张量"""
        # 检查批次维度
        batch1 = tensor1.shape[0]
        batch2 = tensor2.shape[0]

        if batch1 != batch2:
            if batch1 == 1:
                # tensor1是单批次，扩展到与tensor2相同的批次
                tensor1 = tensor1.repeat(batch2, 1, 1, *([1] * (len(tensor1.shape) - 3)))
            elif batch2 == 1:
                # tensor2是单批次，扩展到与tensor1相同的批次
                tensor2 = tensor2.repeat(batch1, 1, 1, *([1] * (len(tensor2.shape) - 3)))
            else:
                raise ValueError(f"批次数量不匹配且都不为1: {batch1} vs {batch2}")

        # 处理Alpha通道
        tensor1, tensor2 = self._handle_alpha_channels(tensor1, tensor2)

        # 自动缩放
        if auto_scale:
            tensor2 = self._auto_scale_tensor(tensor1, tensor2, direction)

        # 执行拼接
        if direction == "right":
            return torch.cat([tensor1, tensor2], dim=2)  # 沿宽度拼接
        elif direction == "left":
            return torch.cat([tensor2, tensor1], dim=2)  # 沿宽度拼接
        elif direction == "down":
            return torch.cat([tensor1, tensor2], dim=1)  # 沿高度拼接
        elif direction == "up":
            return torch.cat([tensor2, tensor1], dim=1)  # 沿高度拼接
        else:
            raise ValueError(f"不支持的拼接方向: {direction}")

    def _handle_alpha_channels(self, tensor1, tensor2):
        """处理Alpha通道逻辑"""
        # 只对图像张量处理Alpha通道（4维张量）
        if len(tensor1.shape) != 4 or len(tensor2.shape) != 4:
            return tensor1, tensor2

        channels1 = tensor1.shape[-1]
        channels2 = tensor2.shape[-1]

        # 如果都有Alpha通道，保持不变
        if channels1 == 4 and channels2 == 4:
            return tensor1, tensor2

        # 如果都没有Alpha通道，保持不变
        if channels1 == 3 and channels2 == 3:
            return tensor1, tensor2

        # 如果一个有Alpha通道一个没有，去除Alpha通道
        if channels1 == 4 and channels2 == 3:
            tensor1 = tensor1[..., :3]  # 去除Alpha通道
        elif channels1 == 3 and channels2 == 4:
            tensor2 = tensor2[..., :3]  # 去除Alpha通道

        return tensor1, tensor2

    def _auto_scale_tensor(self, tensor1, tensor2, direction):
        """自动缩放tensor2以匹配tensor1的对应边，保持原比例"""
        current_height = tensor2.shape[1]
        current_width = tensor2.shape[2]

        if direction in ["left", "right"]:
            # 左右拼接，需要匹配高度，按比例缩放宽度
            target_height = tensor1.shape[1]

            if target_height != current_height:
                # 计算缩放比例
                scale_ratio = target_height / current_height
                new_width = int(current_width * scale_ratio)

                # 使用插值缩放
                if len(tensor2.shape) == 4:  # 图像
                    # (batch, height, width, channels) -> (batch, channels, height, width)
                    tensor2_scaled = tensor2.permute(0, 3, 1, 2)
                    tensor2_scaled = torch.nn.functional.interpolate(
                        tensor2_scaled, size=(target_height, new_width),
                        mode='bilinear', align_corners=False
                    )
                    # (batch, channels, height, width) -> (batch, height, width, channels)
                    tensor2 = tensor2_scaled.permute(0, 2, 3, 1)
                else:  # 遮罩
                    # (batch, height, width) -> (batch, 1, height, width)
                    tensor2_scaled = tensor2.unsqueeze(1)
                    tensor2_scaled = torch.nn.functional.interpolate(
                        tensor2_scaled, size=(target_height, new_width),
                        mode='bilinear', align_corners=False
                    )
                    # (batch, 1, height, width) -> (batch, height, width)
                    tensor2 = tensor2_scaled.squeeze(1)

        elif direction in ["up", "down"]:
            # 上下拼接，需要匹配宽度，按比例缩放高度
            target_width = tensor1.shape[2]

            if target_width != current_width:
                # 计算缩放比例
                scale_ratio = target_width / current_width
                new_height = int(current_height * scale_ratio)

                # 使用插值缩放
                if len(tensor2.shape) == 4:  # 图像
                    # (batch, height, width, channels) -> (batch, channels, height, width)
                    tensor2_scaled = tensor2.permute(0, 3, 1, 2)
                    tensor2_scaled = torch.nn.functional.interpolate(
                        tensor2_scaled, size=(new_height, target_width),
                        mode='bilinear', align_corners=False
                    )
                    # (batch, channels, height, width) -> (batch, height, width, channels)
                    tensor2 = tensor2_scaled.permute(0, 2, 3, 1)
                else:  # 遮罩
                    # (batch, height, width) -> (batch, 1, height, width)
                    tensor2_scaled = tensor2.unsqueeze(1)
                    tensor2_scaled = torch.nn.functional.interpolate(
                        tensor2_scaled, size=(new_height, target_width),
                        mode='bilinear', align_corners=False
                    )
                    # (batch, 1, height, width) -> (batch, height, width)
                    tensor2 = tensor2_scaled.squeeze(1)

        return tensor2

class image_scale_pixel_v2:
    DESCRIPTION = """
    图像按像素缩放
    旨在自动缩放图像至wan视频所需大小，及控制总像素以避免oom
    输入说明：
        TotalPixels：缩放后的总像素(接近),单位：百万像素，
                0.32对应伪720P视频，1对应720P视频，2对应1080P视频，8.3对应4K视频
        alignment：缩放后的宽度和高度将为alignment的倍数
                wan视频一般情况可为64的倍数，vace/i2v等需要为128的倍数
    输出说明：
        遮罩和图像各自缩放，尺寸互不影响
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TotalPixels": ("FLOAT", {"max": 512.0, "min": 0.0001, "step": 0.0001,"default": "1.0"}),
                "alignment": (["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"], {"default": "32"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "option": ("SO",), #新增输入设置
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "scale_image"
    CATEGORY = CATEGORY_NAME

    def scale_image(self, TotalPixels, alignment, images=None, masks=None, option=None):
        # 初始化设置
        option_dict = {"TotalPixels": TotalPixels, 
                       "alignment": alignment, 
                       "size_mode": "crop", 
                       "crop_bbox_method_width": "center",
                       "crop_bbox_method_hight": "center",
                       "method": "LANCZOS",
                       "scale":"original",
                       "fill_color":"#ffffff",
                       "round_width": "round",
                       "round_hight": "round",
                       "Enable_TotalPixels":True,
                       }
        # 更新设置
        if option is not None:
            option_dict.update(option)

        # 获取参数
        target_pixels = option_dict["TotalPixels"] * 1000000 # 将百万像素转换为实际像素数
        alignment = int(option_dict["alignment"]) # 转换alignment对齐值为整数
        size_mode = option_dict["size_mode"] # 缩放时的对齐方式
        crop_bbox_method_width = option_dict["crop_bbox_method_width"] # size_mode为crop或bbox时的宽度对齐的坐标基准
        crop_bbox_method_hight = option_dict["crop_bbox_method_hight"] # size_mode为crop或bbox的高度对齐的坐标基准
        round_width = option_dict["round_width"] # size_mode为crop或bbox时的宽度数值的舍入方式(四舍五入/向上/向下)
        round_hight = option_dict["round_hight"] # size_mode为crop或bbox时的高度数值的舍入方式(四舍五入/向上/向下)
        method = option_dict["method"] # 图像缩放算法
        scale = option_dict["scale"] # 图像/遮罩大小的比例字符串(宽:高)，如果为original则使用输入原始图像/遮罩的大小
        fill_color = option_dict["fill_color"] # size_mode为bbox(扩展到对齐的大小)时的填充色
        # 是否启用总像素限制，如果不启用则按原图像/遮罩裁剪到指定比例(如果为original则不裁剪)后对齐到alignment指定的参数(如果为1则不处理)
        Enable_TotalPixels = option_dict["Enable_TotalPixels"]

        # 处理图像
        processed_images = None
        if images is not None:
            processed_images = self._scale_tensor(images, target_pixels, alignment, size_mode, crop_bbox_method_width, crop_bbox_method_hight, round_width, round_hight, method, scale, fill_color, Enable_TotalPixels, is_image=True)
        # 处理遮罩
        processed_masks = None
        if masks is not None:
            processed_masks = self._scale_tensor(masks, target_pixels, alignment, size_mode, crop_bbox_method_width, crop_bbox_method_hight, round_width, round_hight, method, scale, fill_color, Enable_TotalPixels, is_image=False)
        return (processed_images, processed_masks)

    def _scale_tensor(self, tensor, target_pixels, alignment, size_mode, crop_bbox_method_width, crop_bbox_method_hight, round_width, round_hight, method, scale, fill_color, Enable_TotalPixels, is_image=True):
        """缩放张量到目标像素数"""
        import math
        if tensor is None:
            return None
        # 获取原始尺寸
        original_height = tensor.shape[1]
        original_width = tensor.shape[2]
        original_pixels = original_height * original_width

        if Enable_TotalPixels:
            # 启用总像素限制：按总像素数计算目标尺寸
            # 1. 根据scale参数确定目标比例
            if scale == "original":
                # 使用输入图像的原始宽:高比例
                target_ratio = original_width / original_height
            else:
                # 解析scale参数的比例字符串(宽:高)
                try:
                    width_ratio, height_ratio = map(float, scale.split(':'))
                    target_ratio = width_ratio / height_ratio
                except:
                    # 如果解析失败，回退到原始比例
                    target_ratio = original_width / original_height

            # 1.1 根据target_pixels和目标比例计算目标尺寸
            # target_pixels = target_width * target_height
            # target_ratio = target_width / target_height
            # 所以: target_height = sqrt(target_pixels / target_ratio)
            #      target_width = target_height * target_ratio
            target_height = int(math.sqrt(target_pixels / target_ratio))
            target_width = int(target_height * target_ratio)
        else:
            # 不启用总像素限制：按原图像/遮罩裁剪到指定比例后对齐
            if scale == "original":
                # 如果为original则不裁剪，直接使用原始尺寸
                target_width = original_width
                target_height = original_height
            else:
                # 解析scale参数的比例字符串(宽:高)，按此比例裁剪原图像
                try:
                    width_ratio, height_ratio = map(float, scale.split(':'))
                    target_ratio = width_ratio / height_ratio

                    # 按比例裁剪：选择较小的缩放比例，确保完全包含
                    if original_width / original_height > target_ratio:
                        # 原图更宽，按高度计算
                        target_height = original_height
                        target_width = int(original_height * target_ratio)
                    else:
                        # 原图更高，按宽度计算
                        target_width = original_width
                        target_height = int(original_width / target_ratio)
                except:
                    # 如果解析失败，使用原始尺寸
                    target_width = original_width
                    target_height = original_height
        # 对齐到指定倍数
        if alignment > 1:
            # 根据round_width选择宽度舍入方式
            if round_width == "ceil":
                target_width = ((target_width + alignment - 1) // alignment) * alignment
            elif round_width == "floor":
                target_width = (target_width // alignment) * alignment
            else:  # round_width == "round"
                target_width = round(target_width / alignment) * alignment

            # 根据round_hight选择高度舍入方式
            if round_hight == "ceil":
                target_height = ((target_height + alignment - 1) // alignment) * alignment
            elif round_hight == "floor":
                target_height = (target_height // alignment) * alignment
            else:  # round_hight == "round"
                target_height = round(target_height / alignment) * alignment
        # 确保最小尺寸
        target_height = max(target_height, alignment)
        target_width = max(target_width, alignment)
        # 2. 根据size_mode处理
        if size_mode == "fill":
            # 拉伸模式：直接拉伸到目标尺寸
            final_height = target_height
            final_width = target_width
            crop_needed = False
            bbox_needed = False
        elif size_mode == "bbox":
            # bbox模式：等比缩放后外扩填充到目标尺寸
            # 选择较小的缩放比例，确保图像完全包含在目标尺寸内
            width_scale = target_width / original_width
            height_scale = target_height / original_height
            scale = min(width_scale, height_scale)

            final_width = int(original_width * scale)
            final_height = int(original_height * scale)
            crop_needed = False
            bbox_needed = True
        else:  # size_mode == "crop"
            # 裁剪模式：等比缩放后裁剪
            crop_needed = True
            bbox_needed = False
            # 3.1: 按宽度对齐到目标宽度等比缩放
            width_scale = target_width / original_width
            scaled_height_by_width = int(original_height * width_scale)
            if scaled_height_by_width > target_height:
                # 3.1: 高度超出，按宽度缩放后裁剪高度
                final_width = target_width
                final_height = scaled_height_by_width
                crop_dim = "height"
                crop_target = target_height
            elif scaled_height_by_width == target_height:
                # 完全匹配，不需要裁剪
                final_width = target_width
                final_height = target_height
                crop_needed = False
            else:
                # 3.2: 高度没有超出，按高度缩放后裁剪宽度
                height_scale = target_height / original_height
                scaled_width_by_height = int(original_width * height_scale)
                final_height = target_height
                final_width = scaled_width_by_height
                crop_dim = "width"
                crop_target = target_width
        # 执行缩放
        if is_image:
            # 图像张量 (batch, height, width, channels)
            tensor_scaled = tensor.permute(0, 3, 1, 2)  # (batch, channels, height, width)
            # 选择插值模式
            mode_map = {
                "LANCZOS": "bicubic",
                "BILINEAR": "bilinear",
                "BICUBIC": "bicubic",
                "BOX": "nearest",
                "HAMMING": "bilinear",
                "NEAREST": "nearest"
            }
            interpolation_mode = mode_map.get(method, "bilinear")
            tensor_scaled = torch.nn.functional.interpolate(
                tensor_scaled,
                size=(final_height, final_width),
                mode=interpolation_mode,
                align_corners=False if interpolation_mode != "nearest" else None
            )
            # 如果需要裁剪，根据crop_bbox_method进行裁剪
            if crop_needed:
                if crop_dim == "height":
                    # 裁剪高度，根据crop_bbox_method_hight确定起始位置
                    if crop_bbox_method_hight == "up":
                        start_h = 0
                    elif crop_bbox_method_hight == "down":
                        start_h = final_height - crop_target
                    else:  # center
                        start_h = (final_height - crop_target) // 2
                    tensor_scaled = tensor_scaled[:, :, start_h:start_h + crop_target, :]
                else:  # crop_dim == "width"
                    # 裁剪宽度，根据crop_bbox_method_width确定起始位置
                    if crop_bbox_method_width == "left":
                        start_w = 0
                    elif crop_bbox_method_width == "right":
                        start_w = final_width - crop_target
                    else:  # center
                        start_w = (final_width - crop_target) // 2
                    tensor_scaled = tensor_scaled[:, :, :, start_w:start_w + crop_target]

            # 如果需要bbox填充，进行外扩填充
            if bbox_needed:
                # 解析fill_color
                try:
                    fill_color_hex = fill_color.lstrip('#')
                    r = int(fill_color_hex[0:2], 16) / 255.0
                    g = int(fill_color_hex[2:4], 16) / 255.0
                    b = int(fill_color_hex[4:6], 16) / 255.0
                    fill_rgb = [r, g, b]
                except:
                    fill_rgb = [1.0, 1.0, 1.0]  # 默认白色

                # 创建目标尺寸的画布
                batch_size = tensor_scaled.shape[0]
                canvas = torch.full((batch_size, 3, target_height, target_width), 0.0, dtype=tensor_scaled.dtype, device=tensor_scaled.device)
                for i in range(3):
                    canvas[:, i, :, :] = fill_rgb[i]

                # 计算放置位置
                if crop_bbox_method_width == "left":
                    start_w = 0
                elif crop_bbox_method_width == "right":
                    start_w = target_width - final_width
                else:  # center
                    start_w = (target_width - final_width) // 2

                if crop_bbox_method_hight == "up":
                    start_h = 0
                elif crop_bbox_method_hight == "down":
                    start_h = target_height - final_height
                else:  # center
                    start_h = (target_height - final_height) // 2

                # 将缩放后的图像放置到画布上
                canvas[:, :, start_h:start_h + final_height, start_w:start_w + final_width] = tensor_scaled
                tensor_scaled = canvas

            tensor_scaled = tensor_scaled.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        else:
            # 遮罩张量 (batch, height, width)
            tensor_scaled = tensor.unsqueeze(1)  # (batch, 1, height, width)
            tensor_scaled = torch.nn.functional.interpolate(
                tensor_scaled,
                size=(final_height, final_width),
                mode="bilinear",
                align_corners=False
            )
            # 如果需要裁剪，根据crop_bbox_method进行裁剪
            if crop_needed:
                if crop_dim == "height":
                    # 裁剪高度，根据crop_bbox_method_hight确定起始位置
                    if crop_bbox_method_hight == "up":
                        start_h = 0
                    elif crop_bbox_method_hight == "down":
                        start_h = final_height - crop_target
                    else:  # center
                        start_h = (final_height - crop_target) // 2
                    tensor_scaled = tensor_scaled[:, :, start_h:start_h + crop_target, :]
                else:  # crop_dim == "width"
                    # 裁剪宽度，根据crop_bbox_method_width确定起始位置
                    if crop_bbox_method_width == "left":
                        start_w = 0
                    elif crop_bbox_method_width == "right":
                        start_w = final_width - crop_target
                    else:  # center
                        start_w = (final_width - crop_target) // 2
                    tensor_scaled = tensor_scaled[:, :, :, start_w:start_w + crop_target]

            # 如果需要bbox填充，进行外扩填充
            if bbox_needed:
                # 创建目标尺寸的画布，遮罩用0填充（黑色）
                batch_size = tensor_scaled.shape[0]
                canvas = torch.zeros((batch_size, 1, target_height, target_width), dtype=tensor_scaled.dtype, device=tensor_scaled.device)

                # 计算放置位置
                if crop_bbox_method_width == "left":
                    start_w = 0
                elif crop_bbox_method_width == "right":
                    start_w = target_width - final_width
                else:  # center
                    start_w = (target_width - final_width) // 2

                if crop_bbox_method_hight == "up":
                    start_h = 0
                elif crop_bbox_method_hight == "down":
                    start_h = target_height - final_height
                else:  # center
                    start_h = (target_height - final_height) // 2

                # 将缩放后的遮罩放置到画布上
                canvas[:, :, start_h:start_h + final_height, start_w:start_w + final_width] = tensor_scaled
                tensor_scaled = canvas

            tensor_scaled = tensor_scaled.squeeze(1)  # (batch, height, width)
        return tensor_scaled

class image_scale_pixel_option:
    DESCRIPTION = """
    图像按像素缩放的高级选项
        默认参数将和image_scale_pixel_v2节点不输入设置时保持一致
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale": (["original", "1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "original"}),
                "size_mode": (["fill", "crop","bbox"], {"default": "fill"}),
                "crop_bbox_method_width": (["center", "left", "right"], {"default": "center"}),
                "crop_bbox_method_hight": (["center", "up", "down"], {"default": "center"}),
                "fill_color":("STRING",{"default":"#ffffff"}),
                "round_width": (["round", "ceil", "floor"], {"default": "round"}),
                "round_hight": (["round", "ceil", "floor"], {"default": "round"}),
                "method": (["LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING", "NEAREST"], {"default":"LANCZOS"}),
                "Enable_TotalPixels":("BOOLEAN",{"default":True}),
            },
            "optional": {
                "refer_to":(any,),
            }
        }

    RETURN_TYPES = ("SO",)
    RETURN_NAMES = ("scale_pixel_option")
    FUNCTION = "scale_image"
    CATEGORY = CATEGORY_NAME

    def scale_image(self, scale, size_mode, crop_bbox_method_width, crop_bbox_method_hight, fill_color, round_width, round_hight, method, Enable_TotalPixels, refer_to=None):
        original_size = scale
        if refer_to is not None :
            # 检测refer_to的类型
            if isinstance(refer_to, torch.Tensor):
                # 检测到为图像类
                original_size = self._cast_to_str(refer_to.shape[1:3])
            elif isinstance(refer_to, dict):
                # 检测是否为latent
                if "samples" in refer_to and isinstance(refer_to["samples"], torch.Tensor):
                    if "type" in refer_to and refer_to["type"] == "audio":
                        print("Warning: The refer_to input is a audio-latent, Cannot obtain aspect ratio, reference input will be ignored")
                    elif refer_to["samples"].dim() == 4:
                        original_size = self._cast_to_str(refer_to["samples"].shape[2:4])
                    else:
                        print("Warning: The refer_to input is unknown samples data , Cannot obtain aspect ratio, reference input will be ignored")
                else:
                    print("Warning1: The refer_to input is not a image/mask/image-latent, please check your input")
            else:
                print("Warning0: The refer_to input is not a image/mask/image-latent, please check your input")
        return ({"scale": original_size,
                 "size_mode": size_mode,
                 "crop_bbox_method_width": crop_bbox_method_width,
                 "crop_bbox_method_hight": crop_bbox_method_hight,
                 "fill_color": fill_color,
                 "round_width": round_width,
                 "round_hight": round_hight,
                 "method": method,
                 "Enable_TotalPixels": Enable_TotalPixels,
                 },)
    
    # 输出约分后的比例
    def _cast_to_str(self, data):
        data = np.array(data)
        h,w = (data/reduce(gcd, data.tolist())).astype(int)
        return str(w)+":"+str(h)


# ------------------Math------------------
CATEGORY_NAME = "WJNode/Math"

class any_math:
    DESCRIPTION = """
        expression: Advanced expression
        Function: Perform numerical calculations on images using expressions.
            The mask will be treated as a 3-channel image.
        Input:
            - a~j: Arbitrary inputs as variables
            - expression1~3: Enter mathematical expressions. The results will be correspondingly output to 1 - 3. Spaces are allowed.
            - PreprocessingTensor: Do you want to preprocess the image 
                (if you need to perform operations on the image, please open it)
                Preprocessing images will unify batches/sizes/channels, 
                allowing image and mask operations to be performed
            - RGBA_to_RGB: When turned on, if the input image is 4-channel, 
                it will be converted to 3-channel.
            - clamp: Limit the image values to 0-1. 
                It is not recommended to turn it on if you want to continue with the next image_math_value.
        Instructions:
            1. Note that if the expression result is not an image (tensor), 
                the result will be output to the corresponding value. 
                In this case, the image and mask outputs will be empty. 
                Conversely, the value will be empty.
            2. Support some Python methods, please pay attention to the output type
            3. Leave the expressions for the outputs you don't need empty to ignore unnecessary calculations. 
            4. 中文说明请看 -any_math_v2-
    """
    required = {
                "expression":("STRING",{"default":"a+b","multiline": True}),
                "PreprocessingTensor":("BOOLEAN",{"default":True}),
                "RGBA_to_RGB":("BOOLEAN",{"default":True}),
                "image_clamp":("BOOLEAN",{"default":True}),
                }
    optional = {"a":(any,),"b":(any,),"c":(any,),"d":(any,)}

    def __init__(self):
        self.clamp = True
        self.tensors = {}

    @classmethod
    def INPUT_TYPES(s):
        return {"required":s.required,
                "optional":s.optional}
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK",any)
    RETURN_NAMES = ("image1","mask1","value1")
    FUNCTION = "image_math"

    def image_math(self,expression,PreprocessingTensor,image_clamp,RGBA_to_RGB,**kwargs):
        #初始化值
        self.clamp = image_clamp
        if PreprocessingTensor: 
            self.process_input(RGBA_to_RGB,kwargs)
        else: 
            self.tensors = kwargs
        return (*self.handle_img(expression,1),)

    def process_input(self, RGBA_to_RGB, kwargs):
        try:
            # 获取有效的张量输入
            tensor_inputs = {k: v for k, v in kwargs.items() if v is not None and isinstance(v, torch.Tensor)}
            if not tensor_inputs:
                self.tensors = {k: v for k, v in kwargs.items() if v is not None}
                return

            # 获取张量形状信息并计算目标尺寸
            shapes = {k: v.shape for k, v in tensor_inputs.items()}
            batch_sizes = []
            for s in shapes.values():
                if len(s) >= 1:  # 确保张量至少有一个维度
                    batch_sizes.append(s[0])
                else:
                    batch_sizes.append(1)

            target_batch = max(batch_sizes)
            target_height = min(s[1] for s in shapes.values() if len(s) > 1)
            target_width = min(s[2] for s in shapes.values() if len(s) > 2)
            target_channels = 3 if (max((s[-1] for s in shapes.values() if len(s) >= 3), default=3) != 4 or RGBA_to_RGB) else 4

            # 检查批次大小
            multi_batches = {b for b in batch_sizes if b > 1}
            if len(multi_batches) > 1:
                print(f"Warning: Multiple different batch sizes detected {multi_batches} May affect the calculation results!")

            # 处理所有输入
            for k, v in kwargs.items():
                if v is None:
                    continue

                if isinstance(v, torch.Tensor):
                    # 处理维度
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    elif v.dim() == 3 and v.shape[-1] != target_channels:
                        v = v.unsqueeze(-1)

                    # 批次处理
                    if v.shape[0] < target_batch:
                        repeat_times = target_batch // v.shape[0]
                        if repeat_times * v.shape[0] == target_batch:
                            v = v.repeat(repeat_times, *(1 for _ in range(v.dim()-1)))
                        else:
                            v = v.expand(target_batch, *(v.shape[i] for i in range(1, v.dim())))

                    # 尺寸调整
                    if v.dim() >= 3 and (v.shape[1] != target_height or v.shape[2] != target_width):
                        if v.dim() == 3:
                            v = v.unsqueeze(-1)
                            v = self._resize_tensor(v, target_height, target_width)
                            v = v.squeeze(-1)
                        else:
                            v = self._resize_tensor(v, target_height, target_width)

                    # 通道处理
                    if v.dim() == 3:
                        v = v.unsqueeze(-1).expand(-1, -1, -1, target_channels)
                    elif v.dim() == 4:
                        if v.shape[-1] == 4 and RGBA_to_RGB:
                            v = v[..., :target_channels]
                        elif v.shape[-1] == 1:
                            v = v.expand(-1, -1, -1, target_channels)

                    self.tensors[k] = v
                else:
                    self.tensors[k] = v

        except Exception as e:
            print(f"Error details:{str(e)}")
            raise ValueError("Preprocessing failed! Please check the image input, such as whether the quantity is consistent(any_math)")

    def _resize_tensor(self, tensor, target_height, target_width):
        """调整张量尺寸"""
        return F.interpolate(
            tensor.permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)

    def _process_channels(self, tensor, target_channels, RGBA_to_RGB):
        """处理张量通道"""
        if tensor.dim() == 3:
            return tensor.unsqueeze(-1).expand(-1, -1, -1, target_channels)
        elif tensor.dim() == 4:
            return tensor[..., :target_channels] if (tensor.shape[-1] == 4 and RGBA_to_RGB) else tensor
        else:
            print(f"警告：输入张量维度异常！(维度：{tensor.dim()})")
            return tensor

    def handle_img(self,expression,n=1):
        #初始化输出
        mask,image,data = None,None,None
        #检查字符串
        try:
            if expression not in [""," ",None]:
                safe, reason = is_safe_eval(expression)
                if safe: image = eval(expression, {}, self.tensors)
                else: print("warn: "+ reason + ",Ignore this expression"+ str(n))
        except:
            print(f"Error: Expression{n} error! (image_math_value_v2)")

        #若结果不为空则进行处理
        if image is not None:
            try:
                if isinstance(image,torch.Tensor): #若输出为张量(图像类)
                    if self.clamp:#是否钳制到0-1
                        image = torch.clamp(image, 0.0, 1.0) 
                    if image.dim() == 3: #如果结果是遮罩，将遮罩作为通道转为图像输出
                        mask = image
                        image = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
                        print("Warning: The result may be a mask. (image_math_value_v2)")
                    elif image.dim() == 4: #如果结果是图像，将图像以求均值的方式压缩为遮罩
                        mask = torch.mean(image, dim=3, keepdim=False)
                else:
                    data = image #若输出为非张量直接返回
            except:
                print("Warning: You have calculated non image data! (image_math_value_v2)")
        return (image,mask,data)

class any_math_v2(any_math):
    DESCRIPTION = """
        expression:高级表达式
            功能：使用表达式对图像进行数值计算，mask将被视为3通道图像
        输入：
            a~j：输入任意作为变量
            expression：输入数学表达式，可包含空格
            PreprocessingTensor:是否预处理图像(若需要对图像进行运算请打开)
                预处理图像会将批次/大小/通道统一，使image和mask可以进行运算
            RGBA_to_RGB：打开时如果输入image为4通道则将其转为3通道
            clamp:限制图像数值为0-1，如果要继续进行下一次image_math_value建议不打开
        说明：
            1：注意表达式结果若不是图像(张量)，则结果将输出到对应的value
                此时image和mask输出将为空，反之value为空
            2：支持部分python方法，请注意输出类型
            3：不需要的输出对应的表达式请留空，可忽略不必要的计算
            4：Please refer to the English explanation -any_math-
    """

    @classmethod
    def INPUT_TYPES(s):
        required = {"expression1":("STRING",{"default":"a+b","multiline": True}),
                    "expression2":("STRING",{"default":"","multiline": True}),
                    "expression3":("STRING",{"default":"","multiline": True}),
                    **s.required}
        required.pop("expression")
        return {
                "required":required,
                "optional":{**s.optional,
                    "e":(any,),"f":(any,),"g":(any,),"h":(any,),"i":(any,),"j":(any,)}
                }

    RETURN_TYPES = ("IMAGE","MASK",any,"IMAGE","MASK",any,"IMAGE","MASK",any)
    RETURN_NAMES = ("image1","mask1","value1","image2","mask2","value2","image3","mask3","value3")
    def image_math(self,expression1,expression2,expression,PreprocessingTensor,image_clamp,RGBA_to_RGB,**kwargs):
        #初始化值
        self.clamp = image_clamp
        if PreprocessingTensor: 
            self.process_input(RGBA_to_RGB,kwargs)
        else: 
            self.tensors = kwargs

        return (*self.handle_img(expression1,1),
                *self.handle_img(expression2,2),
                *self.handle_img(expression,3),
                )

# The following is a test and has not been imported yet 以下为测试，暂未导入
CATEGORY_NAME = "WJNode/temp"

class SaveImage1: #参考
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
    #WJNode/ImageFile
    "Load_Image_From_Path": Load_Image_From_Path,
    "Save_Image_To_Path": Save_Image_To_Path,
    "Save_Image_Out": Save_Image_Out,
    "Load_Image_Adv": Load_Image_Adv,
    "image_url_download": image_url_download,
    #WJNode/ImageCrop
    "adv_crop": adv_crop,
    #WJNode/ImageEdit
    "invert_channel_adv": invert_channel_adv,
    "ListMerger": ListMerger,
    "Bilateral_Filter": Bilateral_Filter,
    "image_math": image_math,
    "image_math_value": image_math_value,
    "Robust_Imager_Merge": Robust_Imager_Merge,
    "image_scale_pixel_v2": image_scale_pixel_v2,
    "image_scale_pixel_option": image_scale_pixel_option,
    #WJNode/ImageMath
    "any_math": any_math,
    "any_math_v2": any_math_v2,
}
