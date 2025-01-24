# https://github.com/yuvraj108c
# https://huggingface.co/yuvraj108c

import torch
import time
import os

from .utilities import Engine
from .utilities_Rifet import Engine_Rife
import folder_paths

def inspect_path(*path:str,prefix:str="",mk=True): #2025-0120
    """
    功能：组合路径，如果路径不存在则新建\n
    输入：多个字符串， prefix-文件名加前缀(默认不加), mk-是否新建路径\n
    输出：组合路径，路径，文件名，是否已新建
    """
    out_dir = os.path.join(*path)
    out_path, out_file = os.path.split(out_dir)
    out = not os.path.isdir(out_path)
    if mk :
        if out: 
            os.mkdir(out_path)
    else:
        out = False
    if prefix != "":
        out_file = prefix + "_" + out_file
    return (out_dir, out_path, out_file, out)

def re_file_name (dir:str,new_name:str="",extensions:str=""): #2025-0120
    """
    功能：改路径字符串的文件名\n
    输入：字符串，新名称，扩展名(会覆盖新名称的扩展名)\n
    输出：新路径字符串
    """
    f_n,e_n = os.path.splitext(new_name)
    path,file = os.path.split(dir)
    f_o,e_o = os.path.splitext(file)
    if extensions != "":
        if extensions[0] != ".":
            extensions = "." + extensions

    if new_name == "":
        if extensions == "":
            return dir
        else:
            return os.path.join(path,f_o) + extensions
    else:
        if extensions != "":
            return os.path.join(path,f_n) + extensions
        else:
            if e_n == "" and e_o != "":
                return os.path.join(path,f_n) + e_o
            else:
                return os.path.join(path,new_name)

# -------------------------------------------------------------------------------
# 单通道转换函数：支持以下多个
# 姿态1
# https://github.com/yuvraj108c/ComfyUI-Dwpose-Tensorrt
# https://huggingface.co/yzd-v/DWPose/tree/main
# 姿态2
# https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt
# https://huggingface.co/yuvraj108c/yolo-nas-pose-onnx/tree/main

# 深度1
# https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt
# https://huggingface.co/yuvraj108c/Depth-Anything-Onnx
# https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx
# 深度2 非cf插件
# https://github.com/yuvraj108c/ml-depth-pro-tensorrt
# https://huggingface.co/yuvraj108c/ml-depth-pro-onnx/blob/main/depth_pro.onnx

# 人脸高清修复
# https://github.com/yuvraj108c/ComfyUI-Facerestore-Tensorrt
# https://huggingface.co/yuvraj108c/facerestore-onnx
def trt_depth_anything(onnx_path = None, use_fp16 = True):
    _, file_name = os.path.split(onnx_path)
    trt_path,_ = inspect_path(folder_paths.models_dir,"tensorrt/depth-anything",file_name,prefix="trt")

    ret = None
    if os.path.exists(trt_path):
        print(f"Engine file already exists, skip build !")
    else:
        engine = Engine(trt_path)
        torch.cuda.empty_cache()

        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
    return ret


# 放大-3通道转换
# https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt
# https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/tree/main

def trt_upscaler(onnx_path=None, use_fp16=True):
    _, file_name = os.path.split(onnx_path)
    trt_path,_ = inspect_path(folder_paths.models_dir,"tensorrt/upscaler",file_name,prefix="trt")

    ret = None
    if os.path.exists(trt_path):
        print(f"Engine file already exists, skip build !")
    else:
        engine = Engine(trt_path)
        torch.cuda.empty_cache()

        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
            input_profile=[
                {"input": [(1,3,256,256), (1,3,512,512), (1,3,1280,1280)]}, # any sizes from 256x256 to 1280x1280
            ],
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
    return ret


# 插帧-3通道转换+多分辨率
# https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt
# https://huggingface.co/yuvraj108c/rife-onnx
def trt_rife(onnx_path=None, use_fp16=True):
    _, file_name = os.path.split(onnx_path)
    trt_path,_ = inspect_path(folder_paths.models_dir,"tensorrt/rife",file_name,prefix="trt")

    engine = Engine_Rife(trt_path)
    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
        input_profile=[
            # any sizes from 256x256 to 3840x3840, batch size 1
            {
                "img0": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 3840, 3840)],
                "img1": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 3840, 3840)],
            },
        ],
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")
    return ret

# -------------------------------------------------------------------------------

CATEGORY_NAME = "WJNode/TensorRT"

class to_onnx:
    """
    pt转onnx
    """
    def __init__(self):
        self.pt_model = []
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "select_model": (s.pt_model, {"default": ""})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = None
    FUNCTION = "to_onnx"

    def to_onnx(self, select_model):
        ...
    
    # 转onnx - YoloNasPose #加载路径待优化
    def to_onnx_yolo1(self, pt_dir):
        try:
            from super_gradients.training import models
            from super_gradients.common.object_names import Models
        except:
            print("please pip install super_gradients !")
        _, file_name = os.path.split(pt_dir)
        onnx_path,_ = inspect_path(folder_paths.models_dir,"tensorrt/onnx",file_name)
        onnx_path = re_file_name(onnx_path,extensions = "onnx")

        model = models.get(Models.YOLO_NAS_POSE_L, pretrained_weights="coco_pose")
        export_result = model.export(onnx_path, confidence_threshold=0.5)
        print(export_result)

    # 转onnx - depth_pro #加载路径待优化
    def to_onnx_depth_pro(self, pt_dir):
        try:
            import depth_pro
        except:
            print("please pip install depth_pro !")
        DEVICE = "cuda"
        model, transform = depth_pro.create_model_and_transforms(device=DEVICE)
        model.eval()

        _, file_name = os.path.split(pt_dir)
        onnx_path,_ = inspect_path(folder_paths.models_dir,"tensorrt/onnx",file_name)
        onnx_path = re_file_name(onnx_path,extensions = "onnx")

        with torch.no_grad():
            torch.onnx.export(model,
                              torch.randn(1, 3, 1536, 1536).to(DEVICE),
                              onnx_path,
                              opset_version=19,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['depth', "focallength_px"],
                              )
            print(f"ONNX model exported to: {onnx_path}")

    # 转onnx - BiRefNet #开发中
    def to_onnx_BiRefNet(self,pt_dir):
        try:
            import depth_pro
        except:
            print("please pip install depth_pro !")
        DEVICE = "cuda"
        model, transform = depth_pro.create_model_and_transforms(device=DEVICE)
        model.eval()

        _, file_name = os.path.split(pt_dir)
        onnx_path,_ = inspect_path(folder_paths.models_dir,"tensorrt/onnx",file_name)
        onnx_path = re_file_name(onnx_path,extensions = "onnx")

        with torch.no_grad():
            torch.onnx.export(model,
                              torch.randn(1, 3, 1536, 1536).to(DEVICE),
                              onnx_path,
                              opset_version=19,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['depth', "focallength_px"],
                              )
            print(f"ONNX model exported to: {onnx_path}")

class to_trt_Dwpose:
    """
    onnx转TensorRT引擎
    """
    pt_model = ["All","dw-ll_ucoco_384.onnx","yolox_l.onnx"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "select_model": (s.pt_model, {"default": ""}),
                "rebuild": ("BOOLEAN",{"default": False})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = None
    FUNCTION = "to_onnx"

    def to_onnx(self, select_model):
        if select_model == to_trt_Dwpose.pt_model[0]:
            pass

class to_trt_YoloNasPose:
    """
    onnx转TensorRT引擎
    """
    pt_model = None
    def __init__(s):
        s.pt_model = ["yolo_nas_pose_l_0.1.onnx",
                    "yolo_nas_pose_l_0.2.onnx",
                    "yolo_nas_pose_l_0.35.onnx",
                    "yolo_nas_pose_l_0.5.onnx",
                    "yolo_nas_pose_l_0.8.onnx",]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "select_model": (to_trt_YoloNasPose.pt_model, {"default": ""})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = None
    FUNCTION = "to_onnx"

    def to_onnx(self, select_model):
        ...



