import glob
import os

from typing import List, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

import comfy.model_management as mm
import folder_paths

from .models_BiRefNet.models.birefnet import BiRefNet



folder_paths.add_model_folder_path("BiRefNet",os.path.join(folder_paths.models_dir, "BiRefNet"))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision(["high", "highest"][0])

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Tensor to np
def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
  if len(tensor.shape) == 3:  # Single image
    return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
  else:  # Batch of images
    return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]


CATEGORY_NAME = "WJNode/tensort"



class building_engine:
    @classmethod
    def INPUT_TYPES(cls):
        model_class = ["BiRefNet2","Depth","Pose","FaceRestore","Rife","Upscaler","custom"]

        list_BiRefNet2 = ["BiRefNet-v2-onnx/BiRefNet_lite-general-2K-epoch_232.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-COD-epoch_125.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-DIS-epoch_590.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-general-epoch_244.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-HRSOD_DHU-epoch_115.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-matting-epoch_100.onnx",
                      "BiRefNet-v2-onnx/BiRefNet-portrait-epoch_150.onnx",]
        
        list_Depth = ["Depth-Anything-2-Onnx/depth_anything_v2_vitb.onnx",
                      "Depth-Anything-2-Onnx/depth_anything_v2_vitl.onnx",
                      "Depth-Anything-2-Onnx/depth_anything_v2_vits.onnx",
                      "depth-pro-onnx/depth_pro.onnx",]
        
        list_Pose = ["dwpose-onnx/yolox_l_dynamic_batch_opset_17_sim.onnx",
                      "yolo-nas-pose-onnx/yolo_nas_pose_l_0.1.onnx",
                      "yolo-nas-pose-onnx/yolo_nas_pose_l_0.2.onnx",
                      "yolo-nas-pose-onnx/yolo_nas_pose_l_0.5.onnx",
                      "yolo-nas-pose-onnx/yolo_nas_pose_l_0.8.onnx",
                      "yolo-nas-pose-onnx/yolo_nas_pose_l_0.35.onnx",]
        
        list_FaceRestore = ["facerestore-onnx/codeformer.onnx","facerestore-onnx/gfqgan.onnx",]

        list_Rife = ["rife-onnx/rife47_ensemble_True_scale_1_sim.onnx",
                      "rife-onnx/rife48_ensemble_True_scale_1_sim.onnx",
                      "rife-onnx/rife49_ensemble_True_scale_1_sim.onnx",]
        
        list_Upscaler = ["Upscaler-Onnx/4x_foolhardy_Remacri.onnx",
                      "Upscaler-Onnx/4x_NMKD-Siax_200k.onnx",
                      "Upscaler-Onnx/4x_RealisticRescaler_100000_G.onnx",
                      "Upscaler-Onnx/4x-AnimeSharp.onnx",
                      "Upscaler-Onnx/4x-UltraSharp.onnx",
                      "Upscaler-Onnx/4x-WTP-UDS-Esrgan.onnx",
                      "Upscaler-Onnx/RealESRGAN_x4.onnx",]
        
        list_custom = []

        glob.glob(folder_paths.models_dir)
        
        local_models= folder_paths.get_filename_list("BiRefNet"),
        if isinstance(local_models,tuple):
            local_models = list(local_models[0])
        local_models.append(os.listdir(os.path.join(folder_paths.models_dir,"tensorrt/BiRefNet")))



        return {
            "required": {
                "model_class":(model_class,{"default": local_models[0],}),
                "onnx_model": (list_Depth,{"default": local_models[0],}),
                "force_building": ("BOOLEAN",{"default": False}),
            }
        }

    RETURN_TYPES = ("BRNMODEL",)
    RETURN_NAMES = ("birefnet",)
    FUNCTION = "building"
    CATEGORY = CATEGORY_NAME
    def building():
        ...


class load_BiRefNet2_General:
    def model_name(self):
        self.pretrained_weights = [
        'zhengpeng7/BiRefNet',
        'zhengpeng7/BiRefNet-portrait',
        'zhengpeng7/BiRefNet-legacy', 
        'zhengpeng7/BiRefNet-DIS5K-TR_TEs', 
        'zhengpeng7/BiRefNet-DIS5K',
        'zhengpeng7/BiRefNet-HRSOD',
        'zhengpeng7/BiRefNet-COD',
        'zhengpeng7/BiRefNet_lite',     # Modify the `bb` in `config.py` to `swin_v1_tiny`.
        ]
        # https://objects.githubusercontent.com/github-production-release-asset-2e65be/525717745/81693dcf-8d42-4ef6-8dba-1f18f87de174?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241014%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241014T003944Z&X-Amz-Expires=300&X-Amz-Signature=ec867061341cf6498cf5740c36f49da22d4d3d541da48d6e82c7bce0f3b63eaf&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3DBiRefNet-COD-epoch_125.pth&response-content-type=application%2Foctet-stream

    @classmethod
    def INPUT_TYPES(cls):
        local_models= folder_paths.get_filename_list("BiRefNet"),
        if isinstance(local_models,tuple):
            local_models = list(local_models[0])
        local_models.append(os.listdir(os.path.join(folder_paths.models_dir,"tensorrt/BiRefNet")))
        return {
            "required": {
                "birefnet_model": (local_models,{"default": local_models[0],}),
            }
        }

    RETURN_TYPES = ("BRNMODEL",)
    RETURN_NAMES = ("birefnet",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY_NAME
  
    def load_model(self,birefnet_model):
        model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
        if not os.path.isfile(model_path): 
            model_path = os.path.join(folder_paths.models_dir,"tensorrt/BiRefNet",birefnet_model)
        print(f"load model: {model_path}")

        if birefnet_model.endswith('.onnx'):
                import onnxruntime
                providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
                #model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
                onnx_session = onnxruntime.InferenceSession(
                    model_path,
                    providers=providers
                )
                return (('onnx',onnx_session),),
        elif birefnet_model.endswith('.engine') or birefnet_model.endswith('.trt') or birefnet_model.endswith('.plan'):
            #model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
            import tensorrt as trt
            # 创建logger：日志记录器
            logger = trt.Logger(trt.Logger.WARNING)
            # 创建runtime并反序列化生成engine
            with open(model_path ,'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            return (('tensorrt',engine),)
        else :
            raise TypeError("Only supports  .onnx  .engine  .trt  .plan")

class BiRefNet2_tensort:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "birefnet": ("BRNMODEL",),
                "image": ("IMAGE",),
                "reversal_mask": ("BOOLEAN",{"default":False})
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "remove_background"
    CATEGORY = CATEGORY_NAME
  
    def remove_background(self, birefnet, image,reversal_mask):
        net_type, net = birefnet
        processed_masks = []

        transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = self.resize_image(orig_image)
            im_tensor = transform_image(image).unsqueeze(0)
            im_tensor=im_tensor.to(device)
            if net_type=='onnx':
                input_name = net.get_inputs()[0].name
                input_images_numpy = tensor2np(im_tensor)
                result = torch.tensor(
                    net.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
                ).squeeze(0).sigmoid().cpu()
            
            elif net_type=='tensorrt':
                from .models_BiRefNet import common
                with net.create_execution_context() as context:
                    image_data = np.expand_dims(transform_image(orig_image), axis=0).ravel()
                    engine = net
                    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
                    np.copyto(inputs[0].host, image_data)
                    trt_outputs = common.do_inference(context, engine, bindings, inputs, outputs, stream)
                   
                    numpy_array = np.array(trt_outputs[-1].reshape((1, 1, 1024, 1024)))
                    result = torch.from_numpy(numpy_array).sigmoid().cpu()
                    common.free_buffers(inputs, outputs, stream)
            else:
                with torch.no_grad():
                    result = net(im_tensor)[-1].sigmoid().cpu()
                    
                    
            result = torch.squeeze(F.interpolate(result, size=(h,w)))
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            result = torch.cat(result, dim=0)
            processed_masks.append(result)

        new_masks = torch.cat(processed_masks, dim=0)
        if reversal_mask : new_masks = 1 - new_masks
        return (new_masks,)
    
    def resize_image(self,image):
        image = image.convert('RGB')
        model_input_size = (1024, 1024)
        image = image.resize(model_input_size, Image.BILINEAR)
        return image
    

NODE_CLASS_MAPPINGS = {
    #"load_BiRefNet2_General": load_BiRefNet2_General,
    #'BiRefNet2_tensort':BiRefNet2_tensort
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    #"load_BiRefNet2_General": "load BiRefNet2 General",
    #"BiRefNet2_tensort": "BiRefNet2 tensort",
}
