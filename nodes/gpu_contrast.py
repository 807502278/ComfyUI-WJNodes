import torch
import platform
import os
import psutil
import time
import json
from ..moduel.image_utils import device_list
from .hardware import Graphics_detection_RTX4090reference

CATEGORY_NAME = "WJNode/Other-node"

class GPUContrast:
    # 常见GPU规格数据库
    GPU_SPECS = {
        # 消费级NVIDIA GPU
        "RTX 5090": {
           "memory": 48,
           "bandwidth": 1200,
           "fp32_tflops": 120.0,
           "fp16_tflops": 240.0,
           "int8_tops": 960,
           "fp8_tflops": 480.0,
           "mixed_precision_tflops": 220.0,
           "gemm_tflops": 230.0,
           "throughput": 8000,
           "latency": 3.5,
           "mlperf_score": 1.5,
           "aiperf_score": 1.5,
           "cuda_cores": 20480,
           "tensor_cores": 640,
           "rt_cores": 160,
           "architecture": "NVIDIA RTX 50系列",
           "release_date": "2024-10",
           "tdp": 500,
           "msrp": 2500
           },
        "RTX 5080": {
           "memory": 32,
           "bandwidth": 960,
           "fp32_tflops": 85.0,
           "fp16_tflops": 170.0,
           "int8_tops": 680,
           "fp8_tflops": 340.0,
           "mixed_precision_tflops": 160.0,
           "gemm_tflops": 170.0,
           "throughput": 6000,
           "latency": 4.0,
           "mlperf_score": 1.2,
           "aiperf_score": 1.2,
           "cuda_cores": 16384,
           "tensor_cores": 512,
           "rt_cores": 128,
           "architecture": "NVIDIA RTX 50系列",
           "release_date": "2024-11",
           "tdp": 400,
           "msrp": 1800
           },
        "RTX 4090": {
               "memory": 24,  # GB
               "bandwidth": 1008,  # GB/s
               "fp32_tflops": 82.6,
               "fp16_tflops": 165.2,
               "int8_tops": 661,
               "fp8_tflops": 330.4,
               "mixed_precision_tflops": 150.0,
               "gemm_tflops": 160.0,
               "throughput": 5000,  # 样本/秒
               "latency": 5.0,  # 毫秒
               "mlperf_score": 1.0,
               "aiperf_score": 1.0,
               "cuda_cores": 16384,
               "tensor_cores": 512,
               "rt_cores": 128,
               "architecture": "Ada Lovelace",
               "release_date": "2022-10",
               "tdp": 450,  # 瓦特
               "msrp": 1599  # 美元
           },
        "RTX 4080": {
            "memory": 16,
            "bandwidth": 717,
            "fp32_tflops": 48.7,
            "fp16_tflops": 97.4,
            "int8_tops": 390,
            "fp8_tflops": 194.8,
            "mixed_precision_tflops": 90.0,
            "gemm_tflops": 95.0,
            "throughput": 3000,
            "latency": 7.0,
            "mlperf_score": 0.6,
            "aiperf_score": 0.6,
            "cuda_cores": 9728,
            "tensor_cores": 304,
            "rt_cores": 76,
            "architecture": "Ada Lovelace",
            "release_date": "2022-11",
            "tdp": 320,
            "msrp": 1199
        },
        "RTX 4070 Ti": {
            "memory": 12,
            "bandwidth": 504,
            "fp32_tflops": 40.1,
            "fp16_tflops": 80.2,
            "int8_tops": 321,
            "fp8_tflops": 160.4,
            "mixed_precision_tflops": 75.0,
            "gemm_tflops": 78.0,
            "throughput": 2500,
            "latency": 8.0,
            "mlperf_score": 0.5,
            "aiperf_score": 0.5,
            "cuda_cores": 7680,
            "tensor_cores": 240,
            "rt_cores": 60,
            "architecture": "Ada Lovelace",
            "release_date": "2023-01",
            "tdp": 285,
            "msrp": 799
        },
        "RTX 4070": {
            "memory": 12,
            "bandwidth": 504,
            "fp32_tflops": 29.1,
            "fp16_tflops": 58.2,
            "int8_tops": 233,
            "fp8_tflops": 116.4,
            "mixed_precision_tflops": 55.0,
            "gemm_tflops": 57.0,
            "throughput": 2000,
            "latency": 9.0,
            "mlperf_score": 0.4,
            "aiperf_score": 0.4,
            "cuda_cores": 5888,
            "tensor_cores": 184,
            "rt_cores": 46,
            "architecture": "Ada Lovelace",
            "release_date": "2023-04",
            "tdp": 200,
            "msrp": 599
        },
        "RTX 3090 Ti": {
            "memory": 24,
            "bandwidth": 1008,
            "fp32_tflops": 40.0,
            "fp16_tflops": 80.0,
            "int8_tops": 320,
            "fp8_tflops": 0,  # 不支持FP8
            "mixed_precision_tflops": 75.0,
            "gemm_tflops": 78.0,
            "throughput": 2400,
            "latency": 8.5,
            "mlperf_score": 0.48,
            "aiperf_score": 0.48,
            "cuda_cores": 10752,
            "tensor_cores": 336,
            "rt_cores": 84,
            "architecture": "Ampere",
            "release_date": "2022-03",
            "tdp": 450,
            "msrp": 1999
        },
        "RTX 3090": {
            "memory": 24,
            "bandwidth": 936,
            "fp32_tflops": 35.6,
            "fp16_tflops": 71.2,
            "int8_tops": 285,
            "fp8_tflops": 0,
            "mixed_precision_tflops": 67.0,
            "gemm_tflops": 69.0,
            "throughput": 2200,
            "latency": 9.0,
            "mlperf_score": 0.45,
            "aiperf_score": 0.45,
            "cuda_cores": 10496,
            "tensor_cores": 328,
            "rt_cores": 82,
            "architecture": "Ampere",
            "release_date": "2020-09",
            "tdp": 350,
            "msrp": 1499
        },
        "RTX 3080": {
            "memory": 10,
            "bandwidth": 760,
            "fp32_tflops": 29.8,
            "fp16_tflops": 59.6,
            "int8_tops": 238,
            "fp8_tflops": 0,
            "mixed_precision_tflops": 56.0,
            "gemm_tflops": 58.0,
            "throughput": 1800,
            "latency": 10.0,
            "mlperf_score": 0.38,
            "aiperf_score": 0.38,
            "cuda_cores": 8704,
            "tensor_cores": 272,
            "rt_cores": 68,
            "architecture": "Ampere",
            "release_date": "2020-09",
            "tdp": 320,
            "msrp": 699
        },
        
        # 服务器级NVIDIA GPU
        "A100 80GB": {
            "memory": 80,
            "bandwidth": 2039,
            "fp32_tflops": 19.5,
            "fp16_tflops": 312.0,
            "int8_tops": 624,
            "fp8_tflops": 0,
            "mixed_precision_tflops": 156.0,
            "gemm_tflops": 312.0,
            "throughput": 6000,
            "latency": 4.0,
            "mlperf_score": 1.2,
            "aiperf_score": 1.2,
            "cuda_cores": 6912,
            "tensor_cores": 432,
            "rt_cores": 0,
            "architecture": "Ampere",
            "release_date": "2020-11",
            "tdp": 400,
            "msrp": 15000
        },
        "A100 40GB": {
            "memory": 40,
            "bandwidth": 1555,
            "fp32_tflops": 19.5,
            "fp16_tflops": 312.0,
            "int8_tops": 624,
            "fp8_tflops": 0,
            "mixed_precision_tflops": 156.0,
            "gemm_tflops": 312.0,
            "throughput": 6000,
            "latency": 4.0,
            "mlperf_score": 1.2,
            "aiperf_score": 1.2,
            "cuda_cores": 6912,
            "tensor_cores": 432,
            "rt_cores": 0,
            "architecture": "Ampere",
            "release_date": "2020-05",
            "tdp": 400,
            "msrp": 10000
        },
        "H100 80GB": {
            "memory": 80,
            "bandwidth": 3350,
            "fp32_tflops": 51.0,
            "fp16_tflops": 756.0,
            "int8_tops": 1513,
            "fp8_tflops": 3026.0,
            "mixed_precision_tflops": 756.0,
            "gemm_tflops": 756.0,
            "throughput": 12000,
            "latency": 2.5,
            "mlperf_score": 3.0,
            "aiperf_score": 3.0,
            "cuda_cores": 16896,
            "tensor_cores": 528,
            "rt_cores": 0,
            "architecture": "Hopper",
            "release_date": "2022-03",
            "tdp": 700,
            "msrp": 30000
        },
        "L40": {
            "memory": 48,
            "bandwidth": 864,
            "fp32_tflops": 90.5,
            "fp16_tflops": 181.0,
            "int8_tops": 724,
            "fp8_tflops": 362.0,
            "mixed_precision_tflops": 170.0,
            "gemm_tflops": 175.0,
            "throughput": 5500,
            "latency": 4.5,
            "mlperf_score": 1.1,
            "aiperf_score": 1.1,
            "cuda_cores": 18176,
            "tensor_cores": 568,
            "rt_cores": 142,
            "architecture": "Ada Lovelace",
            "release_date": "2022-09",
            "tdp": 300,
            "msrp": 5000
        },
    "AMD Radeon RX 8000 XT": {
        "memory": 32,
        "bandwidth": 1200,
        "fp32_tflops": 75.0,
        "fp16_tflops": 150.0,
        "int8_tops": 600,
        "fp8_tflops": 0,
        "mixed_precision_tflops": 140.0,
        "gemm_tflops": 145.0,
        "throughput": 5500,
        "latency": 5.0,
        "mlperf_score": 1.0,
        "aiperf_score": 1.0,
        "cuda_cores": 14080,
        "tensor_cores": 448,
        "rt_cores": 112,
        "architecture": "AMD RDNA 4",
        "release_date": "2024-09",
        "tdp": 450,
        "msrp": 1200
        },
    "NVIDIA H100 160GB": {
        "memory": 160,
        "bandwidth": 4000,
        "fp32_tflops": 60.0,
        "fp16_tflops": 900.0,
        "int8_tops": 1800,
        "fp8_tflops": 3600.0,
        "mixed_precision_tflops": 900.0,
        "gemm_tflops": 900.0,
        "throughput": 15000,
        "latency": 2.0,
        "mlperf_score": 3.5,
        "aiperf_score": 3.5,
        "cuda_cores": 20480,
        "tensor_cores": 640,
        "rt_cores": 0,
        "architecture": "Hopper",
        "release_date": "2024-06",
        "tdp": 800,
        "msrp": 40000
        }
    }
    
    DESCRIPTION = """
    GPU对比节点，用于:
    - 对比当前GPU与常见GPU型号的性能差异
    - 显示多种GPU型号的详细规格
    - 生成性能对比报告
    - 输出GPU信息用于多节点级联对比
    """
    
    @classmethod
    def INPUT_TYPES(s):
        device_select = list(device_list.keys())
        gpu_models = list(s.GPU_SPECS.keys())
        return {
            "required": {
                "device": (device_select, {"default": device_select[0]}),
                "compare_with": (gpu_models, {"default": "RTX 4090"}),
                "test_items": (["all", "basic", "precision", "memory", "operator", "ai"], {"default": "all"}),
                "matrix_size": ("INT", {"default": 4096, "min": 1024, "max": 16384, "step": 1024}),
                "test_loops": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "language": (["English", "中文"], {"default": "English"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "gpu_info": ("GPU_INFO", ),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING", "GPU_INFO")
    RETURN_NAMES = ("comparison_result", "gpu_info")
    FUNCTION = "compare_gpu"
    OUTPUT_NODE = True
    
    def compare_gpu(self, device, compare_with, test_items, matrix_size, test_loops, language, batch_size, gpu_info=None):
        # 使用Graphics_detection_RTX4090reference获取当前GPU信息
        detector = Graphics_detection_RTX4090reference()
        current_gpu_info, _ = detector.get_info(device, test_items, matrix_size, test_loops, language)
        
        # 获取比较目标GPU的规格
        target_gpu_specs = self.GPU_SPECS.get(compare_with, self.GPU_SPECS["RTX 4090"])
        
        # 解析当前GPU信息
        current_gpu_data = self._parse_gpu_info(current_gpu_info)
        
        # 生成对比结果
        comparison_result = self._generate_comparison(current_gpu_data, target_gpu_specs, compare_with, language)
        
        # 如果提供了外部GPU信息，则添加到对比结果中
        if gpu_info is not None:
            comparison_result += "\n\n" + self._compare_with_external_gpu(current_gpu_data, gpu_info, language)
        
        # 准备返回的GPU信息
        output_gpu_info = {
            "name": current_gpu_data.get("GPU Device", "Unknown GPU"),
            "specs": current_gpu_data,
            "raw_info": current_gpu_info
        }
        
        return (comparison_result, output_gpu_info)
    
    def _parse_gpu_info(self, gpu_info):
        """解析GPU信息字符串，提取关键性能指标"""
        result = {}
        lines = gpu_info.split('\n')
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value_part = parts[1].strip()
                
                # 提取数值
                if '(' in value_part:
                    value = value_part.split('(')[0].strip()
                else:
                    value = value_part
                    
                # 尝试转换为数值
                try:
                    if ' ' in value:
                        numeric_value = float(value.split(' ')[0])
                        result[key] = numeric_value
                    else:
                        result[key] = value
                except ValueError:
                    result[key] = value
        
        return result
    
    def _generate_comparison(self, current_gpu_data, target_gpu_specs, target_name, language):
        """生成当前GPU与目标GPU的对比结果"""
        if language == "中文":
            header = f"===== 当前GPU与{target_name}对比 =====\n"
            comparison_format = "{}: 当前GPU: {} | {}: {} | 性能比: {}%\n"
            summary = "\n===== 性能总结 =====\n"
            overall_format = "总体性能比: {}%\n"
            advantage = "优势: {}\n"
            disadvantage = "劣势: {}\n"
        else:
            header = f"===== Current GPU vs {target_name} Comparison =====\n"
            comparison_format = "{}: Current: {} | {}: {} | Performance Ratio: {}%\n"
            summary = "\n===== Performance Summary =====\n"
            overall_format = "Overall Performance Ratio: {}%\n"
            advantage = "Advantages: {}\n"
            disadvantage = "Disadvantages: {}\n"
        
        result = header
        performance_ratios = []
        advantages = []
        disadvantages = []
        
        # 映射关系
        key_mapping = {
            "Total Memory": "memory",
            "Memory Bandwidth": "bandwidth",
            "FP32 Performance": "fp32_tflops",
            "FP16 Performance": "fp16_tflops",
            "INT8 Performance": "int8_tops",
            "Mixed Precision Performance": "mixed_precision_tflops",
            "GEMM Performance": "gemm_tflops",
            "Throughput": "throughput",
            "Latency": "latency"
        }
        
        # 中文映射
        cn_mapping = {
            "Total Memory": "总显存",
            "Memory Bandwidth": "内存带宽",
            "FP32 Performance": "FP32性能",
            "FP16 Performance": "FP16性能",
            "INT8 Performance": "INT8性能",
            "Mixed Precision Performance": "混合精度性能",
            "GEMM Performance": "GEMM性能",
            "Throughput": "吞吐量",
            "Latency": "时延"
        }
        
        for key, target_key in key_mapping.items():
            if key in current_gpu_data and target_key in target_gpu_specs:
                current_value = current_gpu_data[key]
                target_value = target_gpu_specs[target_key]
                
                # 特殊处理延迟（越低越好）
                if key == "Latency":
                    ratio = (target_value / current_value) * 100 if current_value > 0 else 0
                else:
                    ratio = (current_value / target_value) * 100 if target_value > 0 else 0
                
                performance_ratios.append(ratio)
                
                # 记录优势和劣势
                if ratio > 110:
                    if language == "中文":
                        advantages.append(f"{cn_mapping.get(key, key)}: {ratio:.1f}%")
                    else:
                        advantages.append(f"{key}: {ratio:.1f}%")
                elif ratio < 90:
                    if language == "中文":
                        disadvantages.append(f"{cn_mapping.get(key, key)}: {ratio:.1f}%")
                    else:
                        disadvantages.append(f"{key}: {ratio:.1f}%")
                
                # 添加到结果
                display_key = cn_mapping.get(key, key) if language == "中文" else key
                result += comparison_format.format(
                    display_key, current_value, 
                    target_name, target_value, 
                    f"{ratio:.1f}"
                )
        
        # 添加总结
        result += summary
        overall_ratio = sum(performance_ratios) / len(performance_ratios) if performance_ratios else 0
        result += overall_format.format(f"{overall_ratio:.1f}")
        
        if advantages:
            result += advantage.format(", ".join(advantages))
        if disadvantages:
            result += disadvantage.format(", ".join(disadvantages))
        
        return result
    
    def _compare_with_external_gpu(self, current_gpu_data, external_gpu_info, language):
        """与外部提供的GPU信息进行对比"""
        if language == "中文":
            header = f"===== 与外部GPU对比: {external_gpu_info.get('name', '未知GPU')} =====\n"
        else:
            header = f"===== Comparison with External GPU: {external_gpu_info.get('name', 'Unknown GPU')} =====\n"
        
        # 使用与_generate_comparison相似的逻辑进行对比
        external_specs = external_gpu_info.get('specs', {})
        return header + self._generate_comparison(current_gpu_data, external_specs, external_gpu_info.get('name', 'External GPU'), language)

NODE_CLASS_MAPPINGS = {
    #"GPUContrast": GPUContrast,
}