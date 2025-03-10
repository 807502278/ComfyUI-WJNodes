import torch
import platform
import os
import psutil
import time
import json
from ..moduel.image_utils import device_list

refit = {
    "RTX 5090":{127:"疯狗版",63:"至尊版"},
    "RTX 4090":{79:"疯狗版",47:"至尊版"},
    "RTX 4080":{15:"豪华版"},
    "RTX 4060 Ti":{15:"豪华版"},
    "RTX 3080":{43:"超级疯狗版",21:"疯狗版",19:"至尊版",11:"豪华版"},
    "RTX 3070":{15:"豪华版"},
    "RTX 2080":{43:"超级疯狗版",21:"疯狗版",19:"至尊版",11:"豪华版"},
    "A100":{79:"豪华版"},
    "RX 580":{15:"至尊版"},
}

CATEGORY_NAME = "WJNode/Other-node"

class Graphics_Detection_Reference:
    RTX_4090 = {
        "memory": 24,  # 总显存GB rtx4060-8.00 GB
        "bandwidth": 1008,  # 显存带宽GB/s rtx4060-217.09 GB/s
        "fp32_tflops": 82.58, # rtx4060-14.84 TFLOPS
        "fp16_tflops": 165.16, #rtx4060-30.48 TFLOPS
        "bf16_tflops": 165.16,
        "int8_tops": 661, # rtx4060-不支持
        "fp8_tflops": 330.4,  # 估计值，基于FP16的2倍 
        "NVFP4": None,
        "mixed_precision_tflops": 43.3,  # 混合精度估计值 rtx4060-7.38 TFLOPS
        "gemm_tflops": 81.06,  # GEMM矩阵乘法估计值 rtx4060-27.47 TFLOPS
        "throughput": 6418.5,  # 吞吐量基准 (样本/秒) rtx4060-1163.66 samples/sec
        "latency": 0.52,  # 时延基准 (毫秒) rtx4060-0.88 ms
        "mlperf_score": 34.67,  # MLPerf相对分数 rtx4060-6.77
        "aiperf_score": 62.84  # AIPerf相对分数 rtx4060-13.40
    }

    DESCRIPTION = """
    用于简单检测显卡各项运算能力，包括:
    - 硬件信息
    - 各数据类型，显存和带宽，算子性能测试
    - 与RTX 4090性能对比(部分4090参数未经实测)
    - AI性能测试(混合精度、GEMM、深度学习基准等)
    输入：
        test_items/all:运行所有测试
        test_items/basic:基础信息检测
        test_items/precision:混合精度性能测试
        test_items/memory:显存带宽测试
        test_items/operator:算子性能测试
        test_items/ai:AI性能测试
        test_batch:测试批次,测试小批量(例如3)时数据会不稳定但速度快
        language:语言选择，英文或中文
    参数说明：
        **Max Threads Per Block：**每块最大线程数，表示一个CUDA线程块中可以包含的最大线程数。
        **Multi-Processor Count：**多处理器数量，表示GPU中CUDA多处理器（SM）的数量。
        **Memory Bandwidth：**显存带宽，表示GPU显存与计算核心之间的数据传输速率。
        **Memory Bandwidth Test：**显存带宽测试，通过特定测试程序测量GPU显存的实际带宽。
        **Tensor Core Performance：**Tensor Core性能，表示Tensor Core在深度学习任务中的计算能力。
        **Tensor Core Test：**Tensor Core测试，通过特定任务评估Tensor Core的实际性能。
        **Mixed Precision Performance：**混合精度性能，表示GPU在混合精度计算（如FP16和FP32）中的表现。
        **GEMM Performance：**GEMM矩阵乘法性能，表示GPU在矩阵乘法任务中的计算能力。
        **DLSS Performance：**DLSS性能，表示GPU在深度学习超采样（DLSS）任务中的表现。
        **Throughput：**吞吐量，表示单位时间内处理的数据量或任务数量。
        **Latency：**时延，表示从输入到输出的时间延迟。
        **MLPerf Score：**MLPerf分数，表示GPU在MLPerf基准测试中的性能表现。
        **AIPerf Score：**AIPerf分数，表示GPU在AIPerf基准测试中的性能表现。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        device_select = list(device_list.keys())[1:]
        
        # 读取硬件数据文件夹中的所有JSON文件
        hardware_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "hardware_data")
        reference_gpu_options = []
        
        # 遍历文件夹中的所有JSON文件
        if os.path.exists(hardware_data_path):
            for file_name in os.listdir(hardware_data_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(hardware_data_path, file_name)
                    prefix = file_name.replace(".json", "")
                    
                    try:
                        # 读取JSON文件内容
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        # 如果文件不为空，提取所有键作为GPU型号
                        if data:
                            for gpu_model in data.keys():
                                reference_gpu_options.append(f"{prefix}/{gpu_model}")
                        else:
                            # 如果文件为空，添加一个默认选项
                            reference_gpu_options.append(f"{prefix}/**To be added**")
                    except Exception as e:
                        # 如果读取失败，添加一个默认选项
                        reference_gpu_options.append(f"{prefix}/**To be added**")
        
        # 如果没有找到任何选项，添加默认选项
        if not reference_gpu_options:
            reference_gpu_options = ["Nvidia/RTX_4090"]
        
        return {
            "required": {
                "device": (device_select, {"default": device_select[-1]}),
                "reference_gpu": (reference_gpu_options, {"default": reference_gpu_options[0]}),
                "test_items": (["all",  # 所有测试
                                "basic",  # 基础信息检测
                                "precision", #混合精度性能测试
                                "memory", # 显存带宽测试
                                "operator", # 算子性能测试
                                "ai" # AI性能测试
                                ], {"default": "all"}),
                "test_batch": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "language": (["English", "中文"], {"default": "English"}),
            },
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    OUTPUT_NODE = True

    def get_info(self, device, reference_gpu, test_items, test_batch, language):
        # 初始化
        device_str = device
        device = torch.device(device)
        info = []
        test_loops = test_batch
        batch_size = 16
        matrix_size = 4096
        
        # 加载参考GPU数据
        reference_data = self.RTX_4090  # 默认使用RTX 4090作为备用
        try:
            if "/" in reference_gpu:
                file_prefix, gpu_model = reference_gpu.split("/")
                if gpu_model != "**To be added**":
                    hardware_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "hardware_data")
                    file_path = os.path.join(hardware_data_path, f"{file_prefix}.json")
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    if gpu_model in data:
                        reference_data = data[gpu_model]
                        info.append(f"Refer to: {gpu_model} as reference")
        except Exception as e:
            info.append(f"Failed to load reference GPU data: {str(e)}. Using default RTX 4090 data.")

        # 若检测cpu则减少参数量
        if device_str == "cpu":
            print("注意：选择了cpu测试，将仅进行基础测试")
            print("be careful:CPU testing has been selected,Only basic testing will be conducted")
            batch_size = 2
            matrix_size = 256

        # 中英文标题映射
        title_map = {
            "Current GPU Device": "当前GPU设备",
            "Architecture": "架构",
            "CUDA Version": "CUDA版本",
            "Total Memory": "总显存",
            "Max Threads Per Block": "每块最大线程数",
            "Multi-Processor Count": "多处理器数量",
            "CPU Device": "CPU设备",
            "CPU Cores": "CPU核心数",
            "CPU Architecture": "CPU架构",
            "System Memory": "系统内存",
            "MPS Device": "MPS设备",
            "Memory Bandwidth": "显存带宽",
            "Memory Bandwidth Test": "显存带宽测试",
            "Tensor Core Performance": "Tensor Core性能",
            "Tensor Core Test": "Tensor Core测试",
            "Mixed Precision Performance": "混合精度性能",
            "GEMM Performance": "GEMM矩阵乘法性能",
            "DLSS Performance": "DLSS性能",
            "Throughput": "吞吐量",
            "Latency": "时延",
            "MLPerf Score": "MLPerf分数",
            "AIPerf Score": "AIPerf分数"
        }
        
        # 语言处理辅助函数
        def get_text(cn_text, en_text):
            return cn_text if language == "中文" else en_text
        
        def add_info(title, content, reference_value=None):
            # 根据语言选择标题
            display_title = title_map.get(title, title) if language == "中文" else title
            
            
            if reference_value is None: #若不是需要对比的数值
                if title == "Latency": # 时延（latency）不计算百分比
                    info.append(f"{display_title}: {content}")
                else: #其它数据直接打印
                    info.append(f"{display_title}: {content}")
            else: #计算百分比
                percentage = (float(content.split()[0]) / reference_value) * 100
                vs_text = get_text("相当于", "Equivalent to")
                ref_gpu_name = reference_gpu.split("/")[1]
                if ref_gpu_name == "**To be added**": ref_gpu_name = "RTX_4090"
                info.append(f"{display_title}: {content} ({vs_text} {ref_gpu_name}: {percentage:.1f}%)")

        
        # 基础信息检测
        if test_items in ["all", "basic"]:
            if device.type == "cuda":
                gpu_name = torch.cuda.get_device_name() #设备名称
                total_memory = torch.cuda.get_device_properties(device).total_memory/1024**3 #总显存大小
                free_memory = torch.cuda.memory_allocated()/1024**3 #当前可用显存
                compute_capability = torch.cuda.get_device_capability()

                # 检测是否是显存改装版
                is_refit = False
                for k,v in refit.items():
                    if not is_refit: break
                    if k in gpu_name :
                        for kk,vv in v.items():
                            if total_memory > kk:
                                is_refit = True
                                add_info("GPU Device", gpu_name+" "+vv)
                                break
                if not is_refit:
                    add_info("GPU Device", gpu_name)

                # 架构和cuda版本
                add_info("Architecture", f"Compute {compute_capability[0]}.{compute_capability[1]}")
                add_info("CUDA Version", torch.version.cuda)
                # 获取更多GPU信息
                gpu_props = torch.cuda.get_device_properties(device)
                add_info("Max Threads Per Block", str(gpu_props.max_threads_per_multi_processor))
                add_info("Multi-Processor Count", str(gpu_props.multi_processor_count))
                # 显存对比
                if device.type != "cpu":
                    add_info("Total Memory", f"{total_memory:.2f} GB", reference_data.get("memory"))

            elif device.type == "cpu":
                add_info("CPU Device", platform.processor())
                add_info("CPU Cores", str(os.cpu_count()))
                add_info("CPU Architecture", platform.machine())
                add_info("System Memory", f"{psutil.virtual_memory().total/1024**3:.1f} GB")
            elif device.type == "mps":
                add_info("MPS Device", "Apple Silicon")
        
        # 精度支持和性能检测
        if test_items in ["all", "precision"]:
            size = matrix_size  # 用于性能测试的矩阵大小
            dtypes = [
                ("FP32", torch.float32),
                ("FP16", torch.float16),
                ("BF16", torch.bfloat16),
                ("FP8", "fp8"),  # 新增FP8测试
                ("INT8", torch.int8)
            ]
            
            for dtype_name, dtype in dtypes:
                try:
                    # 创建测试张量
                    x = torch.randn(size, size, device=device)
                    if dtype in [torch.int8, torch.quint4x2]:
                        x = torch.randint(-128, 127, (size, size), device=device)
                    elif dtype == "fp8":
                        try:
                            x = torch.float8_e4m3fn(x)
                            y = x.clone()
                            
                            torch.cuda.synchronize()
                            tflops_list = []
                            for loop in range(test_loops):
                                start_time = time.time()
                                for _ in range(10):
                                    z = torch.matmul(x, y)
                                    torch.cuda.synchronize()
                                end_time = time.time()
                                
                                ops = (2 * size**3 * 10) / (end_time - start_time)
                                tflops_list.append(ops / (1e12))
                            tflops = sum(tflops_list) / len(tflops_list)
                            tflops = ops / (1e12)
                            perf_title = f"{dtype_name} {get_text('性能', 'Performance')}"
                            add_info(perf_title, f"{tflops:.2f} TFLOPS", reference_data.get("fp8_tflops"))
                            continue
                        except (ImportError, AttributeError):
                            support_text = get_text("否 (不支持FP8)", "No (FP8 not supported)")
                            add_info(f"{dtype_name} {get_text('支持', 'Support')}", support_text)
                            continue

                    if dtype != torch.float32:
                        x = x.to(dtype)
                    y = x.clone()
                    # 性能测试
                    torch.cuda.synchronize()
                    tflops_list = []
                    for loop in range(test_loops):
                        start_time = time.time()
                        for _ in range(10):
                            z = x @ y
                            torch.cuda.synchronize()
                        end_time = time.time()
                        # 计算TFLOPS/TOPS
                        ops = (2 * size**3 * 10) / (end_time - start_time)
                        tflops_list.append(ops / (1e12))
                    tflops = sum(tflops_list) / len(tflops_list)
                    # 获取对比基准
                    if dtype_name == "FP32":
                        baseline = reference_data.get("fp32_tflops")
                    elif dtype_name == "FP16":
                        baseline = reference_data.get("fp16_tflops")
                    elif dtype_name == "BF16":
                        baseline = reference_data.get("bf16_tflops")
                    elif dtype_name == "INT8":
                        baseline = reference_data.get("int8_tops")
                    else:
                        baseline = None
                    perf_title = f"{dtype_name} {get_text('性能', 'Performance')}"
                    add_info(perf_title, f"{tflops:.2f} TFLOPS", baseline)
                except Exception as e:
                    support_text = get_text("否", "No")
                    add_info(f"{dtype_name} {get_text('支持', 'Support')}", support_text)
        # 显存带宽测试
        if test_items in ["all", "memory"]:
            try:
                size = matrix_size
                x = torch.randn(size, size, device=device)
                
                torch.cuda.synchronize()
                bandwidth_list = []
                for loop in range(test_loops):
                    start_time = time.time()
                    for _ in range(10):
                        y = x.clone()
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    bandwidth_list.append((size * size * 4 * 20) / (end_time - start_time) / (1024**3))  # GB/s
                bandwidth = sum(bandwidth_list) / len(bandwidth_list)
                bandwidth_title = get_text("显存带宽", "Memory Bandwidth")
                add_info(bandwidth_title, f"{bandwidth:.2f} GB/s", reference_data.get("bandwidth"))
            except Exception as e:
                test_title = get_text("显存带宽测试", "Memory Bandwidth Test")
                fail_text = get_text(f"失败: {str(e)}", f"Failed: {str(e)}")
                add_info(test_title, fail_text)
        
        # Tensor Core 特殊测试
        if test_items in ["all", "operator"] and device.type == "cuda":
            try:
                size = matrix_size
                x = torch.randn(size, size, device=device, dtype=torch.float16)
                y = torch.randn(size, size, device=device, dtype=torch.float16)
                
                torch.cuda.synchronize()
                tc_tflops_list = []
                for loop in range(test_loops):
                    start_time = time.time()
                    for _ in range(10):
                        z = torch.matmul(x, y)
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    tc_flops = (2 * size**3 * 10) / (end_time - start_time)
                    tc_tflops_list.append(tc_flops / (1e12))
                tc_tflops = sum(tc_tflops_list) / len(tc_tflops_list)
                tc_title = get_text("Tensor Core性能", "Tensor Core Performance")
                add_info(tc_title, f"{tc_tflops:.2f} TFLOPS", reference_data.get("fp16_tflops"))
            except Exception as e:
                test_title = get_text("Tensor Core测试", "Tensor Core Test")
                fail_text = get_text(f"失败: {str(e)}", f"Failed: {str(e)}")
                add_info(test_title, fail_text)
        

        # AI性能测试
        if test_items in ["all", "ai"] and device_str != "cpu":
            # 添加分隔线
            ai_header = get_text("AI性能测试", "AI Performance Tests")
            info.append(f"\n--- {ai_header} ---")
            
            # 混合精度测试
            try:
                size = matrix_size // 2  # 减小矩阵大小以适应混合精度测试
                batch = batch_size
                # 创建FP32和FP16混合的测试 - 确保维度匹配，第一个矩阵乘法: (batch, size, size) @ (batch, size, size) -> (batch, size, size)
                x_fp32 = torch.randn(batch, size, size, device=device, dtype=torch.float32)
                x_fp16 = torch.randn(batch, size, size, device=device, dtype=torch.float16)
                
                torch.cuda.synchronize()
                mixed_tflops_list = []
                for loop in range(test_loops):
                    start_time = time.time()
                    for _ in range(5):
                        # 混合精度计算
                        x_mid = torch.matmul(x_fp32, x_fp16.to(torch.float32))
                        result = torch.matmul(x_mid, x_fp32.transpose(1, 2))
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # 计算TFLOPS (考虑两次矩阵乘法)
                    ops = batch * (2 * size * (size//2) * size + 2 * size * size * (size//2)) * 5
                    ops = ops / (end_time - start_time)
                    mixed_tflops_list.append(ops / (1e12))
                
                mixed_tflops = sum(mixed_tflops_list) / len(mixed_tflops_list)
                add_info("Mixed Precision Performance", f"{mixed_tflops:.2f} TFLOPS", reference_data.get("mixed_precision_tflops"))
            except Exception as e:
                fail_text = get_text(f"失败: {str(e)}", f"Failed: {str(e)}")
                add_info("Mixed Precision Performance", fail_text)
            
            # GEMM矩阵乘法测试，使用不同大小的矩阵进行GEMM测试
            try:
                gemm_sizes = [int(matrix_size/4), int(matrix_size/2), matrix_size]
                gemm_results = []
                
                for gemm_size in gemm_sizes:
                    a = torch.randn(gemm_size, gemm_size, device=device, dtype=torch.float16)
                    b = torch.randn(gemm_size, gemm_size, device=device, dtype=torch.float16)
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(10):
                        c = torch.matmul(a, b)
                        torch.cuda.synchronize()
                    end_time = time.time()
                    # 计算GEMM性能 (TFLOPS)
                    ops = (2 * gemm_size**3 * 10) / (end_time - start_time)
                    gemm_tflops = ops / (1e12)
                    gemm_results.append(gemm_tflops)
                
                avg_gemm_tflops = sum(gemm_results) / len(gemm_results)
                add_info("GEMM Performance", f"{avg_gemm_tflops:.2f} TFLOPS", reference_data.get("gemm_tflops"))
            except Exception as e:
                fail_text = get_text(f"失败: {str(e)}", f"Failed: {str(e)}")
                add_info("GEMM Performance", fail_text)
            
            # 吞吐量测试
            try:
                batch = batch_size
                input_size = int(matrix_size/20)
                inputs = torch.randn(batch, 3, input_size, input_size, device=device)
                # 创建一个简单的卷积网络模拟推理
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(256, 1000)
                ).to(device)
                # 预热
                for _ in range(5):
                    _ = model(inputs)
                torch.cuda.synchronize()
                # 测量吞吐量
                iterations = 20
                start_time = time.time()
                for _ in range(iterations):
                    _ = model(inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                # 计算吞吐量 (样本/秒)
                throughput = (batch * iterations) / (end_time - start_time)
                add_info("Throughput", f"{throughput:.2f} samples/sec", reference_data.get("throughput"))
                # 测量时延
                latency_iterations = 50
                latency_times = []
                for _ in range(latency_iterations):
                    start_time = time.time()
                    _ = model(inputs[:1])  # 单样本推理
                    torch.cuda.synchronize()
                    end_time = time.time()
                    latency_times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                avg_latency = sum(latency_times) / len(latency_times)
                add_info("Latency", f"{avg_latency:.2f} ms", reference_data.get("latency"))
            except Exception as e:
                fail_text = get_text(f"失败: {str(e)}", f"Failed: {str(e)}")
                add_info("Throughput", fail_text)
                add_info("Latency", fail_text)
            
            # 使用简化的模型来模拟MLPerf测试，实际上需要特定的基准测试套件和数据集
            try:
                mlperf_model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(256, 1000),
                    torch.nn.Softmax(dim=1)
                ).to(device)
                
                # 模拟MLPerf测试
                mlperf_batch = 8
                mlperf_input = torch.randn(mlperf_batch, 3, 224, 224, device=device)
                
                # 预热
                for _ in range(5):
                    _ = mlperf_model(mlperf_input)
                torch.cuda.synchronize()
                
                # 测量性能
                mlperf_iterations = 10
                mlperf_start = time.time()
                for _ in range(mlperf_iterations):
                    _ = mlperf_model(mlperf_input)
                torch.cuda.synchronize()
                mlperf_end = time.time()
                
                # 计算相对MLPerf分数
                mlperf_time = mlperf_end - mlperf_start
                mlperf_score = (mlperf_batch * mlperf_iterations) / mlperf_time / 100.0  # 归一化分数
                add_info("MLPerf Score", f"{mlperf_score:.2f} (relative)", reference_data.get("mlperf_score"))
                
                # 模拟AIPerf测试 (使用更大批次和更复杂模型)
                aiperf_batch = 16
                aiperf_input = torch.randn(aiperf_batch, 3, 224, 224, device=device)
                
                # 预热
                for _ in range(3):
                    _ = mlperf_model(aiperf_input)  # 复用模型
                torch.cuda.synchronize()
                
                # 测量性能
                aiperf_iterations = 5
                aiperf_start = time.time()
                for _ in range(aiperf_iterations):
                    _ = mlperf_model(aiperf_input)
                torch.cuda.synchronize()
                aiperf_end = time.time()
                
                # 计算相对AIPerf分数
                aiperf_time = aiperf_end - aiperf_start
                aiperf_score = (aiperf_batch * aiperf_iterations) / aiperf_time / 50.0  # 归一化分数
                add_info("AIPerf Score", f"{aiperf_score:.2f} (relative)", reference_data.get("aiperf_score"))
                
            except Exception as e:
                fail_text = get_text(f"失败: {str(e)}", f"Failed: {str(e)}")
                add_info("MLPerf Score", fail_text)
                add_info("AIPerf Score", fail_text)
        
        result = "\n".join(info)
        header = get_text("\n=== 设备性能测试结果 ===", "\n=== Device Performance Test Results ===")
        footer = "==========================="
        print(header)
        print("\n"+result)
        print("\n")

        help_text = "选项说明:all:所有测试,\nbasic:基础信息检测,\nprecision:混合精度性能测试,\nmemory:显存带宽测试,\noperator:算子性能测试,\nai:AI性能测试\n"
        return (result,)

NODE_CLASS_MAPPINGS = {
    "Graphics_Detection_Reference": Graphics_Detection_Reference,
}