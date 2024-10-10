#使用的ComfyUI-Impact-Pack批量导入https://github.com/ltdrdata/ComfyUI-Impact-Pack
import glob# 导入glob模块，用于文件路径的模式匹配
import importlib.util# 导入importlib.util模块，用于动态导入模块
import sys# 导入sys模块，用于访问与Python解释器紧密相关的变量和函数
import os
extension_folder = os.path.dirname(os.path.realpath(__file__))# 获取当前文件所在的文件夹路径
#sys.path.append(extension_folder)
NODE_CLASS_MAPPINGS = {}# 初始化两个空字典，用于存储节点类映射和节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {}

pyPath = os.path.join(extension_folder,'nodes')# 将'nodes'添加到模块搜索路径

def loadCustomNodes():# 定义loadCustomNodes函数，用于加载自定义节点和API文件
    # 使用glob.glob搜索pyPath目录下所有有.py结尾的文件，包括子目录中的文件
    find_files = glob.glob(os.path.join(pyPath, "*.py"), recursive=True)

    # 使用glob.glob搜索pyPath目录下的api.py文件
    #api_files = glob.glob(os.path.join(pyPath, "api.py"), recursive=True)
    # 将找到的节点文件和API文件列表合并
    #find_files = files + api_files

    for file in find_files:# 遍历文件列表
        file_relative_path = file[len(extension_folder):]# 计算文件相对于extension_folder的路径
        model_name = file_relative_path.replace(os.sep, '.')# 将文件路径中的目录分隔符替换为点号，构建模块名称
        model_name = os.path.splitext(model_name)[0]# 移除模块名称中的文件扩展名，得到模块名
        module = importlib.import_module(model_name, __name__)# 使用importlib.import_module动态导入模块
        # 如果模块中有NODE_CLASS_MAPPINGS属性并且它不为空，则更新全局映射
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            # 如果模块中还有NODE_DISPLAY_NAME_MAPPINGS属性并且它不为空，则更新显示名称映射
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        if hasattr(module, "init"):# 如果模块中有init函数，则调用它进行初始化
            getattr(module, "init")()

loadCustomNodes()# 调用loadCustomNodes函数，执行加载操作

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']# 定义__all__变量，列出模块中可供外部访问的变量或函数