#使用的ComfyUI-Impact-Pack批量导入https://github.com/ltdrdata/ComfyUI-Impact-Pack
import glob# 导入glob模块，用于文件路径的模式匹配
import importlib.util# 导入importlib.util模块，用于动态导入模块
import os

extension_folder = os.path.dirname(os.path.realpath(__file__))# 获取当前文件所在的文件夹路径
NODE_CLASS_MAPPINGS = {}# 初始化节点类映射
NODE_DISPLAY_NAME_MAPPINGS = {} #初始化节点显示名称映射
WEB_DIRECTORY = "./web" #指定js代码加载路径

pyPath = [os.path.join(extension_folder,'nodes'), # 'nodes'添加到节点搜索
          os.path.join(extension_folder,'nodes','nodes_test'), # 测试节点搜索
          os.path.join(extension_folder,'Other','nodes_hide'), 
          ]

def loadCustomNodes(pyPath):# 加载自定义节点和API文件
    find_files = []
    for i in pyPath:
        if os.path.isdir(i):
            files = glob.glob(os.path.join(i, "*.py"))
            find_files = find_files + files
    if len(find_files) == 0:
        print("Error:Node code not found, cancel importing ComfyUI-WJNode!")
    else:
        for file in find_files:# 遍历文件列表
            file_relative_path = file[len(extension_folder):]# 计算文件相对于extension_folder的路径
            model_name = file_relative_path.replace(os.sep, '.')# 将文件路径中的目录分隔符替换为点号，构建模块名称
            model_name = os.path.splitext(model_name)[0]# 移除扩展名，得到模块名
            module = importlib.import_module(model_name, __name__)# 使用importlib.import_module动态导入模块
            # 如果模块中有NODE_CLASS_MAPPINGS属性并且它不为空，则更新全局映射
            if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                # 如果模块中还有NODE_DISPLAY_NAME_MAPPINGS属性并且它不为空，则更新显示名称映射
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                # 没有NODE_DISPLAY_NAME_MAPPINGS属性，则自动名称映射(下划线替换为空格作为名称)
                else:
                    NODE_DISPLAY_NAME_MAPPINGS.update({i:i.replace("_"," ") for i in module.NODE_CLASS_MAPPINGS.keys()})
            if hasattr(module, "init"):# 如果模块中有init函数，则调用它进行初始化
                getattr(module, "init")()

loadCustomNodes(pyPath)

# 定义__all__变量，列出模块中可供外部访问的变量或函数
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']