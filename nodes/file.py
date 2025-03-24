import folder_paths
import os

DelFile = True
# DelFile = False

from ..moduel.custom_class import any
CATEGORY_NAME = "WJNode/Path"


class ComfyUI_Path_Out:
    DESCRIPTION = """
        Common paths for outputting ComfyUI 
            (root, output/input, plugin, model, cache, Python environment)
        输出ComfyUI常用路径
            (根,输出/输入,插件,模型,缓存,python环境)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("comfy_path",
                    "output_path",
                    "input_path",
                    "custom_node",
                    "model_path",
                    "cache_path",
                    "python_env")
    FUNCTION = "output_path"

    def output_path(self,):
        # comfy_path = os.path.dirname(folder_paths.__file__)
        comfy_path = folder_paths.base_path
        cache_path = f"{comfy_path}/.cache"
        custom_node = f"{comfy_path}/custom_nodes"
        python_env = os.getcwd()
        return (comfy_path,
                folder_paths.output_directory,
                folder_paths.input_directory,
                custom_node,
                folder_paths.models_dir,
                cache_path,
                python_env)


class Str_Append:
    DESCRIPTION = """
        Add prefixes and suffixes to strings, referring to Kijia's node a long time ago
        给字符串增加前缀后缀,很久以前参考kijia大佬的节点
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (any,),
            },
            "optional": {
               "prefix": ("STRING", {"default": ""}),
               "suffix": ("STRING", {"default": ""}),
               }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "String_Append"

    def String_Append(self, input, prefix="", suffix=""):
        if isinstance(input, (int, float, bool)):
            input = str(input)
        if not (isinstance(input, str)):
            print("Error: input is not a string")
            return (None,)
        return (prefix+input+suffix,)


class del_file:
    DESCRIPTION = """
        Detect whether a file or folder exists.
        If "Delete" is enabled, delete it after detection. Use with caution.
        检测文件或文件夹是否存在
        若Delete打开则在检测到后删除,谨慎使用
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Signal": (any,),
                "FilePath": ("STRING", {"default": ""},),
                "Delete": ("BOOLEAN", {"default": False},),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = (any, "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("Signal", "exists?", "is_file?",)
    FUNCTION = "Delete"

    def Delete(self, Signal, FilePath, Delete):
        exists = os.path.exists(FilePath)
        is_file = os.path.isfile(FilePath)
        if exists and Delete:
            try:
                os.remove(FilePath)
                print(f"prompt:Deleted {FilePath}")
            except:
                print(
                    f"warn:File deletion failed! \ntarget file:{FilePath}\n May not have permission")
        return (Signal, exists, is_file)


class Split_Path:
    DESCRIPTION = """
        Detect whether a file or folder exists.
        If "Deletes" is enabled, delete it after detection. Use with caution.\n
        拆分路径(盘符,路径,文件名,扩展名),检测是文件夹还是文件
        若Deletes打开则在检测到后删除,谨慎使用
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Path": ("STRING", {"default": ""},),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN", "BOOLEAN","BOOLEAN")
    RETURN_NAMES = ("drive", "path", "file_name", "file_ext","file_str", "is_file", "is_path","is_file_or_path")
    FUNCTION = "split"

    def split(self,Path):
        is_path = os.path.isdir(Path)
        is_file = os.path.isfile(Path)
        if is_path or is_file:
            drive, path_and_file = os.path.splitdrive(Path)
            path, full_file_name = os.path.split(path_and_file)
            file_name, file_ext = os.path.splitext(full_file_name)
            file_str = file_name+file_ext
            return (drive, path, file_name, file_ext, file_str , is_file, is_path, True)
        else:
            print("Error: Path is not valid")
            return (None, None, None, None, None, None, None, False)


class PrimitiveNode1:  # test node for the primitive node system
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("*",),},},
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("*",)
    FUNCTION = "doit"
    CATEGORY = "API"

    @staticmethod
    def doit(**kwargs):
        return (kwargs['value'], )


class Folder_Operations_CH:
    DESCRIPTION = """
        文件夹的增删查改操作，支持批量处理
        功能：
        - 创建文件夹（支持递归创建）
        - 删除文件夹（可选是否递归删除）
        - 查询文件夹是否存在
        - 重命名/移动文件夹
        
        注意：删除操作具有风险，请谨慎使用
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Signal": (any,),
                "operation": (["创建", "删除", "查询", "重命名/移动"], {"default": "查询"}),
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "new_path": ("STRING", {"default": ""}),  # 用于重命名/移动操作
                "recursive": ("BOOLEAN", {"default": True}),  # 用于创建和删除操作
                "batch_paths": ("STRING", {"default": "", "multiline": True}),  # 批量操作，每行一个路径
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = (any, "BOOLEAN", "STRING")
    RETURN_NAMES = ("Signal", "success", "message")
    FUNCTION = "folder_operations"

    def folder_operations(self, Signal, operation, folder_path, new_path="", recursive=True, batch_paths=""):
        import shutil
        
        # 处理批量操作
        paths_to_process = [folder_path]
        if batch_paths.strip():
            # 添加批量路径（按行分割）
            additional_paths = [p.strip() for p in batch_paths.split('\n') if p.strip()]
            paths_to_process.extend(additional_paths)
        
        success = True
        message = ""
        
        for path in paths_to_process:
            try:
                if operation == "创建":
                    # 创建文件夹
                    if not os.path.exists(path):
                        if recursive:
                            os.makedirs(path)
                        else:
                            os.mkdir(path)
                        message += f"成功创建文件夹: {path}\n"
                    else:
                        message += f"文件夹已存在: {path}\n"
                
                elif operation == "删除":
                    # 删除文件夹
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            if recursive:
                                shutil.rmtree(path)
                            else:
                                os.rmdir(path)  # 只能删除空文件夹
                            message += f"成功删除文件夹: {path}\n"
                        else:
                            message += f"路径不是文件夹: {path}\n"
                            success = False
                    else:
                        message += f"文件夹不存在: {path}\n"
                
                elif operation == "查询":
                    # 查询文件夹是否存在
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            message += f"文件夹存在: {path}\n"
                        else:
                            message += f"路径存在但不是文件夹: {path}\n"
                            success = False
                    else:
                        message += f"文件夹不存在: {path}\n"
                        success = False
                
                elif operation == "重命名/移动":
                    # 重命名/移动文件夹
                    target_path = new_path
                    # 如果是批量操作且没有指定新路径，则跳过
                    if path != folder_path and not target_path:
                        message += f"批量重命名/移动需要为每个路径指定新路径: {path}\n"
                        success = False
                        continue
                    
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            # 确保目标父目录存在
                            parent_dir = os.path.dirname(target_path)
                            if parent_dir and not os.path.exists(parent_dir):
                                if recursive:
                                    os.makedirs(parent_dir)
                                else:
                                    message += f"目标父目录不存在: {parent_dir}\n"
                                    success = False
                                    continue
                            
                            # 执行重命名/移动
                            shutil.move(path, target_path)
                            message += f"成功重命名/移动文件夹: {path} -> {target_path}\n"
                        else:
                            message += f"源路径不是文件夹: {path}\n"
                            success = False
                    else:
                        message += f"源文件夹不存在: {path}\n"
                        success = False
            
            except Exception as e:
                message += f"操作失败 ({path}): {str(e)}\n"
                success = False
        
        return (Signal, success, message.strip())


NODE_CLASS_MAPPINGS = {
    "ComfyUI_Path_Out": ComfyUI_Path_Out,
    "Str_Append": Str_Append,
    "Split_Path": Split_Path,
    # "PrimitiveNode": PrimitiveNode1
    "Folder_Operations_CH": Folder_Operations_CH
}

if DelFile:
    NODE_CLASS_MAPPINGS["del_file"] = del_file