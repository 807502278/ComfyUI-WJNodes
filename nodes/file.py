import folder_paths
import os

DelFile = True
# DelFile = False

from ..moduel.custom_class import any
CATEGORY_NAME_WJnode = "WJNode/Path"


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
    CATEGORY = CATEGORY_NAME_WJnode
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
    CATEGORY = CATEGORY_NAME_WJnode
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
    CATEGORY = CATEGORY_NAME_WJnode
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
    CATEGORY = CATEGORY_NAME_WJnode
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "BOOLEAN",)
    RETURN_NAMES = ("drive", "path", "Documents", "Extension", "is_file",)
    FUNCTION = "split"

    def split(file_path):
        is_path = os.path.isdir(file_path)
        is_file = os.path.isfile(file_path)
        if is_path or is_file:
            drive, path_and_file = os.path.splitdrive(file_path)
            path, full_file_name = os.path.split(path_and_file)
            file_name, ext = os.path.splitext(full_file_name)
            return (drive, path, file_name, ext, is_file)
        else:
            print("Error: Path is not valid")
            return (None, None, None, None, None)


class PrimitiveNode1:  # test node for the primitive node system
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("*",),
        },
        },
    CATEGORY = CATEGORY_NAME_WJnode
    RETURN_TYPES = ("*",)
    FUNCTION = "doit"
    CATEGORY = "API"

    @staticmethod
    def doit(**kwargs):
        return (kwargs['value'], )


NODE_CLASS_MAPPINGS = {
    "ComfyUI_Path_Out": ComfyUI_Path_Out,
    "Str_Append": Str_Append,
    "Split_Path": Split_Path,
    # "PrimitiveNode": PrimitiveNode1
}

if DelFile:
    NODE_CLASS_MAPPINGS["del_file"] = del_file