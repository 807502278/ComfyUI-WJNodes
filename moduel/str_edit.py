import numpy as np
import ast
import re

class str_edit:
    DESCRIPTION = """
    str_to_list.convert_list(string_input, arrangement=True): Converts a string to a list.
        If arrangement=True, it cleans unnecessary characters by default.
    str_to_list.str_arrangement(user_input): Converts a string to a list-style string.
    中文说明
    str_to_list.convert_list(string_input,arrangement=True)将字符串转换为列表。
        如果arrangement=True,默认清除不必要的字符。
    str_to_list.str_arrangement(user_input)将字符串转换为列表样式的字符串。
    """
    def __init__(self):
        pass
    @classmethod
    def convert_list(cls, string_input:str,arrangement=True):
        if string_input == "":
            return ([],)
        if arrangement:
            string_input = cls.tolist_v1(string_input)
        if string_input[0] != "[":
            string_input = "[" + string_input + "]"
            return (ast.literal_eval(string_input),)
        else:
            return (ast.literal_eval(string_input),)
        
    def tolist_v1(cls,user_input):#转换为简单的带负数多维数组格式
        user_input = user_input.replace('{', '[').replace('}', ']')# 替换大括号
        user_input = user_input.replace('(', '[').replace(')', ']')# 替换小括号
        user_input = user_input.replace('，', ',')# 替换中文逗号
        user_input = re.sub(r'\s+', '', user_input)#去除空格和换行符
        user_input = re.sub(r'[^\d,.\-[\]]', '', user_input)#去除非数字字符，但不包括,.-[]
        return user_input
    @classmethod
    def tolist_v2(cls,str_input:str,to_list=True,to_oneDim=False,to_int=False,positive=False):#转换为数组格式
        if str_input == "":
            if to_list:return ([],)
            else:return ""
        else:
            str_input = str_input.replace('，', ',')# 替换中文逗号
            if to_oneDim:
                str_input = re.sub(r'[\(\)\[\]\{\}（）【】｛｝]', "" , str_input)
                str_input = "[" + str_input + "]"
            else:
                str_input=re.sub(r'[\(\[\{（【｛]', '[', str_input)#替换括号
                str_input=re.sub(r'[\)\]\}）】｝]', ']', str_input)#替换反括号
                if str_input[0] != "[":
                    str_input = "[" + str_input + "]"
            str_input = re.sub(r'[^\d,.\-[\]]', '', str_input)#去除非数字字符，但不包括,.-[]
            str_input = re.sub(r'(?<![0-9])[,]', '', str_input)#如果,前面不是数字则去除
            #str_input = re.sub(r'(-{2,}|\.{2,})', '', str_input)#去除多余的.和-
            str_input = re.sub(r'\.{2,}', '.', str_input)#去除多余的.
            if positive:
                str_input = re.sub(r'-','', str_input)#移除-
            else:
                str_input = re.sub(r'-{2,}', '-', str_input)#去除多余的-
            list1=np.array(ast.literal_eval(str_input))
            if to_int:
                list1=list1.astype(int)
            if to_list:
                return list1.tolist()
            else:
                return str_input
    @classmethod 
    def list_to_str(cls,str_list:list):
        return f"{str_list}"[1:-1].replace("'","")