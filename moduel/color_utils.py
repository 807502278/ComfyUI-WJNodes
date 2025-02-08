import colorsys
from typing import Union, List, Tuple
import numpy as np

def convert_color(colors: Union[List[List], List[str]], #将输入的颜色列表值的类型转换为指定的输出类型
                    input_type: str,
                    output_type: str
                    ) -> Union[List[List], List[str]]:
    """
    说明：将输入的颜色列表值的类型转换为指定的输出类型。
    参数:
    - colors: 输入的颜色值列表，可以是 RGB (0-1)、RGB (0-255)、HEX 或 HSV。
    - input_type/output_type: 输入/输出颜色值的类型，只能是 'RGB0-1', 'RGB', 'HEX', 'HSV'
    返回:
    - 转换后的颜色值列表。

    instructions: Convert the input color value to the specified output type.
    Input instructions:
        colors: Union[List[Tuple[float, float, float]],  List[str]]
        input_type/output_type : ('RGB0-1', 'RGB','HEX', 'HSV')
    Output instructions:
        color list : Union[List[Tuple[float, float, float]], List[str]]
    """
    #检测输入类型，若不匹配则自动更改类型
    if isinstance(colors[0],str) and input_type != "hex":
        print(f"Warning: Detected that the input class {input_type} is actually a hex class and has been automatically changed to hex")
        input_type = 'HEX'
    elif isinstance(colors[0],list) or isinstance(colors[0],tuple):
        if isinstance(colors[0][0],int) and input_type != "rgb0-255":
            print(f"Warning: Detected that the input class {input_type} is actually an int type and has been changed to rgb0-255")
            input_type = 'RGB'
        elif isinstance(colors[0][0],float) and (input_type != "rgb0-1" or input_type != "hsv"):
            print("Warning: Detected that the input class rgb0-255 is actually a float type and has been changed to rgb0-1")
            input_type = 'RGB0-1'
    
    import webcolors
    #转换函数
    def rgb01_to_rgb255(rgb): return tuple(int(x * 255) for x in rgb)
    def rgb255_to_rgb01(rgb): return tuple(x / 255.0 for x in rgb)
    def rgb01_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*(int(x * 255) for x in rgb))
    def rgb255_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)
    def hex_to_rgb01(hex_color):
        rgb = webcolors.hex_to_rgb(hex_color)
        return tuple(x / 255.0 for x in rgb)
    def hex_to_rgb255(hex_color): return webcolors.hex_to_rgb(hex_color)
    def rgb01_to_hsv(rgb): return colorsys.rgb_to_hsv(*rgb)
    def rgb255_to_hsv(rgb): return colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    def hsv_to_rgb01(hsv): return colorsys.hsv_to_rgb(*hsv)
    def hsv_to_rgb255(hsv): return tuple(int(x * 255) for x in colorsys.hsv_to_rgb(*hsv))
    def hsv_to_hex(hsv): return rgb01_to_hex(hsv_to_rgb01(hsv))
    def hex_to_hsv(hex_color): return rgb01_to_hsv(hex_to_rgb01(hex_color))

    # 定义转换函数
    conversion_functions = {
        ('RGB0-1', 'RGB'): rgb01_to_rgb255,
        ('RGB0-1', 'HSV'): rgb01_to_hsv,
        ('RGB0-1', 'HEX'): rgb01_to_hex,
        ('RGB', 'RGB0-1'): rgb255_to_rgb01,
        ('RGB', 'HEX'): rgb255_to_hex,
        ('RGB', 'HSV'): rgb255_to_hsv,
        ('HEX', 'RGB0-1'): hex_to_rgb01,
        ('HEX', 'RGB'): hex_to_rgb255,
        ('HEX', 'HSV'): hex_to_hsv,
        ('HSV', 'RGB0-1'): hsv_to_rgb01,
        ('HSV', 'RGB'): hsv_to_rgb255,
        ('HSV', 'HEX'): hsv_to_hex,

    }
    # 检查输入和输出类型是否有效
    if (input_type, output_type) not in conversion_functions:
        raise ValueError(f"Unsupported conversion from {input_type} to {output_type}")
    elif input_type != output_type:
        convert = conversion_functions[(input_type, output_type)] # 获取转换函数
        result = [convert(color) for color in colors] # 转换颜色值
        return result
    else:
        return colors


def distance_color(colors: Union[List[List], List[str]], #计算指定类型的颜色空间距离
                   input_type: str,
                   output_type: str = None
                   )-> List[List]:
    """
    说明: 计算指定类型的颜色空间距离
    输入: 
        colors: 输入的颜色值列表，可以是 RGB (0-1)、RGB (0-255)、HEX 或 HSV。
        input_type: 输入颜色值的类型，只能是 'rgb01', 'rgb255', 'HEX', 'HSV'
        output_type: 输出颜色值的类型，只能是 'rgb01', 'rgb255', 'HSV'
    输出: 
        颜色与0,0,0的距离列表List[float]
    instructions: Calculate the color space distance of a specified type
    Input instructions:
        colors: Union[List[Tuple[float, float, float]], List[str]]
        input_type : ('RGB0-1', 'RGB','HEX', 'HSV')
        output_type : ('RGB0-1', 'RGB','HSV')
    Output instructions:
        Color Space Distance : List[float]
    """
    all_type = ('RGB0-1', 'RGB','HEX', 'HSV')
    distance_type = ('RGB','HSV')
    if input_type not in all_type or output_type not in distance_type:
        raise ValueError(f"Unsupported conversion from {input_type} to {output_type}")
    if input_type != output_type :
        colors = convert_color(colors,input_type,output_type)
    return [sum(i**2)**0.5 for i in np.array(colors)]


def convert_rgb_1_255(rgb:list,to255:bool=None): #默认rgb0-1与rgb0-255互转，to255可指定是否转0-255
    def to_255(rgb,inttype,to255):
        if inttype == "int":
            if to255 is not None:
                if not to255: rgb = (np.array(rgb,dtype=np.float32)/255.0).tolist()
            else: rgb = (np.array(rgb,dtype=np.float32)/255.0).tolist()
        else:
            if to255 is not None:
                if to255: rgb = (np.array(rgb)*255).astype(np.uint8).tolist()
            else: rgb = (np.array(rgb)*255).astype(np.uint8).tolist()
        return rgb

    if isinstance(rgb[0],list) or isinstance(rgb[0],tuple):
        if isinstance(rgb[0][0],int):
            rgb = to_255(rgb,"int",to255)
        elif isinstance(rgb[0][0],float):
            rgb = to_255(rgb,"float",to255)
        elif isinstance(rgb[0][0],list) or isinstance(rgb[0][0],tuple):
            raise TypeError("Error:color_utils/convert_rgb_1_255 Too many dimensions of RGB data input !")
        else:
            raise TypeError("Error:color_utils/convert_rgb_1_255 Unknown RGB array data !")
    elif isinstance(rgb[0],int):
        rgb = to_255(rgb,"int",to255)
    elif isinstance(rgb[0],float):
        rgb = to_255(rgb,"float",to255)
    else:
        raise TypeError("Error:color_utils/convert_rgb_1_255 Incorrect RGB data structure input !")
    return rgb
        