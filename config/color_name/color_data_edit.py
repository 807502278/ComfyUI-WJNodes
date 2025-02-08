
import numpy as np
import colorsys
import json
import os

#组织结构
class Color_data_conversion:
    """
    功能：
        计算rgb值和颜色名称表的HEX/HSV/RGB空间距离/HSV空间距离，用于查找近似颜色和方便数据查询
    输入：
        初始化：
            输入1必选: 字典格式为-键为英文名对应值为[R,G,B]，RGB数据为 int 0-255
            输入2可选: 字典格式为-键为英文名对应值为中文名字符串，建议AI翻译：将这些翻译为中文，格式为一一对应的字典
        增加数据：(待开发)
            输入1必选: 输入已经处理好的数据，初始化处理的数据附加进去
    输出：
        实例属性: color_data 为初始化处理好的数据
        实例属性: color_dict 为原颜色数据
            格式：{default1_RGB:[{"name_EN":英文名,"name_CH":中文名,"distance_RGB":RGB距离,"RGB","HEX","HSV"}], #RGB距离排序数据
                 {default1_HSV:[{"name_EN":英文名,"name_CH":中文名,"distance_HSV":HSV距离,"RGB","HEX","HSV"}], #HSV距离排序数据
                 {"default1_OriginalData":[{"name_EN":英文名,"RGB"}] #原始数据,无排序
                }
        实例属性: color_default 为附加后的数据(暂时为None)
    """

    def __init__(self, color_dict: dict, color_name_ch:dict=None, debug=True):
        self.color_default = None
        self.color_name = color_name_ch
        self.color_dict = color_dict
        self.color_data = {
            "default1_RGB_D": [],
            "default1_HSV_D": [],
            "default1_value_all": []
        }
        i = 0
        for k, v in color_dict.items():
            # 计算值
            HEX = "#{:02X}{:02X}{:02X}".format(*v)  # RGB需为0-255
            HSV = self._rgb_to_hsv(v)
            name_ch = ""
            if color_name_ch is not None and color_name_ch[k] != None:
                name_ch = color_name_ch[k]

            # 写入原数据
            self.color_data["default1_value_all"].append(
                {
                    "Name_EN": k,
                    "Name_CH": name_ch,
                    "RGB": v,
                    "HEX": HEX,
                    "HSV": HSV
                }
            )

            # 写入值
            self.color_data["default1_RGB_D"].append(
                {
                    "Name_EN": k,
                    "RGB_distance": self._distance_xyz(v),
                }
            )
            self.color_data["default1_HSV_D"].append(
                {
                    "Name_EN": k,
                    "HSV_distance": self._distance_xyz(HSV),
                }
            )
            i += 1

        # 排序，方便对比相似度
        self.color_data["default1_RGB_D"].sort(key=lambda x: x["RGB_distance"])
        self.color_data["default1_HSV_D"].sort(key=lambda x: x["HSV_distance"])
        if debug:
            print(self.color_data)
    def _rgb_to_hsv(self, rgb):
        h, s, v = colorsys.rgb_to_hsv(*(np.array(rgb)/255.0))
        #return [h * 360, s, v]  # 将H的范围从[0, 1]转换为[0, 360]
        return [h, s, v]  # 将H的范围从[0, 1]转换为[0, 360]

    def _distance_xyz(self, zyz):
        return sum(np.array(zyz)**2)**0.5

#转换测试
#from default1_OriginalData import color_dict,color_name
#data = Color_data_conversion(color_dict,color_name)
#hsv = [i["HSV"] for i in data.color_data["default1_value_all"]]
#print(hsv)




#编辑ColorData

def load_color_data(color_file_path = None):#加载json
    if color_file_path is None:
        color_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "color_name_default_v3.json")
    with open(color_file_path, 'r', encoding='utf-8') as file:
        color_dict = json.load(file)
    return color_dict
    #for i in range(len(color_dict["default1_HSV"])):
    #    color_dict2["default1"][i]["rgb0-1"] = color_dict["default1_RGB"][i]["rgb0-1"]


def select_region(data, value:list, is_min=True, key:str="HSV"):#按值截断,返回选择和排除元组
    # 初始化选择列表和排除列表
    selected_list = []
    excluded_list = []

    # 获取索引和值
    index,threshold= value

    # 遍历data
    for item in data:
        # 从item中获取指定key对应的值
        item_value = item.get(key)
        if item_value is None:
            # 如果指定的key不存在于item中，跳过该item
            continue

        # 检查item_value是否为列表或元组
        if isinstance(item_value, (list, tuple)) and len(item_value) > index:
            # 获取item_value中指定索引的值
            value_to_compare = item_value[index]

            # 根据is_min的值判断是选择还是排除
            if is_min:
                if value_to_compare >= threshold:
                    selected_list.append(item)
                else:
                    excluded_list.append(item)
            else:
                if value_to_compare <= threshold:
                    selected_list.append(item)
                else:
                    excluded_list.append(item)
        else:
            # 如果item_value不是列表或元组，或者长度不足，跳过该item
            excluded_list.append(item)

    return (selected_list,excluded_list)


def NameSelect_ColorData(color_data,name_list): #按英文名称Name_EN数组选择color_data,保留结构和顺序,返回选择和排除元组
    def name_select(data_list,name_list):
        data_sel = []
        data_ex = []
        for i in range(len(data_list)):
            if data_list[i]["Name_EN"] in name_list:
                data_sel.append(data_list[i])
            else:
                data_ex.append(data_list[i])
        return data_sel,data_ex
    
    output_sel = {}
    output_ex = {}
    #{"default1_RGB_D":[],"default1_HSV_D":[],"default1_value_all":[],"default1_class":color_data["default1_class"],}
    for k,v in color_data.items():
        if k != "default1_class":
            output_sel[k],output_ex[k] = name_select(v,name_list)
        else:
            output_sel[k] = color_data["default1_class"]
            output_ex[k] = color_data["default1_class"]
    return (output_sel, output_ex)






#编辑ColorList






#检测ColorList类型
