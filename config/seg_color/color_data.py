import json
import re
import os

#定义用于DensePose中英文筛选的数据
DensePose_Data = {
    "tag_en" : ["arm","forearm","thigh","calf","head","body","hand","foot",
              "left","right","front","behind","background",],

    "tag_ch" : ["胳","臂","大","腿","头","身","手","脚",
              "左","右","前","后","景"],

    "DP_tag_en2ch" : {'arm': '胳',
                    'forearm': '臂',
                    'thigh': '大',
                    'calf': '腿',
                    'head': '头',
                    'body': '身',
                    'hand': '手',
                    'foot': '脚',
                    'left': '左',
                    'right': '右',
                    'front': '前',
                    'behind': '后',
                    'inside': '内',
                    'outside': '外',
                    'background': '景'},

    "DP_filter" : {
                'left arm outside': {'keyword_ch': ['左', '胳', '膊', '外'], 'keyword_en': ['left', 'arm', 'outside']}, 
                 'left arm inside': {'keyword_ch': ['左', '胳', '膊', '内'], 'keyword_en': ['left', 'arm', 'inside']}, 
                 'right arm outside': {'keyword_ch': ['右', '胳', '膊', '外'], 'keyword_en': ['right', 'arm', 'outside']}, 
                 'right arm inside': {'keyword_ch': ['右', '胳', '膊', '内'], 'keyword_en': ['right', 'arm', 'inside']}, 
                 'left forearm outside': {'keyword_ch': ['左', '小', '臂', '外'], 'keyword_en': ['left', 'forearm', 'outside']}, 
                 'left forearm inside': {'keyword_ch': ['左', '小', '臂', ' 内'], 'keyword_en': ['left', 'forearm', 'inside']}, 
                 'right forearm outside': {'keyword_ch': ['右', '小', '臂', '外'], 'keyword_en': ['right', 'forearm', 'outside']}, 
                 'right forearm inside': {'keyword_ch': ['右', '小', '臂', '内'], 'keyword_en': ['right', 'forearm', 'inside']}, 
                 'left thigh front': {'keyword_ch': ['左', '大', '腿', '前'], 'keyword_en': ['left', 'thigh', 'front']}, 
                 'left thigh behind': {'keyword_ch': ['左', '大', '腿', '后'], 'keyword_en': ['left', 'thigh', 'behind']}, 
                 'right thigh front': {'keyword_ch': ['右', '大', '腿', '前'], 'keyword_en': ['right', 'thigh', 'front']}, 
                 'right thigh behind': {'keyword_ch': ['右', '大', '腿', '后'], 'keyword_en': ['right', 'thigh', 'behind']}, 
                 'left calf front': {'keyword_ch': ['左', '小', '腿', '前'], 'keyword_en': ['left', 'calf', 'front']}, 
                 'left calf behind': {'keyword_ch': ['左', '小', '腿', '后'], 'keyword_en': ['left', 'calf', 'behind']}, 
                 'right calf front': {'keyword_ch': ['右', '小', '腿', '前'], 'keyword_en': ['right', 'calf', 'front']}, 
                 'right calf behind': {'keyword_ch': ['右', '小', '腿', '后'], 'keyword_en': ['right', 'calf', 'behind']}, 
                 'left head': {'keyword_ch': ['左', '头', '部'], 'keyword_en': ['left', 'head']}, 
                 'right head': {'keyword_ch': ['右', '头', '部'], 'keyword_en': ['right', 'head']}, 
                 'front body': {'keyword_ch': ['身', '体', '前'], 'keyword_en': ['front', 'body']}, 
                 'behind body': {'keyword_ch': ['身', '体', '后'], 'keyword_en': ['behind', 'body']}, 
                 'left hand': {'keyword_ch': ['左', '手'], 'keyword_en': ['left', 'hand']}, 
                 'right hand': {'keyword_ch': ['右', '手'], 'keyword_en': ['right', 'hand']}, 
                 'left foot': {'keyword_ch': ['左', '脚'], 'keyword_en': ['left', 'foot']}, 
                 'right foot': {'keyword_ch': ['右', '脚'], 'keyword_en': ['right', 'foot']}, 
                 'back': {'keyword_ch': ['景'], 'keyword_en': ['back']}
                 },

    "DP_filter_en" : {'left arm outside': ['left', 'arm', 'outside'], 
                    'left arm inside': ['left', 'arm', 'inside'], 
                    'right arm outside': ['right', 'arm', 'outside'], 
                    'right arm inside': ['right', 'arm', 'inside'], 
                    'left forearm outside': ['left', 'forearm', 'outside'], 
                    'left forearm inside': ['left', 'forearm', 'inside'], 
                    'right forearm outside': ['right', 'forearm', 'outside'], 
                    'right forearm inside': ['right', 'forearm', 'inside'], 
                    'left thigh front': ['left', 'thigh', 'front'], 
                    'left thigh behind': ['left', 'thigh', 'behind'], 
                    'right thigh front': ['right', 'thigh', 'front'], 
                    'right thigh behind': ['right', 'thigh', 'behind'], 
                    'left calf front': ['left', 'calf', 'front'], 
                    'left calf behind': ['left', 'calf', 'behind'], 
                    'right calf front': ['right', 'calf', 'front'], 
                    'right calf behind': ['right', 'calf', 'behind'], 
                    'left head': ['left', 'head'], 
                    'right head': ['right', 'head'], 
                    'front body': ['front', 'body'], 
                    'behind body': ['behind', 'body'], 
                    'left hand': ['left', 'hand'], 
                    'right hand': ['right', 'hand'], 
                    'left foot': ['left', 'foot'], 
                    'right foot': ['right', 'foot'], 
                    'back': ['back']},

    "DP_filter_ch" : {'左胳膊外侧': ['左', '胳', '膊', '外'], 
                    '左胳膊内侧': ['左', '胳', '膊', '内'], 
                    '右胳膊外侧': ['右', '胳', '膊', '外'], 
                    '右胳膊内侧': ['右', '胳', '膊', '内'], 
                    '左小臂外侧': ['左', '臂', '外'], 
                    '左小臂内侧': ['左', '臂', '内'], 
                    '右小臂外侧': ['右', '臂', '外'], 
                    '右小臂内侧': ['右', '臂', '内'], 
                    '左大腿前面': ['左', '大', '腿', '前'], 
                    '左大腿后面': ['左', '大', '腿', '后'], 
                    '右大腿前面': ['右', '大', '腿', '前'], 
                    '右大腿后面': [' 右', '大', '腿', '后'], 
                    '左小腿前面': ['左', '腿', '前'], 
                    '左小腿后面': ['左', '腿', '后'], 
                    '右小腿前面': ['右', '腿', '前'], 
                    '右小腿后面': ['右', '腿', '后'], 
                    '左侧头部': ['左', '头'], 
                    '右侧头部': ['右', '头'], 
                    '身体前面': ['身', '体', '前'], 
                    '身体后面': ['身', '体', '后'], 
                    '左手': ['左', '手'], 
                    '右手': ['右', '手'], 
                    '左脚': ['左', '脚'], 
                    '右脚': ['右', '脚'], 
                    '背景': ['景']
                    },
}

def GetDP_FilterList(file_path=None): #获取json数据中英文名称key并转换为列表
    if file_path == None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "DensePose_v1.1_Parula(CivitAI).json")
    with open(file_path, 'r', encoding='utf-8') as file:
        color_data = json.load(file)
    color_data_k = list(color_data.keys())

    filter_en = [re.split(" ",i) for i in color_data.keys()]

    ch_del_list = ["侧","面","背","小","部"]
    DP_ch_data = [list(i["CH"]) for i in color_data.values()]
    DP_filter_ch = {}
    DP_filter_en = {}
    for i in range(len(DP_ch_data)):
        k = color_data_k[i]
        DP_filter_en[k] = filter_en[i]
        DP_filter_ch[color_data[k]["CH"]] = [i for i in DP_ch_data[i] if i not in ch_del_list]
    return DP_filter_en,DP_filter_ch

#DP_filter_en,DP_filter_ch = GetDP_FilterList() #建立查找字典，节省加载时间已改为静态，dp姿态数据更改时需重新建立
#print(DP_filter_ch)