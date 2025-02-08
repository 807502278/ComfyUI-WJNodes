#原数据
color_dict = {
    "Snow": [
        255,
        250,
        250
    ],
    "GhostWhite": [
        248,
        248,
        255
    ],
    "WhiteSmoke": [
        245,
        245,
        245
    ],
    "Gainsboro": [
        220,
        220,
        220
    ],
    "FloralWhite": [
        255,
        250,
        240
    ],
    "OldLace": [
        253,
        245,
        230
    ],
    "Linen": [
        250,
        240,
        230
    ],
    "AntiqueWhite": [
        250,
        235,
        215
    ],
    "PapayaWhip": [
        255,
        239,
        213
    ],
    "BlanchedAlmond": [
        255,
        235,
        205
    ],
    "Bisque": [
        255,
        228,
        196
    ],
    "PeachPuff": [
        255,
        218,
        185
    ],
    "NavajoWhite": [
        255,
        222,
        173
    ],
    "Moccasin": [
        255,
        228,
        181
    ],
    "Cornsilk": [
        255,
        248,
        220
    ],
    "Ivory": [
        255,
        255,
        240
    ],
    "LemonChiffon": [
        255,
        250,
        205
    ],
    "Seashell": [
        255,
        245,
        238
    ],
    "Honeydew": [
        240,
        255,
        240
    ],
    "MintCream": [
        245,
        255,
        250
    ],
    "Azure": [
        240,
        255,
        255
    ],
    "AliceBlue": [
        240,
        248,
        255
    ],
    "Lavender": [
        230,
        230,
        250
    ],
    "LavenderBlush": [
        255,
        240,
        245
    ],
    "MistyRose": [
        255,
        228,
        225
    ],
    "White": [
        255,
        255,
        255
    ],
    "Black": [
        0,
        0,
        0
    ],
    "DarkSlateGray": [
        47,
        79,
        79
    ],
    "DimGrey": [
        105,
        105,
        105
    ],
    "SlateGrey": [
        112,
        128,
        144
    ],
    "LightSlateGray": [
        119,
        136,
        153
    ],
    "Grey": [
        190,
        190,
        190
    ],
    "LightGray": [
        211,
        211,
        211
    ],
    "MidnightBlue": [
        25,
        25,
        112
    ],
    "NavyBlue": [
        0,
        0,
        128
    ],
    "CornflowerBlue": [
        100,
        149,
        237
    ],
    "DarkSlateBlue": [
        72,
        61,
        139
    ],
    "SlateBlue": [
        106,
        90,
        205
    ],
    "MediumSlateBlue": [
        123,
        104,
        238
    ],
    "LightSlateBlue": [
        132,
        112,
        255
    ],
    "MediumBlue": [
        0,
        0,
        205
    ],
    "RoyalBlue": [
        65,
        105,
        225
    ],
    "Blue": [
        0,
        0,
        255
    ],
    "DodgerBlue": [
        30,
        144,
        255
    ],
    "DeepSkyBlue": [
        0,
        191,
        255
    ],
    "SkyBlue": [
        135,
        206,
        235
    ],
    "LightSkyBlue": [
        135,
        206,
        250
    ],
    "SteelBlue": [
        70,
        130,
        180
    ],
    "LightSteelBlue": [
        176,
        196,
        222
    ],
    "LightBlue": [
        173,
        216,
        230
    ],
    "PowderBlue": [
        176,
        224,
        230
    ],
    "PaleTurquoise": [
        175,
        238,
        238
    ],
    "DarkTurquoise": [
        0,
        206,
        209
    ],
    "MediumTurquoise": [
        72,
        209,
        204
    ],
    "Turquoise": [
        64,
        224,
        208
    ],
    "Cyan": [
        0,
        255,
        255
    ],
    "LightCyan": [
        224,
        255,
        255
    ],
    "CadetBlue": [
        95,
        158,
        160
    ],
    "MediumAquamarine": [
        102,
        205,
        170
    ],
    "Aquamarine": [
        127,
        255,
        212
    ],
    "DarkGreen": [
        0,
        100,
        0
    ],
    "DarkOliveGreen": [
        85,
        107,
        47
    ],
    "DarkSeaGreen": [
        143,
        188,
        143
    ],
    "SeaGreen": [
        46,
        139,
        87
    ],
    "MediumSeaGreen": [
        60,
        179,
        113
    ],
    "LightSeaGreen": [
        32,
        178,
        170
    ],
    "PaleGreen": [
        152,
        251,
        152
    ],
    "SpringGreen": [
        0,
        255,
        127
    ],
    "LawnGreen": [
        124,
        252,
        0
    ],
    "Green": [
        0,
        255,
        0
    ],
    "Chartreuse": [
        127,
        255,
        0
    ],
    "MedSpringGreen": [
        0,
        250,
        154
    ],
    "GreenYellow": [
        173,
        255,
        47
    ],
    "LimeGreen": [
        50,
        205,
        50
    ],
    "YellowGreen": [
        154,
        205,
        50
    ],
    "ForestGreen": [
        34,
        139,
        34
    ],
    "OliveDrab": [
        107,
        142,
        35
    ],
    "DarkKhaki": [
        189,
        183,
        107
    ],
    "PaleGoldenrod": [
        238,
        232,
        170
    ],
    "LtGoldenrodYello": [
        250,
        250,
        210
    ],
    "LightYellow": [
        255,
        255,
        224
    ],
    "Yellow": [
        255,
        255,
        0
    ],
    "Gold": [
        255,
        215,
        0
    ],
    "LightGoldenrod": [
        238,
        221,
        130
    ],
    "goldenrod": [
        218,
        165,
        32
    ],
    "DarkGoldenrod": [
        184,
        134,
        11
    ],
    "RosyBrown": [
        188,
        143,
        143
    ],
    "IndianRed": [
        205,
        92,
        92
    ],
    "SaddleBrown": [
        139,
        69,
        19
    ],
    "Sienna": [
        160,
        82,
        45
    ],
    "Peru": [
        205,
        133,
        63
    ],
    "Burlywood": [
        222,
        184,
        135
    ],
    "Beige": [
        245,
        245,
        220
    ],
    "Wheat": [
        245,
        222,
        179
    ],
    "SandyBrown": [
        244,
        164,
        96
    ],
    "Tan": [
        210,
        180,
        140
    ],
    "Chocolate": [
        210,
        105,
        30
    ],
    "Firebrick": [
        178,
        34,
        34
    ],
    "Brown": [
        165,
        42,
        42
    ],
    "DarkSalmon": [
        233,
        150,
        122
    ],
    "Salmon": [
        250,
        128,
        114
    ],
    "LightSalmon": [
        255,
        160,
        122
    ],
    "Orange": [
        255,
        165,
        0
    ],
    "DarkOrange": [
        255,
        140,
        0
    ],
    "Coral": [
        255,
        127,
        80
    ],
    "LightCoral": [
        240,
        128,
        128
    ],
    "Tomato": [
        255,
        99,
        71
    ],
    "OrangeRed": [
        255,
        69,
        0
    ],
    "Red": [
        255,
        0,
        0
    ],
    "HotPink": [
        255,
        105,
        180
    ],
    "DeepPink": [
        255,
        20,
        147
    ],
    "Pink": [
        255,
        192,
        203
    ],
    "LightPink": [
        255,
        182,
        193
    ],
    "PaleVioletRed": [
        219,
        112,
        147
    ],
    "Maroon": [
        176,
        48,
        96
    ],
    "MediumVioletRed": [
        199,
        21,
        133
    ],
    "VioletRed": [
        208,
        32,
        144
    ],
    "Magenta": [
        255,
        0,
        255
    ],
    "Violet": [
        238,
        130,
        238
    ],
    "Plum": [
        221,
        160,
        221
    ],
    "Orchid": [
        218,
        112,
        214
    ],
    "MediumOrchid": [
        186,
        85,
        211
    ],
    "DarkOrchid": [
        153,
        50,
        204
    ],
    "DarkViolet": [
        148,
        0,
        211
    ],
    "BlueViolet": [
        138,
        43,
        226
    ],
    "Purple": [
        160,
        32,
        240
    ],
    "MediumPurple": [
        147,
        112,
        219
    ],
    "Thistle": [
        216,
        191,
        216
    ]
}

color_name = {
    "Snow": "雪白",
    "GhostWhite": "幽灵白",
    "WhiteSmoke": "白烟",
    "Gainsboro": "甘蔗灰",
    "FloralWhite": "花白",
    "OldLace": "老花边",
    "Linen": "亚麻",
    "AntiqueWhite": "古董白",
    "PapayaWhip": "木瓜色",
    "BlanchedAlmond": "杏仁色",
    "Bisque": "桔黄色",
    "PeachPuff": "桃色",
    "NavajoWhite": "纳瓦霍白",
    "Moccasin": "鹿皮色",
    "Cornsilk": "玉米丝",
    "Ivory": "象牙色",
    "LemonChiffon": "柠檬绸",
    "Seashell": "贝壳色",
    "Honeydew": "哈密瓜色",
    "MintCream": "薄荷奶油",
    "Azure": "天蓝色",
    "AliceBlue": "爱丽丝蓝",
    "Lavender": "薰衣草色",
    "LavenderBlush": "薰衣草粉",
    "MistyRose": "薄雾玫瑰",
    "White": "白色",
    "Black": "黑色",
    "DarkSlateGray": "深石板灰",
    "DimGrey": "暗灰色",
    "SlateGrey": "石板灰",
    "LightSlateGray": "浅石板灰",
    "Grey": "灰色",
    "LightGray": "浅灰色",
    "MidnightBlue": "午夜蓝",
    "NavyBlue": "海军蓝",
    "CornflowerBlue": "矢车菊蓝",
    "DarkSlateBlue": "深石板蓝",
    "SlateBlue": "石板蓝",
    "MediumSlateBlue": "中等石板蓝",
    "LightSlateBlue": "浅石板蓝",
    "MediumBlue": "中等蓝",
    "RoyalBlue": "皇家蓝",
    "Blue": "蓝色",
    "DodgerBlue": "道奇蓝",
    "DeepSkyBlue": "深天蓝",
    "SkyBlue": "天蓝",
    "LightSkyBlue": "浅天蓝",
    "SteelBlue": "钢蓝",
    "LightSteelBlue": "浅钢蓝",
    "LightBlue": "浅蓝",
    "PowderBlue": "粉末蓝",
    "PaleTurquoise": "浅绿松石色",
    "DarkTurquoise": "深绿松石色",
    "MediumTurquoise": "中等绿松石色",
    "Turquoise": "绿松石色",
    "Cyan": "青色",
    "LightCyan": "浅青色",
    "CadetBlue": "军校蓝",
    "MediumAquamarine": "中等碧绿色",
    "Aquamarine": "碧绿色",
    "DarkGreen": "深绿色",
    "DarkOliveGreen": "深橄榄绿",
    "DarkSeaGreen": "深海绿",
    "SeaGreen": "海绿",
    "MediumSeaGreen": "中等海绿",
    "LightSeaGreen": "浅海绿",
    "PaleGreen": "浅绿色",
    "SpringGreen": "春绿",
    "LawnGreen": "草坪绿",
    "Green": "绿色",
    "Chartreuse": "查特酒绿",
    "MedSpringGreen": "中等春绿",
    "GreenYellow": "绿黄色",
    "LimeGreen": "酸橙绿",
    "YellowGreen": "黄绿色",
    "ForestGreen": "森林绿",
    "OliveDrab": "橄榄绿",
    "DarkKhaki": "深卡其色",
    "PaleGoldenrod": "浅金菊黄",
    "LtGoldenrodYello": "浅金菊黄",
    "LightYellow": "浅黄色",
    "Yellow": "黄色",
    "Gold": "金色",
    "LightGoldenrod": "浅金菊黄",
    "goldenrod": "金菊黄",
    "DarkGoldenrod": "深金菊黄",
    "RosyBrown": "玫瑰棕",
    "IndianRed": "印第安红",
    "SaddleBrown": "马鞍棕",
    "Sienna": "赭石色",
    "Peru": "秘鲁色",
    "Burlywood": "厚木色",
    "Beige": "米色",
    "Wheat": "小麦色",
    "SandyBrown": "沙棕色",
    "Tan": "棕褐色",
    "Chocolate": "巧克力色",
    "Firebrick": "红砖色",
    "Brown": "棕色",
    "DarkSalmon": "深鲑鱼色",
    "Salmon": "鲑鱼色",
    "LightSalmon": "浅鲑鱼色",
    "Orange": "橙色",
    "DarkOrange": "深橙色",
    "Coral": "珊瑚色",
    "LightCoral": "浅珊瑚色",
    "Tomato": "番茄色",
    "OrangeRed": "橙红色",
    "Red": "红色",
    "HotPink": "热粉红",
    "DeepPink": "深粉红",
    "Pink": "粉红",
    "LightPink": "浅粉红",
    "PaleVioletRed": "浅紫红",
    "Maroon": "栗色",
    "MediumVioletRed": "中等紫红",
    "VioletRed": "紫红",
    "Magenta": "品红",
    "Violet": "紫罗兰色",
    "Plum": "李子色",
    "Orchid": "兰花色",
    "MediumOrchid": "中等兰花色",
    "DarkOrchid": "深兰花色",
    "DarkViolet": "深紫罗兰色",
    "BlueViolet": "蓝紫罗兰色",
    "Purple": "紫色",
    "MediumPurple": "中等紫色",
    "Thistle": "蓟色",
}
