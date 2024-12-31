
**ComfyUI-WJNodes explain**

- Ready to use upon download. No need to install dependencies for the time being.
- If there are new functions or suggestions, please provide feedback.
- Attention! The delfile node is not recommended for use on servers. I am not responsible for any losses incurred.
- To disable the delfile node, change the 'DelFile=True' of node/file.py to 'DelFile=False'

**Node list:**

- Image: WJNode/Image
  - ✅`load image from path` : Load image from path
  - ✅`save image to path` : Save image by overwriting the path
  - ✅`save image out` : Save image to output and output the path
  - ✅`select images batch` : Batch selection and recombination of batches
  - ✅`load image adv` : Load image with mask inversion and path output
  - 🟩Load value feature recognition model (e.g., nsfw, aesthetic score, AI value, time)
  - 🟩Input recognition model and image batch, output batch and corresponding feature values
  - 🟩Sort image batches through specified arrays (e.g., feature value arrays)
- Mask Editing: WJNode/MaskEdit
  - ✅`load_color_config` : Load color configuration for color block to mask, currently supports loading ADE20K pre-processing color data
  - ✅`color_segmentation` : Color block to mask, currently supports pre-processing ADE20K and SAM2 data
  - ✅`color_segmentation_v2` : Color block to mask v2, uses keys in color configuration to select masks, only supports ADE20K data
  - ✅`mask_select_mask` : Mask selection within a mask batch (intersection represents selection)
  - 🟩`coords_select_mask` : Coordinate selection of masks, used to assist SAM2 video keying (under development)
  - ✅`mask_line_mapping` : Mask line mapping, can automatically calculate maximum and minimum values when input is -1 or 256, 
                  can map to specified values
  - ✅`mask_and_mask_math` : Mask to mask operations, supports addition/subtraction/intersection/multiplication operations, 
                  Adjustable cv2 and torch modes, if cv2 is not installed, automatically switches to torch
  - 🟩`Accurate_mask_clipping` : Precise search for mask bbox boundaries (under development)
- Image Editing: WJNode/ImageEdit
  - ✅`adv crop` : Advanced cropping: can quickly crop/expand/move/flip images, can output background masks and custom filling 
                  (Usage method included in the node, known bug: expansion size more than 1 times cannot use tiling and mirror filling)
  - ✅`mask detection` : Mask detection: detect whether there is a mask, detect whether it is all hard edges, 
                  detect whether the mask is pure white/pure black/pure gray and output values 0-255
  - ✅`InvertChannelAdv` : Invert/separate image channels Image ⭐
                  RGBA to mask batch Replace channels 
                  Any channel to RGBA
  - ✅`Video_fade` : Two video segments can choose two ways to fade in and out, 
                  Mask: Local fade in and out under development... 
                  Exponential: Exponential gradient under development...
- Others: WJNode/Other-functions
  - ✅`any_data` : Group any data, known bug: nested grouping will split
  - ✅`show_type` : Display data type
  - ✅`array_element_count` : Display the number of array elements
- Plugins: WJNode/Other-plugins
  - ✅`WAS_Mask_Fill_Region_batch` : Optimize WAS plugin's WAS_Mask_Fill_Region (mask cleanup) to support batches
  - ✅`SegmDetectorCombined_batch` : Optimize impack-pack plugin's SegmDetectorCombined (segm detection mask) to support batches
  - ✅`bbox_restore_mask` : Add impack-pack plugin's seg decomposition, restore cropped images through cropping data (SEG editing)
  - ✅`Sam2AutoSegmentation_data` : Add Sam2AutoSegmentation (kijia) node's color list/coordinate output, used to assist SAM2 video keying
- Path: WJNode/Path
  - ✅`comfyui path` : Output comfyui common paths (root, output/input, plugins, models, cache, Python environment)
  - ✅`path append` : Add prefix/suffix to strings (reference KJNode)
  - ✅`del file` : Detect whether file or path exists, whether to delete file, operation requires input signal, deletion requires write permission
  - ✅`split path` : Path slicing, input path, output: disk symbol/path/file/extension + detect whether it is a file


**ComfyUI-WJNodes介绍**

- 下载即用，暂时无需安装依赖(可选安装cv2)，有新功能或建议请反馈。
- 注意！delfile节点不建议在服务器上使用，产生任何损失与本人无关
- 修改node/file.py的“DelFile = True”为“DelFile = False”即可禁用delfile节点

**节点列表**

- 图像：WJNode/Image
  - ✅`load image from path` : 从路径加载图片
  - ✅`save image to path` : 通过路径覆盖保存图片
  - ✅`save image out` : 保存图片到output并输出该路径
  - ✅`select images batch` : 批次选择和重新组合批次
  - ✅`load image adv` : 带遮罩反转和路径输出的加载图片
  - 🟩加载值特征识别模型(例如nsfw,美学分数,AI值,time)
  - 🟩输入识别模型和图像批次，输出批次和对应特征值
  - 🟩通过指定数组(例如特征值数组)排序图片批次
- 遮罩编辑：WJNode/MaskEdit
  - ✅`load_color_config` : 加载颜色配置，用于色块转遮罩,目前支持加载 ADE20K 预处理颜色数据
  - ✅`color_segmentation` : 色块转遮罩，目前支持预处理 ADE20K 和 SAM2 数据
  - ✅`color_segmentation_v2` : 色块转遮罩v2，使用颜色配置内的key来选取遮罩，仅支持ADE20K数据
  - ✅`mask_select_mask` : 遮罩选择遮罩批次内的遮罩(有交集即代表选择)
  - 🟩`coords_select_mask` : 坐标选择遮罩，用于辅助SAM2视频抠图(待开发)
  - ✅`mask_line_mapping` : 遮罩线映射，当输入为-1或256时可自动计算最值，可映射到指定值
  - ✅`mask_and_mask_math` : 遮罩与遮罩的运算，支持加/减/(交集)/乘运算，
                            可调cv2和torch两种模式,若未安装cv2则自动切换到torch
  - 🟩`Accurate_mask_clipping` : 精确查找遮罩bbox边界 (待开发)
- 图像编辑：WJNode/ImageEdit
  - ✅`adv crop` : 高级裁剪:可快速裁剪/扩展/移动/翻转图片,可输出背景遮罩和自定义填充
  -   (节点内附使用方法,已知bug:扩展尺寸超过1倍时无法使用平铺和镜像填充)
  - ✅`mask detection` : 遮罩检测:检测是否有遮罩,检测是否是全硬边,检测遮罩是否是纯白/纯黑/纯灰并输出值0-255
  - ✅`InvertChannelAdv` : 翻转/分离图像通道⭐
                          图像RGBA转遮罩批次
                          替换通道
                          任意通道合成RGBA
  - ✅`Video_fade` : 两段视频可选两种方式渐入渐出，
                          遮罩:局部渐入渐出开发中...
                          指数:指数渐变开发中...
- 其它：WJNode/Other-functions
  - ✅`any_data` : 将任意数据打组，已知bug:嵌套打组会裂开
  - ✅`show_type` : 显示数据类型
  - ✅`array_element_count` : 显示数组元素数量
- 插件：WJNode/Other-plugins
  - ✅`WAS_Mask_Fill_Region_batch` : 优化WAS插件的的WAS_Mask_Fill_Region(遮罩清理)支持批次
  - ✅`SegmDetectorCombined_batch` : 优化impack-pack插件的的SegmDetectorCombined(segm检测遮罩)支持批次
  - ✅`bbox_restore_mask` : 增加impack-pack插件的seg分解后，通过裁剪数据恢复裁剪后的图像（SEG编辑）
  - ✅`Sam2AutoSegmentation_data` : 增加Sam2AutoSegmentation(kijia)节点的颜色列表/坐标输出，用于辅助SAM2视频抠图
- 路径：WJNode/Path
  - ✅`comfyui path` : 输出comfyui常用路径(根,输出/输入,插件,模型,缓存,python环境)
  - ✅`path append` : 给字符串增加前缀后缀(参考KJNode)
  - ✅`del file` : 检测文件或路径是否存在,是否删除文件,运行需输入信号,删除需有写入权限
  - ✅`split path` : 路径切片,输入路径,输出:盘符/路径/文件/扩展名+检测是否是文件
