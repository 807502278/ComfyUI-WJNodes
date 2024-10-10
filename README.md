
**ComfyUI-WJNodes explain**
Ready to use upon download. No need to install dependencies for the time being.
If there are new functions or suggestions, please provide feedback.
Attention! The delfile node is not recommended for use on servers. I am not responsible for any losses incurred.

**Node list:**
Image operations
  - ✅`load image from path` : Load image from path
  - ✅`save image to path` : Overwrite and save image via path
  - ✅`save image out` : Save image to output and output the path
  - ✅`select images batch` : Batch selection and re-combination of batches
  - ✅`load image adv` : Load image with masked inversion and path output
  - 🟩Load value feature recognition model (e.g., nsfw, aesthetic score, AI value, time)
  - 🟩Input recognition model and image batch, output batch and corresponding feature values
  - 🟩Sort image batches by specifying an array (e.g., feature value array)
Image editing
  - ✅`adv crop` : Advanced cropping: Can quickly crop/expand/move/flip images. Can output background mask and custom fill.
         (Usage method included in the node. Known bug: When the expansion size is more than 1 time, tiled and mirrored filling cannot be used.)
  - ✅`mask detection` : Mask detection: Detect if there is a mask, detect if it is all hard edges, detect if the mask is pure white/pure black/pure gray and output a value of 0-255.
Others
  - ✅`any_data` : Group any data. Known bug: Nested grouping will split.
Paths
  - ✅`comfyui path` : Output commonly used paths of ComfyUI (root, output/input, plugins, models, cache, Python environment)
  - ✅`path append` : Add prefix and suffix to a string (refer to KJNode)
  - ✅`del file` : Detect if a file or path exists, whether to delete a file. Running requires an input signal. Deleting requires write permission.
  - ✅`split path` : Path slicing. Input path, output: drive letter/path/file/extension + detect if it is a file.

**ComfyUI-WJNodes介绍**
下载即用，暂时无需安装依赖，有新功能或建议请反馈。
注意！delfile节点不建议在服务器上使用，产生任何损失与本人无关

**节点列表**
图片操作
  - ✅`load image from path` : 从路径加载图片
  - ✅`save image to path` : 通过路径覆盖保存图片
  - ✅`save image out` : 保存图片到output并输出该路径
  - ✅`select images batch` : 批次选择和重新组合批次
  - ✅`load image adv` : 带遮罩反转和路径输出的加载图片
  - 🟩加载值特征识别模型(例如nsfw,美学分数,AI值,time)
  - 🟩输入识别模型和图像批次，输出批次和对应特征值
  - 🟩通过指定数组(例如特征值数组)排序图片批次
图像编辑
  - ✅`adv crop` : 高级裁剪:可快速裁剪/扩展/移动/翻转图片,可输出背景遮罩和自定义填充
         (节点内附使用方法,已知bug:扩展尺寸超过1倍时无法使用平铺和镜像填充)
  - ✅`mask detection` : 遮罩检测:检测是否有遮罩,检测是否是全硬边,检测遮罩是否是纯白/纯黑/纯灰并输出值0-255
其它
  - ✅`any_data` : 将任意数据打组，已知bug:嵌套打组会裂开
路径
  - ✅`comfyui path` : 输出comfyui常用路径(根,输出/输入,插件,模型,缓存,python环境)
  - ✅`path append` : 给字符串增加前缀后缀(参考KJNode)
  - ✅`del file` : 检测文件或路径是否存在,是否删除文件,运行需输入信号,删除需有写入权限
  - ✅`split path` : 路径切片,输入路径,输出:盘符/路径/文件/扩展名+检测是否是文件