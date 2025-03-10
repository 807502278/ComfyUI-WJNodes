
**ComfyUI-WJNodes explain**

- Ready to use upon download. No need to install dependencies for the time being.\
  If you want to use modified versions of other plugins, you need to install their corresponding dependencies
- If there are new functions or suggestions, please provide feedback.
- Attention! The delfile node is not recommended for use on servers. I am not responsible for any losses incurred.
- To disable the delfile node, change the 'DelFile=True' of node/file.py to 'DelFile=False'

**Node list:**

- Image: WJNode/Image
  - ✅`load image from path` : Load image from path
  - ✅`save image to path` : Save image by overwriting the path
  - ✅`save image out` : Save image to output and output the path
  - ✅`select images batch` : Batch selection and recombination of batches,updated mask support in 20250115
  - ✅`select images batch` : 20250115 Batch Selection and Recombination of More Functions
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
  - ✅`mask_and_mask_math` : Mask to mask operations, supports addition/subtraction/intersection/multiplication operations, \
                  Adjustable cv2 and torch modes, if cv2 is not installed, automatically switches to torch
  - 🟩`Accurate_mask_clipping` : Precise search for mask bbox boundaries (under development)
- Image Editing: WJNode/ImageEdit
  - ✅`adv crop` : Advanced cropping: can quickly crop/expand/move/flip images, can output background masks and custom filling \
                  (Usage method included in the node, known bug: expansion size more than 1 times cannot use tiling and mirror filling)
  - ✅`mask detection` : Mask detection: detect whether there is a mask, detect whether it is all hard edges, \
                  detect whether the mask is pure white/pure black/pure gray and output values 0-255
  - ✅`InvertChannelAdv` : Invert/separate image channels Image ⭐\
                  RGBA to mask batch Replace channels \
                  Any channel to RGBA
  - ✅`Bilateral Filter` : Image/Mask Bilateral Filtering: Can repair layered distortion caused by color or brightness scaling in images
  - ✅`any_math` : Any data calculation, supports pure data input such as images/values/arrays, and outputs images or any data type
  - ✅`any_math_v2` : Support arbitrary data calculation with more inputs and 3 sets of outputs
  - ✅`Image_ValueMath` : Image and image calculation, optional cv2 or torch mode
-Video Editor: WJNode/Video
  - ✅`Video_fade` : Two video segments can choose two ways to fade in and out, \
                  Mask: Local fade in and out under development... \
                  Exponential: Exponential gradient under development...
- Others: WJNode/Other-functions
  - ✅`any_data` : Group any data, known bug: nested grouping will split
  - ✅`show_type` : Display data type
  - ✅` array_count` :  20250109 Change the original array_element_comunt node to array_count\
                        Retrieve data shape (array format), count the number of elements at a specified depth, \
                            count the number of all elements, and count image data\
                        If changes to this node cause your workflow to fail to run, please notify me
  - ✅` get image data` :  20250109 Obtain basic data from images/masks (batch/width/height/maximum value)
- Detection: WJNode/Detection
  - ✅`load_similarity_model` : Load pre-trained image similarity models (ResNet, DenseNet, etc.) for feature extraction
  - ✅`image_similarity` : Calculate similarity between images using loaded models and various distance metrics
- Hardware: WJNode/Other-node
  - ✅`Graphics_Detection_Reference` : Test GPU computing capabilities including hardware info, precision tests, memory bandwidth, \
                  operator performance, and AI benchmarks with RTX 4090 comparison
- Plugins: WJNode/Other-plugins(To use the following nodes, you must install the following plugins)
  - ✅`WAS_Mask_Fill_Region_batch` : Optimize WAS plugin's WAS_Mask_Fill_Region (mask cleanup) to support batches\
  [Thanks to @WASasquatch](https://github.com/WASasquatch/was-node-suite-comfyui)
  - ✅`SegmDetectorCombined_batch` : Optimize impack-pack plugin's SegmDetectorCombined (segm detection mask) to support batches\
  [Thanks to @ltdrdata](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
  - ✅`bbox_restore_mask` : Add impack-pack plugin's seg decomposition, restore cropped images through cropping data (SEG editing)
  - ✅`Sam2AutoSegmentation_data` : Add Sam2AutoSegmentation (kijia) node's color list/coordinate output, used to assist SAM2 video keying\
  [Thanks to @kijai](https://github.com/kijai/ComfyUI-segment-anything-2)
  - ✅`ApplyEasyOCR batch` : Modify OCR recognition nodes to load models separately for faster operation and model caching\
  [Thanks to @prodogape](https://github.com/prodogape/ComfyUI-EasyOCR)
  - ✅`load EasyOCR model` : load OCR models.
- Path: WJNode/Path
  - ✅`comfyui path` : Output comfyui common paths (root, output/input, plugins, models, cache, Python environment)
  - ✅`path append` : Add prefix/suffix to strings (reference KJNode)
  - ✅`del file` : Detect whether file or path exists, whether to delete file, operation requires input signal, deletion requires write permission
  - ✅`split path` : Path slicing, input path, output: disk symbol/path/file/extension + detect whether it is a file


## models dir: （Not required）
These models are in the same path as the original plugin and do not need to be downloaded repeatedly
```
models
    ├──torchvision
    │   └──resnet
    │       ├──resnet50-11ad3fa6.pth
    │       └──...
    ├──EasyOCR
    │   ├──craft_mlt_25k.pth
    │   ├──latin_g2.pth
    │   └──zh_sim_g2.pth
    └──sam2
        ├──sam2_hiera_small.safetensors
        └──...
```