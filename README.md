
**ComfyUI-WJNodes**
- This is a simple node package that I use myself. If there are new functions or suggestions, please provide feedback.
- If you want to use modified versions of other nodes, you need to install their corresponding dependencies

**Node list:**

- ImageFile: WJNode/ImageFile
  - ✅`Load_Image_From_Path` : Load image from path
  - ✅`Save_Image_To_Path` : Save image by overwriting the path
  - ✅`Save_Image_Out` : Save image to output and output the path
  - ✅`Load_Image_Adv` : Load image with mask inversion and path output, supports multiple formats (jpg, png, jpeg, webp, tiff, bmp, gif, ico, svg)
  - ✅`image_url_download` : Download images from URLs with configurable timeout, supports batch processing
- ImageEdit: WJNode/ImageEdit
  - ✅`invert_channel_adv` : Invert/separate image channels, RGBA to mask batch, replace channels, any channel to RGBA
  - ✅`ListMerger` : Merge multiple image lists into a single batch
  - ✅`Bilateral_Filter` : Image/Mask Bilateral Filtering: Can repair layered distortion caused by color or brightness scaling
  - ✅`image_math` : Image mathematical operations with expression support
  - ✅`image_math_value` : Image and value mathematical calculations
  - ✅`Robust_Imager_Merge` : Advanced image merging with robust handling
  - ✅`image_scale_pixel_v2` : Advanced image scaling by total pixels with alignment, cropping, and bbox filling options
  - ✅`image_scale_pixel_option` : Generate advanced options for image_scale_pixel_v2 node
- Crop: WJNode/ImageEdit/image_crop
  - ✅`adv_crop` : Advanced cropping: can quickly crop/expand/move/flip images, can output background masks and custom filling
  - ✅`Accurate_mask_clipping` : Accurately find mask boundaries and optionally crop to those boundaries
  - ✅`crop_by_bboxs` : Crop images using bounding box data
- Mask Crop: WJNode/ImageEdit/mask_crop
  - ✅`mask_crop_square` : Square cropping based on mask data
  - ✅`mask_crop_option_SmoothCrop` : Smooth cropping with advanced options
  - ✅`mask_crop_option_Basic` : Basic mask cropping options
  - ✅`crop_data_edit` : Edit and modify crop data
  - ✅`crop_data_CoordinateSmooth` : Coordinate smoothing for crop data
- Mask Editing: WJNode/ImageEdit/MaskEdit
  - ✅`mask_select_mask` : Mask selection within a mask batch (intersection represents selection)
  - 🟩`coords_select_mask` : Coordinate selection of masks, used to assist SAM2 video keying (under development)
  - ✅`mask_line_mapping` : Mask line mapping, can automatically calculate maximum and minimum values when input is -1 or 256
  - ✅`mask_and_mask_math` : Mask to mask operations, supports addition/subtraction/intersection/multiplication operations, \
                  Adjustable cv2 and torch modes, if cv2 is not installed, automatically switches to torch
- Math: WJNode/Math
  - ✅`any_math` : Any data calculation, supports pure data input such as images/values/arrays, and outputs images or any data type
  - ✅`any_math_v2` : Support arbitrary data calculation with more inputs and 3 sets of outputs
- Batch: WJNode/Batch
  - ✅`Select_Images_Batch` : Batch selection and recombination of images/masks with index support
  - ✅`Select_Batch_v2` : Advanced batch selection with loop, limit, and processing options
  - ✅`SelectBatch_paragraph` : Paragraph-based batch selection
  - ✅`Batch_Average` : Average cutting of image/mask batches with division and completion options
- Color: WJNode/Color
  - ✅`load_color_config` : Load color configuration for color block to mask, supports ADE20K preprocessing color data
  - ✅`color_segmentation` : Color block to mask conversion, supports ADE20K and SAM2 data preprocessing
  - ✅`color_segmentation_v2` : Enhanced color block to mask v2, uses keys in color configuration to select masks
  - ✅`filter_DensePose_color` : Filter DensePose color data
  - ✅`load_ColorName_config` : Load color name configuration
  - ✅`Color_check_Name` : Check color names and filter color data
  - ✅`Color_Data_Break` : Break down color data into components
- Video Merge: WJNode/video/merge
  - ✅`Video_fade` : Two video segments can choose two ways to fade in and out
  - ✅`SaveMP4` : Save single video as MP4 format
  - ✅`SaveMP4_batch` : Save video batch as MP4 format
  - ✅`Video_MaskBasedSplit` : Split video based on mask data
  - ✅`Detecting_videos_mask` : Detect masks in video sequences
  - ✅`Cutting_video` : Cut video sequences based on segment data
  - ✅`Video_OverlappingSeparation_test` : Test overlapping separation in videos
- GetData: WJNode/GetData
  - ✅`Mask_Detection` : Mask detection: detect whether there is a mask, detect whether it is all hard edges, \
                  detect whether the mask is pure white/pure black/pure gray and output values 0-255
  - ✅`get_image_data` : Obtain image size data from images/masks (batch/width/height/channels/shape)
  - ✅`get_image_ratio` : Obtain image aspect ratio data (max/min dimensions, ratio float/string, ratio classification)
- Other Functions: WJNode/Other-functions
  - ✅`Any_Pipe` : Group any data, known bug: nested grouping will split
  - ✅`Determine_Type` : Display data type and determine data characteristics
- Other: WJNode/Other
  - 🟩Load value feature recognition model (e.g., nsfw, aesthetic score, AI value, time)
  - 🟩Input recognition model and image batch, output batch and corresponding feature values
  - 🟩Sort image batches through specified arrays (e.g., feature value arrays)
- Detection: WJNode/Other-plugins/Detection
  - ✅`load_torchvision_model` : Load pre-trained torchvision models (ResNet, DenseNet, etc.) for feature extraction
  - ✅`Run_torchvision_model` : Calculate similarity between images using loaded models and various distance metrics
- Hardware: WJNode/Other-node
  - ✅`Graphics_Detection_Reference` : Test GPU computing capabilities including hardware info, precision tests, memory bandwidth, \
                  operator performance, and AI benchmarks with RTX 4090 comparison
- WAS Plugins: WJNode/Other-plugins/WAS (To use the following nodes, you must install WAS plugin)
  - ✅`WAS_Mask_Fill_Region_batch` : Optimize WAS plugin's WAS_Mask_Fill_Region (mask cleanup) to support batches\
  [Thanks to @WASasquatch](https://github.com/WASasquatch/was-node-suite-comfyui)
- Impact Pack Plugins: WJNode/Other-plugins (To use the following nodes, you must install Impact Pack plugin)
  - ✅`SegmDetectorCombined_batch` : Optimize impact-pack plugin's SegmDetectorCombined (segm detection mask) to support batches\
  [Thanks to @ltdrdata](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
  - ✅`bbox_restore_mask` : Add impact-pack plugin's seg decomposition, restore cropped images through cropping data (SEG editing)
  - ✅`Sam2AutoSegmentation_data` : Add Sam2AutoSegmentation (kijai) node's color list/coordinate output, used to assist SAM2 video keying\
  [Thanks to @kijai](https://github.com/kijai/ComfyUI-segment-anything-2)
  - ✅`run_yolo_bboxs` : Run YOLO detection and return bounding boxes
  - ✅`run_yolo_bboxs_v2` : Enhanced YOLO detection with additional features
- EasyOCR Plugins: WJNode/Other-plugins/EasyOCR (To use the following nodes, you must install EasyOCR)
  - ✅`load_EasyOCR_model` : Load OCR models separately for faster operation and model caching
  - ✅`ApplyEasyOCR_batch` : Modify OCR recognition nodes to support batch processing\
  [Thanks to @prodogape](https://github.com/prodogape/ComfyUI-EasyOCR)
- Path: WJNode/Path
  - ✅`ComfyUI_Path_Out` : Output ComfyUI common paths (root, output/input, plugins, models, cache, Python environment)
  - ✅`Str_Append` : Add prefix/suffix to strings (reference KJNode)
  - ✅`del_file` : Detect whether file or path exists, whether to delete file, operation requires input signal, deletion requires write permission
  - ✅`Split_Path` : Path slicing, input path, output: disk symbol/path/file/extension + detect whether it is a file
  - ✅`Folder_Operations_CH` : Folder operations with Chinese support


## Models Directory (Optional)
These models are automatically downloaded to ComfyUI's models directory and shared with other plugins:
```
models/
├── torchvision/              # Torchvision models for similarity detection
│   └── resnet/
│       ├── resnet50-11ad3fa6.pth
│       └── ...
├── EasyOCR/                  # OCR models
│   ├── craft_mlt_25k.pth
│   ├── latin_g2.pth
│   └── zh_sim_g2.pth
└── sam2/                     # SAM2 models (if using SAM2 features)
    ├── sam2_hiera_small.safetensors
    └── ...
```

## Installation

1. Clone or download this repository to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/807502278/ComfyUI-WJNodes.git
```

2. Install dependencies (optional, most are already included in ComfyUI):
```bash
cd ComfyUI-WJNodes
pip install -r requirements.txt
```