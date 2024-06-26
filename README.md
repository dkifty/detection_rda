# detection_rda
## start with

```c
git clone https://github.com/dkifty/detection_rda.git
cd detection_rda

conda env create -f env.yaml

conda activate detection_rda
pip install ipykernel
python -m ipykernel install --user --name detection_rda --display-name detection_rda
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

jupyter lab
```
 
```python

import sys
sys.path.append('./utils')

```

## video to frame

```python

from vid2frame import v2f
v2f(folder_name, fomatting, frame)

# folder_name = Folder name that contain the videos ex) '230831' or '230831/1' -> str
# fomatting = format of video ex) 'MP4' -> str
# frame = interval that you want to slice for frames ex) 30 -> int
# ex) v2f('230831', 'MP4', 60)
```

## Setting environment
- If you want to run detection model ... 
- This model runs on conda env
- cuda 11.3
- cudnn 8.2.1
- torch 1.10 / torchvision 0.11  torchaudio 0.10

## Data preprocessing to detection
- Preprocessing annotation made by labelme software

1. __first__
- set parameters
```python
# parameters
label2coco = True                # if True - make labelme format annotation to coco format annotation // if False show information of train/valid/test images, annotations for each classes already made
coco2yolo = True          # if True - make coco format annotation to yolo format annotation and make yolo config files // if False just check the config files
img_format = 'jpg'               # default is jpg // you can put other format of image files -> string type
label_format = 'json'            # default is json // you can put other format of annotation files -> string type
change_label_name = False        # you can change the label names in annotation files // format(a,b,c,d is str type) : change_label_name = {a:b, c:d}
split_rate = False               # train/valid/test split rate // default is 0 - 0.9*0.8 / 0.9*0.8 - 0.9 / 0.9 - 1 // format(int type in list) : split_rate = [0.7, 0.2, 0.1]
FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test']                # you can change the name of train/valid/test folder name // but dont do that.... please...
FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']  # you can change the name of coco form train/valid/test folder name // but dont do that.... please...
annotation = 'annotations.json'  # default is annotations.json // if annotatino file have other name // annotation = annotations.json (string type)
image_size=(3840,2160)           # if you have other size of image // image_size = (3840, 2160) (default / tuple(int, int))

# model
batch = 16
subdivisions = 8
max_batches = 12000
epochs=200
resize_img=1024
NUM_WORKERS = 4
IMS_PER_BATCH = 8
ITER = 12000

# device
device = 0                       # device = 0 or 1 or 2 or 1,2 or cpu
```

2. __second__
- change the file 'label.txt'
```python
from custom_data_preprocessing import label_name_check
label_name_check(img_format='jpg', label_format='json')

from custom_data_preprocessing import make_label_file
make_label_file('Color_checker', 'Flower', 'Fruit_ripen', 'Fruit_unripen', 'Obstacle', 'Old_leaves', 'Picking_point', 'Runner', 'Unidentified')
# 굳이 오름차순으로 안해도 되게 해놓기는 함... labels.txt파일 생성하는 코드
```
- you can get the labels in annotation files -> put in to labels.txt after ignore and background

3. __third__
- run preprocessing

```python
from custom_data_preprocessing import data_preprocessing

# run
data_preprocessing(label2coco = label2coco, coco2yolo = coco2yolo, img_format=img_format, label_format=label_format, change_label_name=change_label_name, split_rate=split_rate, FOLDERS = FOLDERS, FOLDERS_COCO = FOLDERS_COCO, annotation = annotation, image_size=image_size)
```

4. __4th__
- set all configs

```python
from make_configs import make_configs
make_configs(yolo=True, detectron=True, yolact=True, batch = batch, subdivisions = subdivisions, max_batches = max_batches, resize_img = resize_img)
```

- you can get the information
  - count of image, annotaion file
  - labels name in annotation file
  - count of datasets for train / valid / test
  - and count of labels for each
  - coco and yolo form annotation
 
## models...

```python
------------darknet------------
yolov4-tiny-custom
yolov4-tiny-3l
yolov4-p5
yolov4_new
yolov4x-mish
yolov4-csp-s-mish
yolov3_5l
yolov4-csp-x-mish
yolov3-spp
yolov4-p6
yolov4-csp-x-swish-frozen
yolov4-tiny
yolov3-openimages
yolov4-custom
yolov4-csp-x-swish
yolov4-sam-mish-csp-reorg-bfm
yolov4
yolov3-tiny_obj
yolov2-tiny
yolov7-tiny
yolov3-tiny_xnor
yolov3-tiny_3l
yolov4-tiny_contrastive
yolov3-tiny
yolov3
yolov3-tiny_occlusion_track
yolov4-p5-frozen
yolov4-csp
yolov3-tiny-prn
yolov4-csp-swish

------------ultralytics------------
yolov5s
yolov5n
yolov5m
yolov5l
yolov5x

yolov8s
yolov8n
yolov8m
yolov8l
yolov8x
------------detectron2------------
fasterrcnn_R50_C4_1x
fasterrcnn_R50_DC5_1x
fasterrcnn_R50_FPN_1x
fasterrcnn_R50_C4_3x
fasterrcnn_R50_DC5_3x
fasterrcnn_R50_FPN_3x

fasterrcnn_R101_C4_3x
fasterrcnn_R101_DC5_3x
fasterrcnn_R101_FPN_3x

fasterrcnn_X101_FPN_3x

retinanet_R50_FPN_1x
retinanet_R50_FPN_3x
retinanet_R101_FPN_3x

rpn_R50_C4_1x
rpn_R50_FPN_1x

maskrcnn_R50_C4_1x
maskrcnn_R50_DC5_1x
maskrcnn_R50_FPN_1x
maskrcnn_R50_C4_3x
maskrcnn_R50_DC5_3x
maskrcnn_R50_FPN_3x

maskrcnn_R101_C4_3x
maskrcnn_R101_DC5_3x
maskrcnn_R101_FPN_3x

maskrcnn_X101_FPN_3x

------------YOLACT------------
yolact_darknet53
yolact_resnet50
yolact_resnet101
yolact_plus_resnet50
yolact_plus_resnet101
```

## Detection models run
1. __YOLO V3__
```python
from run_model import run_model
run_model(model='yolov3', train=True, val = True, test = True, iou = 0.5) # you can get model parameter include the word 'yolov3'
```

2. __YOLO V4__
```python
from run_model import run_model
run_model(model='yolov4', train=True, val = True, test = True, iou = 0.5) # you can get model parameter include the word 'yolov4'
```

3. __YOLO V5__
```python
from run_model import run_model
run_model(model='yolov5s', train=True, val = True, test = True, iou = 0.5, resize_img=resize_img, batch=batch, epochs=epochs) # you can get model parameter 'yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x'
```

4. __YOLO V8__
```python
from run_model import run_model
run_model(model='yolov8s', train=True, val = True, test = True, iou = 0.5, resize_img=resize_img, batch=batch, epochs=epochs) # you can get model parameter 'yolov8s', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov8x'
```

5. __Faster RCNN__
```python
from run_model import run_model
run_model(model='fasterrcnn_R50_C4_1x', train=True, val = True, test = True, iou = 0.5, NUM_WORKERS = NUM_WORKERS, IMS_PER_BATCH = IMS_PER_BATCH, ITER = ITER) # you can get model parameter 'fasterrcnn_R50_C4_1x', 'fasterrcnn_R50_DC5_1x', 'fasterrcnn_R50_FPN_1x', 'fasterrcnn_R50_C4_3x', 'fasterrcnn_R50_FPN_3x', 'fasterrcnn_R101_C4_3x', 'fasterrcnn_R50_DC5_3x', 'fasterrcnn_R101_DC5_3x', 'fasterrcnn_R101_FPN_3x', 'fasterrcnn_X101_FPN_3x'
```

6. __RetinaNet__
```python
from run_model import run_model
run_model(model='retinanet_R50_FPN_1x', train=True, val = True, test = True, iou = 0.5, NUM_WORKERS = NUM_WORKERS, IMS_PER_BATCH = IMS_PER_BATCH, ITER = ITER) # you can get model parameter 'retinanet_R50_FPN_1x', 'retinanet_R50_FPN_3x', 'retinanet_R101_FPN_3x'
```

7. __others__
```python
from run_model import run_model
run_model(model='rpn_R50_C4_1x', train=True, val = True, test = True, iou = 0.5, NUM_WORKERS = NUM_WORKERS, IMS_PER_BATCH = IMS_PER_BATCH, ITER = ITER) # you can get model parameter 'rpn_R50_C4_1x', 'rpn_R50_FPN_1x'
```

## Segmentation models run

1. __Mask RCNN__
```python
from run_model import run_model
run_model(model='maskrcnn_X101_FPN_3x', train=True, val = True, test = True, iou = 0.5, NUM_WORKERS = NUM_WORKERS, IMS_PER_BATCH = IMS_PER_BATCH, ITER = ITER) # you can get model parameter 'maskrcnn_R50_C4_1x', 'maskrcnn_R50_DC5_1x', 'maskrcnn_R50_FPN_1x', 'maskrcnn_R50_C4_3x', 'maskrcnn_R50_DC5_3x', 'maskrcnn_R50_FPN_3x', 'maskrcnn_R101_C4_3x', 'maskrcnn_R101_DC5_3x', 'maskrcnn_R101_FPN_3x', 'maskrcnn_X101_FPN_3x'
```

2. __YOLACT__
```python
from run_model import run_model
run_model(model='yolact_resnet50', train=True, val = True, test = True, iou = 0.5, batch=batch) # you can get model parameter 'yolact_darknet53', 'yolact_resnet50', 'yolact_resnet101', 'yolact_plus_resnet50', 'yolact_plus_resnet101'
```
