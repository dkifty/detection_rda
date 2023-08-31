# detection_rda
## start with

```c
git clone https://github.com/dkifty/detection_rda.git
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

```c
conda env create -f env.yaml
```

## Data preprocessing to detection
- Preprocessing annotation made by labelme software

```python
from custom_data_preprocessing import data_preprocessing

# parameters
label2coco = True                # if True - make labelme format annotation to coco format annotation // if False show information of train/valid/test images, annotations for each classes already made
coco2yolo2config = True          # if True - make coco format annotation to yolo format annotation and make yolo config files // if False just check the config files
img_format = 'jpg'               # default is jpg // you can put other format of image files -> string type
label_format = 'json'            # default is json // you can put other format of annotation files -> string type
change_label_name = False        # you can change the label names in annotation files // format(a,b,c,d is str type) : change_label_name = {a:b, c:d}
split_rate = False               # train/valid/test split rate // default is 0 - 0.9*0.8 / 0.9*0.8 - 0.9 / 0.9 - 1 // format(int type in list) : split_rate = [0.7, 0.2, 0.1]
FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test']                # you can change the name of train/valid/test folder name // but dont do that.... please...
FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']  # you can change the name of coco form train/valid/test folder name // but dont do that.... please...
annotation = 'annotations.json'  # default is annotations.json // if annotatino file have other name // annotation = annotations.json (string type)
image_size=(3840,2160)           # if you have other size of image // image_size = (3840, 2160) (default / tuple(int, int))

data_preprocessing(label2coco = label2coco, coco2yolo2config = coco2yolo2config, img_format=img_format, label_format=label_format, change_label_name=change_label_name, split_rate=split_rate, FOLDERS = FOLDERS, FOLDERS_COCO = FOLDERS_COCO, annotation = annotation, image_size=image_size)
