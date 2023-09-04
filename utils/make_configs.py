#!/usr/bin/env python

import argparse
import math
import os
import shutil, glob
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline

import sys
sys.path.append('./utils')
from make_yolo_config import *
from set_detectron2 import *
from set_yolact import *
from make_yolact_config import *


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None
sys.path.append('./yolov5')
########################################################################################################################

def make_configs(yolo = True, detectron = True, yolact = True, batch = 16, subdivisions = 8, max_batches = 12000, resize_img=1024):
    
    if yolo == True:
        make_yolo_config(batch = batch, subdivisions = subdivisions, max_batches = max_batches)
    if detectron == True:
    	detectron2_check()
    if yolact == True:
    	yolact_check()
    	make_yolact_config(resize_img=resize_img)
    
    print('')
    print('you can use the models...')
    print('')
    print('------------darknet------------')
    print('yolov4-tiny-custom')
    print('yolov4-tiny-3l')
    print('yolov4-p5')
    print('yolov4_new')
    print('yolov4x-mish')
    print('yolov4-csp-s-mish')
    print('yolov3_5l')
    print('yolov4-csp-x-mish')
    print('yolov3-spp')
    print('yolov4-p6')
    print('yolov4-csp-x-swish-frozen')
    print('yolov4-tiny')
    print('yolov3-openimages')
    print('yolov4-custom')
    print('yolov4-csp-x-swish')
    print('yolov4-sam-mish-csp-reorg-bfm')
    print('yolov4')
    print('yolov3-tiny_obj')
    print('yolov2-tiny')
    print('yolov7-tiny')
    print('yolov3-tiny_xnor')
    print('yolov3-tiny_3l')
    print('yolov4-tiny_contrastive')
    print('yolov3-tiny')
    print('yolov3')
    print('yolov3-tiny_occlusion_track')
    print('yolov4-p5-frozen')
    print('yolov4-csp')
    print('yolov3-tiny-prn')
    print('yolov4-csp-swish')
    print('')
    print('------------ultralytics------------')
    print('yolov5s')
    print('yolov5n')
    print('yolov5m')
    print('yolov5l')
    print('yolov5x')
    print('')
    print('------------detectron2------------')
    print('fasterrcnn_R50_C4_1x')
    print('fasterrcnn_R50_DC5_1x')
    print('fasterrcnn_R50_FPN_1x')
    print('fasterrcnn_R50_C4_3x')
    print('fasterrcnn_R50_DC5_3x')
    print('fasterrcnn_R50_FPN_3x')
    print('')
    print('fasterrcnn_R101_C4_3x')
    print('fasterrcnn_R101_DC5_3x')
    print('fasterrcnn_R101_FPN_3x')
    print('')
    print('fasterrcnn_X101_FPN_3x')
    print('')
    print('retinanet_R50_FPN_1x')
    print('retinanet_R50_FPN_3x')
    print('retinanet_R101_FPN_3x')
    print('')
    print('rpn_R50_C4_1x')
    print('rpn_R50_FPN_1x')
    print('')
    print('maskrcnn_R50_C4_1x')
    print('maskrcnn_R50_DC5_1x')
    print('maskrcnn_R50_FPN_1x')
    print('maskrcnn_R50_C4_3x')
    print('maskrcnn_R50_DC5_3x')
    print('maskrcnn_R50_FPN_3x')
    print('')
    print('maskrcnn_R101_C4_3x')
    print('maskrcnn_R101_DC5_3x')
    print('maskrcnn_R101_FPN_3x')
    print('')
    print('maskrcnn_X101_FPN_3x')
    print('')
    print('yolact_darknet53')
    print('yolact_resnet50')
    print('yolact_resnet101')
    print('yolact_plus_resnet50')
    print('yolact_plus_resnet101')
    print('-------------------------------------------')
