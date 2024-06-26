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

with open('labels.txt', 'r') as label:
    labels = label.readlines()
label_list = []
for a in labels:
    label_list.append(a.rstrip())
label_list.sort()
label_name = label_list

label_list_check = label_list
label_list_check_ = []
for label_list_check__ in  label_list:
    if label_list_check__ == '__ignore__' or label_list_check__ == '_background_':
        pass
    else:
        label_list_check_.append(label_list_check__)
label_list_check_.sort()

if not os.path.exists('./yolo_configs'):
	os.mkdir('./yolo_configs')

# yolo v8
def yolov8_check():
    try:
        import ultralytics
    except ImportError:
        os.system('pip install ultralytics==8.0.221')

    import ultralytics
    if ultralytics.__version__ == '8.0.221':
        pass
    else:
        os.system('pip install ultralytics==8.0.221')

def yolov8_config():
    import site
    ultralytics_package_utils = os.path.join(site.getsitepackages()[0], 'ultralytics/data/utils.py')
    with open(ultralytics_package_utils, 'r') as f:
        lines = f.readlines()
    lines[33] = "    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels_seg{os.sep}'  # /images/, /labels/ substrings\n"
    with open(ultralytics_package_utils, 'w') as a:
        for line in lines:
            a.write(line)

# yolo v5 config

def yolov5_check():
    if os.path.exists('yolov5'):
        print('Ultralytics... Okay')
        
        sys.path.append('./yolov5')
                               
    else:
        os.system('git clone https://github.com/ultralytics/yolov5')
        sys.path.append('./yolov5')
        
def yolov5_config_make(size=False):
    if not os.path.exists('./yolo_configs/models'):
        os.mkdir('./yolo_configs/models')
    if not os.path.exists('./yolo_configs/data'):
        os.mkdir('./yolo_configs/data')
    raw_model_config = './yolov5/models/yolov5'+size+'.yaml'
    changed_model_config = './yolo_configs/models/'+'custom_yolov5'+size+'.yaml'
    shutil.copy(raw_model_config, changed_model_config)
    
    with open(changed_model_config, 'rb') as d:
        lines = d.readlines()
        
    changed_label_parts = 'nc: ' + str(len(label_list_check_))  + ' # number of classes\n'
    lines[3] = changed_label_parts.encode('utf-8')
    
    with open(changed_model_config, 'wb') as d:
        for line in lines:
            d.write(line)
    print('yolo v5_'+size+'_config file created')

def setting_yolov5_config(size=False, FOLDERS_COCO=['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']):
    
    with open('labels.txt', 'r') as label:
            labels = label.readlines()
            label_list = []
            for a in labels:
                label_list.append(a.rstrip())
            label_list.sort()
    label_name = label_list
    
    for a in FOLDERS_COCO:
        globals()['{}_img_list'.format(a.split('_')[-1])] = glob.glob(os.path.join(a, 'images/*.jpg'))
        globals()['{}_img_list'.format(a.split('_')[-1])].sort()
        with open('./yolo_configs/data/'+a.split('_')[-1]+'.txt', 'w') as b:
            b.write('\n'.join(globals()['{}_img_list'.format(a.split('_')[-1])]) + '\n')
    print('train, valid, test txt file.... created')

    with open('./yolo_configs/data/custom.yaml', 'w') as c:
            c.write('train : .' + str(os.path.join(FOLDERS_COCO[0])) + '/images' + '\n')
            c.write('val : .' + str(os.path.join(FOLDERS_COCO[1])) + '/images' + '\n')
            c.write('test : .' + str(os.path.join(FOLDERS_COCO[2])) + '/images' + '\n')
            c.write('\n')
            c.write(f'nc : {len(label_list_check_)}'+'\n')
            c.write(f'names : {label_list_check_}')
    print('costom.yaml file.... created')

    with open('yolov5/utils/dataloaders.py', 'r') as f:
        lines = f.readlines()
    lines[429] = "    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels_det/ substrings\n"
    with open('yolov5/utils/dataloaders.py', 'w') as b:
        for line in lines:
            b.write(line)

    with open('./yolo_configs/data/custom_v8.yaml', 'w') as c:
        c.write('train : ../../.' + str(os.path.join(FOLDERS_COCO[0])) + '/images' + '\n')
        c.write('val : ../../.' + str(os.path.join(FOLDERS_COCO[1])) + '/images' + '\n')
        c.write('test : ../../.' + str(os.path.join(FOLDERS_COCO[2])) + '/images' + '\n')
        c.write('\n')
        c.write(f'nc : {len(label_list_check_)}'+'\n')
        c.write(f'names : {label_list_check_}')
    print('costom.yaml file.... created')
        
    if size == 'x':
        yolov5_config_make(size='x')
        print('yolo v5 x size --- config file --- complete')
    elif size == 'l':
        yolov5_config_make(size='l')
        print('yolo v5 l size --- config file --- complete')
    elif size == 'm':
        yolov5_config_make(size='m')
        print('yolo v5 m size --- config file --- complete')
    elif size == 's':
        yolov5_config_make(size='s')
        print('yolo v5 s size --- config file --- complete')
    elif size == 'n':
        yolov5_config_make(size='n')
        print('yolo v5 n size --- config file --- complete')
        
    elif size == 'all':
        yolov5_config_make(size='x')
        yolov5_config_make(size='l')
        yolov5_config_make(size='m')
        yolov5_config_make(size='s')
        yolov5_config_make(size='n')
        print('yolo v5 all size --- config file --- complete')
        
    assert size != False, "should fill the size of yolov5"

# darknet yolo v3 v4 config

def darknet_check():
	if os.path.exists('darknet'):
		print('Darknet... Okay')
	else:
		os.system('git clone https://github.com/AlexeyAB/darknet')
		os.mkdir('./darknet/build_release')
		os.chdir('./darknet/build_release')
		os.system('cmake ..')
		os.system('cmake --build . --target install --parallel 8')
		
		os.chdir('./..')
		os.system('sed -i \'s/OPENCV=0/OPENCV=1/\' Makefile')
		os.system('sed -i \'s/GPU=0/GPU=1/\' Makefile')
		os.system('sed -i \'s/CUDNN=0/CUDNN=1/\' Makefile')
		os.system('sed -i \'s/CUDNN_HALF=0/CUDNN_HALF=1/\' Makefile')
		os.system('sed -i \'s/LIBSO=0/LIBSO=1/\' Makefile')
		os.system('make')
		os.chdir('./..')
		
def setting_darknet_config(darknet_yolo_file=False, batch = 16, subdivisions = 8, max_batches = 12000):
    if darknet_yolo_file != False:
        
        with open('./darknet/cfg/' + darknet_yolo_file + '.cfg', 'r') as f:
            lines = f.readlines()
            
        yolo_str_contain = []
        hyp_str_contain_batch = []
        hyp_str_contain_subdivisions = []
        hyp_str_contain_max_batchs = []
        hyp_str_contain_steps = []


        for a,i in enumerate(lines):
            if '[yolo]' in i:
                yolo_str_contain.append(a)    
            elif 'batch=' in i:
                hyp_str_contain_batch.append(a)
            elif 'subdivisions=' in i:
                hyp_str_contain_subdivisions.append(a)
            elif 'max_batches' in i:
                hyp_str_contain_max_batchs.append(a)
            elif 'steps=' in i:
                hyp_str_contain_steps.append(a)
            else:
                pass
            
        for j in yolo_str_contain:
            _before_change = lines[j-10:j+5]
            for b,k in enumerate(_before_change):
                if 'filters' in k:
                    lines[j-10+b] = 'filters='+str((len(label_list_check_)+5)*3)+'\n'
                else:
                    pass
                if 'classes' in k:
                    lines[j-10+b] = 'classes='+str(len(label_list_check_))+'\n'
                else:
                    pass
                
        for l in hyp_str_contain_batch:
            lines[l] = 'batch='+str(batch)+'\n'
            
        for m in hyp_str_contain_subdivisions:
            lines[m] = 'subdivisions='+str(subdivisions)+'\n'
            
        for n in hyp_str_contain_max_batchs:
            lines[n] = 'max_batches = '+str(max_batches)+'\n'
            
        for o in hyp_str_contain_steps:
            lines[o] = 'steps=' + str(int(max_batches*0.8)) + ',' + str(int(max_batches*0.9)) + '\n'
        
        if not os.path.exists('./yolo_configs/models'):
            os.mkdir('./yolo_configs/models')
        if not os.path.exists('./yolo_configs/data'):
            os.mkdir('./yolo_configs/data')
		
        with open('./yolo_configs/models/'+darknet_yolo_file + '.cfg', 'w') as f:
            for line in lines:
                f.write(line)
                
        print(darknet_yolo_file, '--- config file --- complete')
        
    else:
        yolo_files_list = glob.glob('./darknet/cfg/*.cfg')
        yolo_files_list = [asdf for asdf in yolo_files_list if 'yolo' in asdf]
        print('you can use the files : ..... maybe .....')
        for qwer in yolo_files_list:
            print('   ', qwer.split('/')[-1])
    assert darknet_yolo_file != False, 'please fill the parameter : darknet_yolo_file = above things'
    
    darknet_data = []
    darknet_data.append('classes = ' + str(len(label_list_check_)) + '\n')
    darknet_data.append('train = yolo_configs/data/train.txt\n')
    darknet_data.append('valid = yolo_configs/data/valid.txt\n')
    darknet_data.append('names = yolo_configs/data/obj.names\n')
    darknet_data.append('backup = darknet')
        
    darknet_names = []
    for names_len in range(len(label_list_check_)):
        darknet_names.append(label_list_check_[names_len]+'\n')
        
    with open('./yolo_configs/data/obj.data', 'w') as a:
        for line in darknet_data:
            a.write(line)
    print('make obj.data file... complete')
            
    with open('./yolo_configs/data/obj.names', 'w') as b:
        for line in darknet_names:
            b.write(line)
    print('make obj.names file... complete')

DARKNET_MODELS = ['yolov4-tiny-custom', 'yolov4-tiny-3l', 'yolov4-p5', 'yolov2', 'yolov4_new', 'yolov4x-mish', 'yolov4-csp-s-mish', 'yolov3_5l', 'yolov4-csp-x-mish', 'yolov3-spp', 'yolov4-p6', 'yolov4-csp-x-swish-frozen', 'yolov4-tiny', 'yolov3-openimages', 'yolov4-custom', 'yolov4-csp-x-swish', 'Gaussian_yolov3_BDD', 'yolov4-sam-mish-csp-reorg-bfm', 'yolo.2.0', 'yolo', 'yolov4', 'yolov3-tiny_obj', 'yolov2-tiny', 'yolov7-tiny', 'yolov3-tiny_xnor', 'yolov3-tiny_3l', 'yolov4-tiny_contrastive', 'yolov3-tiny', 'tiny-yolo', 'yolov7', 'yolo9000', 'yolov3', 'yolov3-tiny_occlusion_track', 'yolov4-p5-frozen', 'yolov4-csp', 'yolov3-tiny-prn', 'yolov7x', 'yolov4-csp-swish']

def daknet_model_all(size=DARKNET_MODELS, BATCH = 16, subdivisions = 8, max_batches = 12000):
    for darknet_models in DARKNET_MODELS:
        setting_darknet_config(darknet_yolo_file=darknet_models, batch = BATCH, subdivisions = subdivisions, max_batches = max_batches)
    print('darknet config... complete')

def make_yolo_config(batch = 16, subdivisions = 8, max_batches = 12000):
    darknet_check()
    print('')
    yolov5_check()
    print('')
    daknet_model_all(size=DARKNET_MODELS, BATCH = 16, subdivisions = 8, max_batches = max_batches)
    print('')
    setting_yolov5_config(size='all', FOLDERS_COCO=['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test'])
    print('')
    yolov8_check()
    yolov8_config()
