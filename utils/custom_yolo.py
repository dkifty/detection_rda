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


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

########################################################################################################################

with open('labels.txt', 'r') as label:
	labels = label.readlines()
label_list = []
for a in labels:
	label_list.append(a.rstrip())
label_list.sort()
label_name = label_list

def yolov5_check():
    if os.path.exists('yolov5'):
        print('yolov5 files are ready')
        
        sys.path.append('./yolov5')
        
        import val as validate  # for end-of-epoch mAP
        from models.experimental import attempt_load
        from models.yolo import Model
        from utils.autoanchor import check_anchors
        from utils.autobatch import check_train_batch_size
        from utils.callbacks import Callbacks
        from utils.dataloaders import create_dataloader
        from utils.downloads import attempt_download, is_url
        from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                                   check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                                   get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                                   labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                                   yaml_save)
        from utils.loggers import Loggers
        from utils.loggers.comet.comet_utils import check_comet_resume
        from utils.loss import ComputeLoss
        from utils.metrics import fitness
        from utils.plots import plot_evolve
        from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first)
                               
    else:
        os.system('git clone https://github.com/ultralytics/yolov5')
        sys.path.append('./yolov5')
        
        import val as validate  # for end-of-epoch mAP
        from models.experimental import attempt_load
        from models.yolo import Model
        from utils.autoanchor import check_anchors
        from utils.autobatch import check_train_batch_size
        from utils.callbacks import Callbacks
        from utils.dataloaders import create_dataloader
        from utils.downloads import attempt_download, is_url
        from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                                   check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                                   get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                                   labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                                   yaml_save)
        from utils.loggers import Loggers
        from utils.loggers.comet.comet_utils import check_comet_resume
        from utils.loss import ComputeLoss
        from utils.metrics import fitness
        from utils.plots import plot_evolve
        from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first)

def yolov3_4_check():
	if os.path.exists('darknet'):
		print('yolov3, yolov4 files are ready')
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
		

def yolov5_config_make(size=False):
    raw_model_config = './yolov5/models/yolov5'+size+'.yaml'
    changed_model_config = 'custom_yolov5'+size+'.yaml'
    shutil.copy(raw_model_config, changed_model_config)
    
    with open(changed_model_config, 'rb') as d:
        lines = d.readlines()
        
    changed_label_parts = 'nc: ' + str(len(label_name[2:]))  + ' # number of classes\n'
    lines[3] = changed_label_parts.encode('utf-8')
    
    with open(changed_model_config, 'wb') as d:
        for line in lines:
            d.write(line)
    print('yolo v5_'+size+'_config file created')

def setting_yolov3_4_config(darknet_yolo_file=False, batch = 16, subdivisions = 8, max_batches = 12000):
    if darknet_yolo_file != False:
        
        with open('./darknet/cfg/' + darknet_yolo_file, 'r') as f:
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
                    lines[j-10+b] = 'filters='+str((len(label_name[2:])+5)*3)+'\n'
                    print(lines[j-10+b])
                else:
                    pass
                if 'classes' in k:
                    lines[j-10+b] = 'classes='+str(len(label_name[2:]))+'\n'
                    print(lines[j-10+b])
                else:
                    pass
                
        for l in hyp_str_contain_batch:
            lines[l] = 'batch='+str(batch)+'\n'
            print(lines[l])
            
        for m in hyp_str_contain_subdivisions:
            lines[m] = 'subdivisions='+str(subdivisions)+'\n'
            print(lines[m])
            
        for n in hyp_str_contain_max_batchs:
            lines[n] = 'max_batches = '+str(max_batches)+'\n'
            print(lines[n])
            
        for o in hyp_str_contain_steps:
            lines[o] = 'steps=' + str(int(max_batches*0.8)) + ',' + str(int(max_batches*0.9)) + '\n'
            print(lines[o])
            
        with open('./'+darknet_yolo_file, 'w') as f:
            for line in lines:
                f.write(line)
    else:
        yolo_files_list = glob.glob('./darknet/cfg/*.cfg')
        yolo_files_list = [asdf for asdf in yolo_files_list if 'yolo' in asdf]
        print('you can use the files : ..... maybe .....')
        for qwer in yolo_files_list:
            print('   ', qwer.split('/')[-1])
    assert darknet_yolo_file != False, 'please fill the parameter : darknet_yolo_file = above things'

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
        with open(a.split('_')[-1]+'.txt', 'w') as b:
            b.write('\n'.join(globals()['{}_img_list'.format(a.split('_')[-1])]) + '\n')
    
    with open('custom.yaml', 'w') as c:
            c.write('train : .' + f'{os.path.join(FOLDERS_COCO[0])}' + '/images' + '\n')
            c.write('val : .' + f'{os.path.join(FOLDERS_COCO[1])}' + '/images' + '\n')
            c.write('test : .' + f'{os.path.join(FOLDERS_COCO[2])}' + '/images' + '\n')
            c.write('\n')
            c.write(f'nc : {len(label_name[2:])}'+'\n')
            c.write(f'names : {label_name[2:]}')
        
    if size == 'x':
        yolov5_config_make(size='x')
    elif size == 'l':
        yolov5_config_make(size='l')
    elif size == 'm':
        yolov5_config_make(size='m')
    elif size == 's':
        yolov5_config_make(size='s')
    elif size == 'n':
        yolov5_config_make(size='n')
    elif size == 'all':
        yolov5_config_make(size='x')
        yolov5_config_make(size='l')
        yolov5_config_make(size='m')
        yolov5_config_make(size='s')
        yolov5_config_make(size='n')
        
    assert size != False, "should fill the size of yolov5"
        
# yolo 3 4 만들기
        
def run_yolo(size=False, imagesize=1024, batch=16, epoch=200):
    yolov5_check()
        
    os.system('cd ./yolov5')
    if os.path.exists('./train.py'):
        print('found yolo v5 run file \'train.py\'')
    
    #train_command = 'python train.py --img imagesize --batch batchsize --epochs epoch --data ../custom.yaml --cfg '../custom_yolov5'+size --weights '' --name 'custom_results_yolo_v5_'+size --cache'
    #python train.py --img imagesize --batch batchsize --epochs epoch --data '../custom.yaml' --cfg '../custom_yolov5'+size --weights '' --name 'custom_results_yolo_v5_'+size --cache
    
    #%cd ./..
