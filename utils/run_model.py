#!/usr/bin/env python

import os
import glob
import shutil


def run_model(model=None, resize_img=1024, batch=16, epochs=200, FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']):
    if 'yolov5' in model:
        yolov5_run = 'python3 yolov5/train.py --img ' + str(resize_img) + ' --batch ' + str(batch) + ' --epochs ' + str(epochs) + ' --data yolo_configs/data/custom.yaml --cfg yolo_configs/models/custom_' + model + '.yaml --name custom_results_' + model + '.yaml --cache'
        os.system(yolov5_run)
        
    elif 'yolov4' in model:
        if not os.path.exists('yolo_configs/models/yolov4.conv.137'):
            os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137')
            shutil.move('yolov4.conv.137', 'yolo_configs/models/yolov4.conv.137')
        else:
            print('model pre-weights already exsits\n')
        
        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                shutil.copy(yolo_annotations, yolo_annotations.replace('labels', 'images'))
        
        yolov4_run = './darknet/darknet detector train yolo_configs/data/obj.data yolo_configs/models/' + model + '.cfg yolo_configs/models/yolov4.conv.137 -dont_show -map'
        os.system(yolov4_run)
        
        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                os.remove(yolo_annotations.replace('labels', 'images'))
        
    elif 'yolov3' in model:
        if not os.path.exists('yolo_configs/models/darknet53.conv.74'):
            os.system('wget https://pjreddie.com/media/files/darknet53.conv.74')
            shutil.move('darknet53.conv.74','yolo_configs/models/darknet53.conv.74')
        else:
            print('model pre-weights already exsits\n')
        
        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                shutil.copy(yolo_annotations, yolo_annotations.replace('labels', 'images'))
            
        yolov3_run = './darknet/darknet detector train yolo_configs/data/obj.data yolo_configs/models/' + model + '.cfg yolo_configs/models/darknet53.conv.74 -dont_show -map'
        os.system(yolov3_run)

        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                os.remove(yolo_annotations.replace('labels', 'images'))
