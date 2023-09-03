#!/usr/bin/env python

import os
import glob
import shutil
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

with open('labels.txt', 'r') as label:
	labels = label.readlines()
label_list = []
for a in labels:
	label_list.append(a.rstrip())
label_list.sort()
label_name = label_list

def detectron_configs(model = False, NUM_WORKERS = 2, IMS_PER_BATCH = 2, ITER = 10000):
    global cfg
         
    dataset_name = 'data_dataset_coco_train'
    
    if not dataset_name in DatasetCatalog.list():
        for d in ["data_dataset_coco_train", "data_dataset_coco_test"]:
            register_coco_instances(f"{d}", {}, f"{d}/annotations.json", f"{d}/images")
        metadata = MetadataCatalog.get("data_dataset_coco_train")

    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("data_dataset_coco_train",) ## 위에서 등록했던 dataset name_train 을 적어주기
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = ITER
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_name) - 1
    cfg.MODEL.RETINANET.NUM_CLASSES = len(label_name) - 1
    
    if 'fasterrcnn' in model:
        backbone_1 = model.split('_')[0]
        backbone_2 = model.split('_')[1]
        backbone_3 = model.split('_')[2]
        schedule  = model.split('_')[3]
        
        _1 = backbone_1[:-4]
        _2 = backbone_1[-4:]
        config_path = 'COCO-Detection/'+_1+'_'+_2+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
    elif 'fastrcnn' in model:
        backbone_1 = model.split('_')[0]
        backbone_2 = model.split('_')[1]
        backbone_3 = model.split('_')[2]
        schedule  = model.split('_')[3]
        _1 = backbone_1[:-4]
        _2 = backbone_1[-4:]
        config_path = 'COCO-Detection/'+_1+'_'+_2+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
    elif 'retinanet' in model:
        backbone_1 = model.split('_')[0]
        backbone_2 = model.split('_')[1]
        backbone_3 = model.split('_')[2]
        schedule  = model.split('_')[3]
        config_path = 'COCO-Detection/'+backbone_1+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        
    elif 'rpn' in model:
        backbone_1 = model.split('_')[0]
        backbone_2 = model.split('_')[1]
        backbone_3 = model.split('_')[2]
        schedule  = model.split('_')[3]
        config_path = 'COCO-Detection/'+backbone_1+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
    elif 'maskrcnn' in model:
        backbone_1 = model.split('_')[0]
        backbone_2 = model.split('_')[1]
        backbone_3 = model.split('_')[2]
        schedule  = model.split('_')[3]
        _1 = backbone_1[:-4]
        _2 = backbone_1[-4:]
        config_path = 'COCO-InstanceSegmentation/'+_1+'_'+_2+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
   

def run_model(model=None, resize_img=1024, batch=16, epochs=200, FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test'], NUM_WORKERS = 2, IMS_PER_BATCH = 2, ITER = 12000):
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
    
    elif 'fasterrcnn' or 'fastrcnn' or 'retinanet' or 'rpn' or 'maskrcnn' in model:
        dataset_name = 'data_dataset_coco_train'
        
        if not dataset_name in DatasetCatalog.list():
            for d in ["data_dataset_coco_train", "data_dataset_coco_valid"]:
                register_coco_instances(f"{d}", {}, f"{d}/annotations.json", f"{d}")
            metadata = MetadataCatalog.get("data_dataset_coco_train")
            
        cfg = get_cfg()    
  
        if 'fasterrcnn' in model:
            backbone_1 = model.split('_')[0]
            backbone_2 = model.split('_')[1]
            backbone_3 = model.split('_')[2]
            schedule  = model.split('_')[3]
                
            _1 = backbone_1[:-4]
            _2 = backbone_1[-4:]
            config_path = 'COCO-Detection/'+_1+'_'+_2+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        
        elif 'fastrcnn' in model:
            backbone_1 = model.split('_')[0]
            backbone_2 = model.split('_')[1]
            backbone_3 = model.split('_')[2]
            schedule  = model.split('_')[3]
            _1 = backbone_1[:-4]
            _2 = backbone_1[-4:]
            config_path = 'COCO-Detection/'+_1+'_'+_2+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
        elif 'retinanet' in model:
            backbone_1 = model.split('_')[0]
            backbone_2 = model.split('_')[1]
            backbone_3 = model.split('_')[2]
            schedule  = model.split('_')[3]
            config_path = 'COCO-Detection/'+backbone_1+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
        elif 'rpn' in model:
            backbone_1 = model.split('_')[0]
            backbone_2 = model.split('_')[1]
            backbone_3 = model.split('_')[2]
            schedule  = model.split('_')[3]
            config_path = 'COCO-Detection/'+backbone_1+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        
        elif 'maskrcnn' in model:
            backbone_1 = model.split('_')[0]
            backbone_2 = model.split('_')[1]
            backbone_3 = model.split('_')[2]
            schedule  = model.split('_')[3]
            _1 = backbone_1[:-4]
            _2 = backbone_1[-4:]
            config_path = 'COCO-InstanceSegmentation/'+_1+'_'+_2+'_'+backbone_2[0]+'_'+backbone_2[1:]+'_'+backbone_3+'_'+schedule+'.yaml'
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
                        
        cfg.DATASETS.TRAIN = ("data_dataset_coco_train",)
        cfg.DATASETS.TEST = ("data_dataset_coco_valid")
        cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
        cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = ITER
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_name) - 1
        cfg.MODEL.RETINANET.NUM_CLASSES = len(label_name) - 1
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
