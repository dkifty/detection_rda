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

def run_model(model=None, train=True, val = True, test = True, iou = 0.5, resize_img=1024, batch=16, epochs=200, FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test'], NUM_WORKERS = 2, IMS_PER_BATCH = 2, ITER = 12000, device=0):
    if 'yolov5' in model:
        if train == True:
            print('-----', model, 'train task -----')
            yolov5_run = 'python3 yolov5/train.py --img ' + str(resize_img) + ' --batch ' + str(batch) + ' --epochs ' + str(epochs) + ' --device' + str(device) + ' --data yolo_configs/data/custom.yaml --cfg yolo_configs/models/custom_' + model + '.yaml --name custom_results_' + model + ' --cache'
            os.system(yolov5_run)
        else:
            pass
        if val == True:
            print('-----', model, 'validation task -----')
            yolov5_val = 'python yolov5/val.py --weights yolov5/runs/train/custom_results_' + model + '/weights/best.pt --data yolo_configs/data/custom.yaml --img ' + str(resize_img) +' --iou ' +str(iou)+' --half --task val --save-txt'
            os.system(yolov5_val)
        else:
            pass
        if test == True:
            print('-----', model, 'test task -----')
            yolov5_test = 'python yolov5/val.py --weights yolov5/runs/train/custom_results_' + model + '/weights/best.pt --data yolo_configs/data/custom.yaml --img ' + str(resize_img) +' --iou ' +str(iou)+' --half --task test'
            yolov5_detect = 'python yolov5/detect.py --weights yolov5/runs/train/custom_results_' + model + '/weights/best.pt --save-txt --img '+str(resize_img)+' --conf 0.4 --source data_dataset_coco_test/images --save-txt --iou ' +str(iou)
            os.system(yolov5_test)
            os.system(yolov5_detect)
        else:
            pass
    elif 'yolov8' in model:
        if not os.path.exists('yolov8'):
            os.mkdir('yolov8')
        
        ROOT = os.getcwd()
        os.chdir('yolov8')
        
        if not os.path.exist('weights'):
            os.mkdir('weights')
            
        if train == True:
            global yolov8_savedir
            print('-----', model, 'train task -----')
            yolov8 = YOLO(os.path.join('weights', model))
            yolov8_train = model.train(model=os.path.join('weights', model), data=os.path.join(ROOT, 'yolo_configs/data/custom.yaml'), , imgsz=resize_img, epochs=epochs, batch=batch, device=device)
            yolov8_savedir = str(results.save_dir)
        else:
            pass
        if val == True:
            print('-----', model, 'validation task -----')
            yolov8_custom = YOLO(os.path.join(ROOT, 'runs/detect', yolov8_savedir, 'weights/last.pt'))
            yolov8_test = yolov8_custom.val(split='val', iou=iou)
        else:
            pass
        if test == True:
            print('-----', model, 'test task -----')
            yolov8_custom = YOLO(os.path.join(ROOT, 'runs/detect', yolov8_savedir, 'weights/last.pt'))
            yolov8_test = yolov8_custom.val(split='test', iou=iou)
            yolov8_predict = yolov8_custom.predict(source = 'data_dataset_coco_test/images', conf=0.4, save_txt=True, save=True)
        else:
            pass
        
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
        
        if train == True:
            print('-----', model, 'train task -----')
            yolov4_run = './darknet/darknet detector train yolo_configs/data/obj.data yolo_configs/models/' + model + '.cfg yolo_configs/models/yolov4.conv.137 -dont_show -map -clear'
            os.system(yolov4_run)    
        else:
            pass
        if val == True: 
            print('-----', model, 'validation task -----')
            darknet_val = './darknet/darknet detector map yolo_configs/data/obj.data yolo_configs/models/' + model + '.cfg darknet/'+model+'_best.weights -points 0'
            os.system(darknet_val)
            print(darknet_val)
        else:
            pass
        if test == True:
            print('-----', model, 'test task -----')
            if not os.path.exists('darknet/'+model+'_outputs'):
                os.mkdir('darknet/'+model+'_outputs')
            
            os.system('sed -i \'s/batch=16/batch=1/\' yolo_configs/models/' + model + '.cfg')
            os.system('sed -i \'s/subdivisions=8/subdivisions=1/\' yolo_configs/models/' + model + '.cfg')
            with open('yolo_configs/data/obj.data', 'r') as obj_data_:
                obj_data = obj_data_.readlines()
            obj_data[3] = 'names = ../yolo_configs/data/obj.names\n'
            with open('yolo_configs/data/obj.data', 'w') as obj_data_:
                for lines in obj_data:
                    obj_data_.write(lines)
            
            os.system('sed -i \'s/names = yolo_configs/data/obj.names/names = ../yolo_configs/data/obj.names/\' yolo_configs/data/obj.data')
                        
            os.chdir('./darknet')

            print(os.getcwd())
            for test_images in glob.glob('../data_dataset_coco_test/images/*.jpg'):
                darknet_test = './darknet detector test ../yolo_configs/data/obj.data ../yolo_configs/models/' + model + '.cfg ./'+model+'_best.weights ' + test_images + ' -thresh 0.5 -dont_show'   
                os.system(darknet_test)
                shutil.move('predictions.jpg', './'+model+'_outputs/'+test_images.split('/')[-1])
            print('test images ---> darknet/'+model+'_outputs')
                
            os.chdir('./..')
            
            with open('yolo_configs/data/obj.data', 'r') as obj_data_:
                obj_data = obj_data_.readlines()
            obj_data[3] = 'names = yolo_configs/data/obj.names\n'
            with open('yolo_configs/data/obj.data', 'w') as obj_data_:
                for lines in obj_data:
                    obj_data_.write(lines)
            os.system('sed -i \'s/batch=1/batch=16/\' yolo_configs/models/' + model + '.cfg')
            os.system('sed -i \'s/subdivisions=1/subdivisions=8/\' yolo_configs/models/' + model + '.cfg')        
        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                os.remove(yolo_annotations.replace('labels', 'images')) 
                
    elif 'yolov3' in model:
        #if not os.path.exists('yolo_configs/models/darknet53.conv.74'):
        #    os.system('wget https://pjreddie.com/media/files/darknet53.conv.74')
        #    shutil.move('darknet53.conv.74','yolo_configs/models/darknet53.conv.74')
        #else:
        #    print('model pre-weights already exsits\n')
        
        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                shutil.copy(yolo_annotations, yolo_annotations.replace('labels', 'images'))
        
        if train == True:
            print('-----', model, 'train task -----')
            yolov3_run = './darknet/darknet detector train yolo_configs/data/obj.data yolo_configs/models/' + model + '.cfg yolo_configs/models/darknet53.conv.74 -dont_show -map -clear'
            os.system(yolov3_run)
        else:
            pass
        if val == True: 
            print('-----', model, 'validation task -----')
            darknet_val = './darknet/darknet detector map yolo_configs/data/obj.data yolo_configs/models/' + model + '.cfg darknet/'+model+'_best.weights -points 0'
            os.system(darknet_val)
            print(darknet_val)
        else:
            pass
        if test == True:
            print('-----', model, 'test task -----')
            if not os.path.exists('darknet/'+model+'_outputs'):
                os.mkdir('darknet/'+model+'_outputs')
            
            os.system('sed -i \'s/batch=16/batch=1/\' yolo_configs/models/' + model + '.cfg')
            os.system('sed -i \'s/subdivisions=8/subdivisions=1/\' yolo_configs/models/' + model + '.cfg')
            with open('yolo_configs/data/obj.data', 'r') as obj_data_:
                obj_data = obj_data_.readlines()
            obj_data[3] = 'names = ../yolo_configs/data/obj.names\n'
            with open('yolo_configs/data/obj.data', 'w') as obj_data_:
                for lines in obj_data:
                    obj_data_.write(lines)
            
            os.system('sed -i \'s/names = yolo_configs/data/obj.names/names = ../yolo_configs/data/obj.names/\' yolo_configs/data/obj.data')
                        
            os.chdir('./darknet')

            print(os.getcwd())
            for test_images in glob.glob('../data_dataset_coco_test/images/*.jpg'):
                darknet_test = './darknet detector test ../yolo_configs/data/obj.data ../yolo_configs/models/' + model + '.cfg ./'+model+'_best.weights ' + test_images + ' -thresh 0.5 -dont_show'   
                os.system(darknet_test)
                shutil.move('predictions.jpg', './'+model+'_outputs/'+test_images.split('/')[-1])
            print('test images ---> darknet/'+model+'_outputs')
                
            os.chdir('./..')
            
            with open('yolo_configs/data/obj.data', 'r') as obj_data_:
                obj_data = obj_data_.readlines()
            obj_data[3] = 'names = yolo_configs/data/obj.names\n'
            with open('yolo_configs/data/obj.data', 'w') as obj_data_:
                for lines in obj_data:
                    obj_data_.write(lines)
            os.system('sed -i \'s/batch=1/batch=16/\' yolo_configs/models/' + model + '.cfg')
            os.system('sed -i \'s/subdivisions=1/subdivisions=8/\' yolo_configs/models/' + model + '.cfg')
        
        for folders_coco in FOLDERS_COCO:
            label_dir = os.path.join(folders_coco, 'labels')
            for yolo_annotations in glob.glob(os.path.join(label_dir, '*.txt')):
                os.remove(yolo_annotations.replace('labels', 'images'))
    
    elif 'fasterrcnn' or 'fastrcnn' or 'retinanet' or 'rpn' or 'maskrcnn' in model:
        dataset_name = 'data_dataset_coco_train'
        
        if not dataset_name in DatasetCatalog.list():
            for d in ["data_dataset_coco_train", "data_dataset_coco_valid", "data_dataset_coco_test"]:
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
        
        if not os.path.exists('detectron2/'+model):
            os.mkdir('detectron2/'+model)
        
        if train == True:
            print('-----', model, 'train task -----')
            trainer.train()
            os.rename(os.path.join(cfg.OUTPUT_DIR, 'model_final.pth'), os.path.join(cfg.OUTPUT_DIR, model+'_weight.pth'))
        else:
            pass
        if val == True:
            print('-----', model, 'validation task -----')
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model+"_weight.pth") # 혹시나 오류가 날때는 그냥 경로 지정해주면 됨
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
            cfg.DATASETS.TEST = ("data_dataset_coco_valid") # 등록된 이름
            predictor = DefaultPredictor(cfg)
            
            evaluator = COCOEvaluator("data_dataset_coco_valid", output_dir="detectron2/"+model)
            val_loader = build_detection_test_loader(cfg, "data_dataset_coco_valid")
            print(inference_on_dataset(predictor.model, val_loader, evaluator))
        else:
            pass
        if test == True:
            print('-----', model, 'test task -----')
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model+"_weight.pth") # 혹시나 오류가 날때는 그냥 경로 지정해주면 됨
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
            cfg.DATASETS.TEST = ("data_dataset_coco_test") # 등록된 이름
            predictor = DefaultPredictor(cfg)
            
            evaluator = COCOEvaluator("data_dataset_coco_test", output_dir="detectron2/"+model)
            val_loader = build_detection_test_loader(cfg, "data_dataset_coco_test")
            print(inference_on_dataset(predictor.model, val_loader, evaluator))
            
            dataset_dicts = glob.glob('data_dataset_coco_test/images/*.jpg') # 위에서 등록한 이미지 path
            for d in dataset_dicts:    
                im = cv2.imread(d)
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1],
                               scale=0.8,
                               metadata = metadata
                              )
                v = v.draw_instance_predictions(outputs["instances"].to("cpu")) ## GPU
                save = v.get_image()[:, :, ::-1]
                cv2.imwrite('detectron2/'+model+'/'+d.split('/')[-1],save)
        else:
            pass
        
    elif 'yolact' in model:
        if train == True:
            print('-----', model, 'train task -----')
            yolact_run = 'python yolact/train.py --config='+model+'_custom_config --batch_size='+batch
            os.system(yolact_run)
        else:
            pass
        if val == True:
            print('-----', model, 'validation task -----')
            pass
        else:
            pass
        if test == True:
            print('-----', model, 'test task -----')
            if not os.path.exists('yolact/output_images'):
                os.mkdir('yolact/output_images')
            os.systemp('python yolact/eval.py --trained_model=yolact/weights/'+[a for a in glob.glob('yolact/weights/*.pth')][-1].split('/')[-1]+' --config='++model+'_custom_config --score_threshold=0.4 --top_k=15 --images=data_dataset_coco_test/images:yolact/output_images')
        else:
            pass
