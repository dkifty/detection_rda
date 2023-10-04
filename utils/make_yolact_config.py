#!/usr/bin/env python
import os
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

def make_yolact_config(resize_img=1024):
    global yolact_config_file
    with open('yolact/data/config.py', 'r') as yolact_config_file_:
        yolact_config_file = yolact_config_file_.readlines()
            
    backbone_position = [d for d in range(len(yolact_config_file)) if '------ MASK BRANCH TYPES -----------' in yolact_config_file[d]] 
    
    yolact_resnet101_config = 'yolact_resnet101_config = yolact_base_config.copy({\'name\': \'yolact_resnet101\', \'backbone\': resnet101_backbone.copy()})\n'
    yolact_config_file.insert(int(backbone_position[0]), yolact_resnet101_config)
    
    yolact_plus_resnet101_config = 'yolact_plus_resnet101_config = yolact_plus_base_config.copy({\'name\': \'yolact_plus_resnet101\',\'backbone\': resnet101_dcn_inter3_backbone.copy({\'selected_layers\': list(range(1, 4)),\'pred_aspect_ratios\': [ [[1, 1/2, 2]] ]*5,\'pred_scales\': [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]],\'use_pixel_scales\': True,\'preapply_sqrt\': False,\'use_square_anchors\': False})})\n'
    yolact_config_file.insert(int(backbone_position[0]), yolact_plus_resnet101_config)
    
    dataset_position = [a for a in range(len(yolact_config_file)) if '-------- TRANSFORMS ---------' in yolact_config_file[a]]
    
    DATASETS_DICT = {}
    DATASETS_DICT['name'] = 'custom_dataset'
    DATASETS_DICT['train_info'] = os.path.join('data_dataset_coco_train', 'annotations.json')
    DATASETS_DICT['train_images'] = os.path.join('data_dataset_coco_train')
    DATASETS_DICT['valid_info'] = os.path.join('data_dataset_coco_valid', 'annotations.json')
    DATASETS_DICT['valid_images'] = os.path.join('data_dataset_coco_valid')
    DATASETS_DICT['class_names'] = tuple(label_list_check_)
    LABEL_MAP = {}
    for a in range(len(tuple(label_list_check_))):
        LABEL_MAP[a+1] = a+1
    DATASETS_DICT['label_map'] = LABEL_MAP
    dataset_ = 'custom_dataset = dataset_base.copy(' + str(DATASETS_DICT) + ')\n'
    
    yolact_config_file.insert(int(dataset_position[0]), dataset_)
    
    yolact_models_position = [b for b in range(len(yolact_config_file)) if '---- YOLACT++ CONFIGS ----' in yolact_config_file[b]]
    
    YOLACT_MODELS = ['darknet53', 'resnet50', 'resnet101']
    for yolact_models in YOLACT_MODELS:
        YOLACT_CONFIG = {}
        YOLACT_CONFIG['name'] = 'yolact_' + yolact_models + '_custom'
        YOLACT_CONFIG['dataset'] = 'custom_dataset'
        YOLACT_CONFIG['num_classes'] = 'len(custom_dataset.class_names) + 1'
        YOLACT_CONFIG['max_size'] = resize_img
        yolact_models_ = 'yolact_'+yolact_models+'_custom_config = yolact_'+yolact_models+'_config.copy('+str(YOLACT_CONFIG)+')\n'
        
        yolact_config_file.insert(int(yolact_models_position[0]), yolact_models_)
        
    yolact_plus_models_position = [c for c in range(len(yolact_config_file)) if '# Default config' in yolact_config_file[c]]
    
    YOLACT_PLUS_MODELS = ['resnet50', 'resnet101']
    for yolact_plus_models in YOLACT_PLUS_MODELS:
        YOLACT_PLUS_CONFIG = {}
        YOLACT_PLUS_CONFIG['name'] = 'yolact_plus_' + yolact_models + '_custom'
        YOLACT_PLUS_CONFIG['dataset'] = 'custom_dataset'
        YOLACT_PLUS_CONFIG['num_classes'] = 'len(custom_dataset.class_names) + 1'
        YOLACT_PLUS_CONFIG['max_size'] = resize_img
        yolact_plus_models = 'yolact_plus_'+yolact_models+'_custom_config = yolact_plus_'+yolact_models+'_config.copy('+str(YOLACT_CONFIG)+')\n'
        
        yolact_config_file.insert(int(yolact_plus_models_position[0]), yolact_plus_models)
        
    with open('yolact/data/config.py', 'w') as yolact_config_file_:
        for line in yolact_config_file:
            yolact_config_file_.write(line)
            
    with open('yolact/train.py', 'r') as train_config_:
        train_config = train_config_.readlines()
    train_config_1 = 'parser.add_argument(\'--num_workers\', default=0, type=int,\n'
    train_config_2 = '                                  pin_memory=True, , generator=torch.Generator(device=\'cuda\'))\n'
    
    train_config[int([c for c in range(len(train_config)) if 'parser.add_argument(\'--num_workers\', default=4, type=int,' in train_config[c]][0])] = train_config_1
    train_config[int([c for c in range(len(train_config)) if 'pin_memory=True)' in train_config[c]][0])] = train_config_2
    
    with open('yolact/train.py', 'w') as train_config_:
        for line in train_config:
            train_config_.write(line)
                        
    with open('yolact/backbone.py', 'r') as backbone_config_:
        backbone_config = backbone_config_.readlines()
    backbone_config[int([c for c in range(len(backbone_config)) if 'state_dict = torch.load(path)' in backbone_config[c]][0])] = '        state_dict = torch.load(path, my_location=\'cuda\')\n'
    
    with open('yolact/backbone.py', 'w') as backbone_config_:
        for line in backbone_config:
            backbone_config_.write(line)
            
    with open('yolact/eval.py', 'r') as eval_config_:
        eval_config = eval_config_.readlines()
    
    __ = int([c for c in range(len(eval_config)) if 'aps[iou_idx][iou_type].append(ap_obj.get_ap())' in eval_config[c]][0])
    eval_config.insert(__+1, '        print_maps(all_maps)\n') 
    eval_config.insert(__+1, '        print(\'#################### Class:\', cfg.dataset.class_names[_class], \'####################\')\n') 
    eval_config.insert(__+1, '            all_maps[iou_type][\'all\'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))\n') 
    eval_config.insert(__+1, '                all_maps[iou_type][int(threshold * 100)] = mAP\n') 
    eval_config.insert(__+1, '                mAP = aps[i][iou_type][_class] * 100 if len(aps[i][iou_type]) > 0 else 0\n') 
    eval_config.insert(__+1, '            for i, threshold in enumerate(iou_thresholds):\n') 
    eval_config.insert(__+1, '            all_maps[iou_type][\'all\'] = 0  # Make this first in the ordereddict\n') 
    eval_config.insert(__+1, '        for iou_type in (\'box\', \'mask\'):\n') 
    eval_config.insert(__+1, '        all_maps = {\'box\': OrderedDict(), \'mask\': OrderedDict()}\n') 
    
    with open('yolact/eval.py', 'w') as eval_config_:
        for line in eval_config:
            eval_config_.write(line)

    print('yolact configs.... complete')
