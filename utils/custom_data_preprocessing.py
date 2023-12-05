#!/usr/bin/env python

import os, sys, shutil, glob, time, subprocess
import random, math, time
import yaml
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd

import ast
import json

import collections
import datetime
import labelme
import uuid
import imgviz

import pycocotools.mask
from tqdm import tqdm

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

def checking_datafile(img_format='jpg', label_format='json'):
    global data_img_list
    global data_label_list
    assert os.path.exists('data_annotated'), 'make data folder named \'data_annotated\' and put image and annoation data in that folder'
    
    data_img_list = glob.glob('./data_annotated/*.'+img_format)
    data_label_list = glob.glob('./data_annotated/*.'+label_format)
    data_img_list.sort()
    data_label_list.sort()
    
    print(f'images : {len(data_img_list)}')
    print(f'labels : {len(data_label_list)}')
        
    assert len(data_img_list) == len(data_label_list), "image, label data files are not matched. should be checked!"
    print('image, label data are checked!')
   
def label_name_check(img_format='jpg', label_format='json'):
    global label_name
    global label_list_check_
    
    data_img_list = glob.glob('./data_annotated/*.'+img_format)
    data_label_list = glob.glob('./data_annotated/*.'+label_format)
    data_img_list.sort()
    data_label_list.sort()
    
    assert os.path.exists('labels.txt'), 'make labels.txt file in this folder : format \n __ignore__ \n _background_ \n label1 \n label2 \n label3 ...'
    
    with open('labels.txt', 'r') as label:
        labels = label.readlines()
    label_list = []
    for a in labels:
        label_list.append(a.rstrip())
    label_list.sort()
    label_name = label_list
    print(f'labels are {label_list}')
    
    annotation_list = []
    for b in data_label_list:
        with open(b, 'r') as annotation:
            anno = json.load(annotation)
            
        for c in range(len(anno['shapes'])):
            annotation_list.append(anno['shapes'][c]['label'])
            
    annotation_name = list(set(annotation_list))
    annotation_name.sort()
    print(f'annotations are {annotation_name}')

    label_list_check = label_list
    label_list_check_ = []
    for label_list_check__ in  label_list:
        if label_list_check__ == '__ignore__' or label_list_check__ == '_background_':
            pass
        else:
            label_list_check_.append(label_list_check__)
    label_list_check_.sort()
    print(label_list_check_)
    if label_list_check_ == annotation_name:
        print('label names are checked!')
    else:
        strange_label = []
        for d in annotation_name:
            if d not in label_list_check_:
                strange_label.append(d)
                
        print(f'strange annotations are founded : {strange_label}')
        
        for g in strange_label:
            
            for e in data_label_list:
                with open(e, 'r') as annotation:
                    anno = json.load(annotation)
                    
                for f in range(len(anno['shapes'])):
                    if anno['shapes'][f]['label'] == g:
                        print(f'strange annotations named ==> {g} \n could founded in file ==> {e} ==> should be checked')
                        
    assert label_list_check_ == annotation_name, "check the above annotation files and try again"
 
def label_name_change(change_label_name=False):
    if change_label_name == False:
        pass
    else:
        # you could use this format : label_name_change(change_label_name = {'beans':'bean', 'weeds':'weed'})
        before_change_label = list(change_label_name.keys())
        after_change_label = list(change_label_name.values())
        
        for a in data_label_list:
            with open(a, 'r') as annotation:
                anno = json.load(annotation)
                
            for b in range(len(anno['shapes'])):
                for c,d in zip(before_change_label, after_change_label):
                    if anno['shapes'][b]['label'] == c:
                        anno['shapes'][b]['label'] = d
                    else:
                        pass
                    
            with open(a, 'w') as annotation_chaged:
                json.dump(anno, annotation_chaged)
                
    assert change_label_name == False, 'complete change label name ==> please restart after change the parameter change_label_name = False // Please change label.txt file'
    
def split_train_valid_test(split_rate=False, img_format='jpg', label_format='json'):
    img_list = data_img_list
    random.shuffle(img_list)
    
    file_name = []
    for a in img_list:
        file_name.append(a.split('/')[-1].split('.')[0].split('\\')[-1])
        
    if not os.path.exists('./data_annotated_train'):
        os.mkdir('./data_annotated_train')
    else:
        for file in os.scandir('./data_annotated_train'):
            os.remove(file.path)
        print('train folder file empty complete')
        
    if not os.path.exists('./data_annotated_valid'):
        os.mkdir('./data_annotated_valid')
    else:
        for file in os.scandir('./data_annotated_valid'):
            os.remove(file.path)
        print('valid folder file empty complete')
        
    if not os.path.exists('./data_annotated_test'):
        os.mkdir('./data_annotated_test')
    else:
        for file in os.scandir('./data_annotated_test'):
            os.remove(file.path)
        print('test folder file empty complete')
        
    print('The empty train, valid, test folder set complete')
        
    file_train = file_name
        
    if split_rate == False:
        file_train = file_name[:int(len(file_name)*0.9*0.8)]
        file_valid = file_name[int(len(file_name)*0.9*0.8):int(len(file_name)*0.9)]
        file_test = file_name[int(len(file_name)*0.9):]
    else:
        # you can you this format : split_train_valid_test(split_rate=[0.7, 0.2, 0.1])
        file_train = file_name[:int(len(file_name)*split_rate[0])]
        file_valid = file_name[int(len(file_name)*split_rate[0]):int(len(file_name)*(split_rate[0] + split_rate[1]))]
        file_test = file_name[int(len(file_name)*(split_rate[0] + split_rate[1])):]
    
    print(f'train : {len(file_train)}, valid : {len(file_valid)}, test : {len(file_test)} set complete')
        
    for a in file_train:
        shutil.copy('./data_annotated/'+a+'.'+img_format, './data_annotated_train/'+a+'.'+img_format)
        shutil.copy('./data_annotated/'+a+'.'+label_format, './data_annotated_train/'+a+'.'+label_format)
    
    for a in file_valid:
        shutil.copy('./data_annotated/'+a+'.'+img_format, './data_annotated_valid/'+a+'.'+img_format)
        shutil.copy('./data_annotated/'+a+'.'+label_format, './data_annotated_valid/'+a+'.'+label_format)
    
    for a in file_test:
        shutil.copy('./data_annotated/'+a+'.'+img_format, './data_annotated_test/'+a+'.'+img_format)
        shutil.copy('./data_annotated/'+a+'.'+label_format, './data_annotated_test/'+a+'.'+label_format)
        
    print('files are moved to each folder')
    
def labelme2coco(input_dir, labels='./labels.txt', viz = False):
    output_dir = './data_dataset_coco_' + input_dir.split('_')[-1]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    else:
        os.mkdir(output_dir)
        
    os.mkdir(os.path.join(output_dir, 'images'))
    os.mkdir(os.path.join(output_dir, 'labels'))

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    out_ann_file = os.path.join(output_dir, "annotations.json")
    label_files = glob.glob(os.path.join(input_dir, "*.json"))
    for image_id, filename in enumerate(tqdm(label_files)):
        #print("Generating dataset from:", filename) ################
        time.sleep(0.1)
        label_file = labelme.LabelFile(filename=filename)

        base = os.path.splitext(os.path.basename(filename))[0]
        out_img_file = os.path.join(output_dir, "images", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        img = img[:,:,:3]
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=os.path.relpath(out_img_file, os.path.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                # x: tolerance of the gap between the arc and the line segment
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )
        
        if viz == True:
            os.mkdir(os.path.join(output_dir, 'Visualization'))
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=30,
                    line_width=10,
                )
            out_viz_file = os.path.join(output_dir, "Visualization", base + ".jpg")
            imgviz.io.imsave(out_viz_file, viz)
                
    with open(out_ann_file, "w") as f:
        json.dump(data, f)

def counting_labels(FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test']):
    for folders in FOLDERS:
        json_list = glob.glob(os.path.join(folders, '*.json'))
        json_list.sort()
        print(f'import {folders} complete : {len(json_list)}')
        
        object_list = []
        for i in json_list:
            with open(i, 'r') as f:
                anno = json.load(f)            
            for j in range(len(anno['shapes'])):
                object_list.append(anno['shapes'][j]['label'])
        
        for k in label_list_check_:
            globals()['{}_list'.format(k)] = [a for a in object_list if k in a]
            
            b = folders.split('_')[-1]
            c = len(globals()['{}_list'.format(k)])
            print(f'{b} {k} : {c}')
            
def coco2yolo(annotation = 'annotations.json', image_size=(3840,2160)):
    with open(annotation, 'r') as f:
        anno = json.load(f)
        
    label_no = []
    img_no = []
    bbox = []
    seg = []
    for a in anno['annotations']:
        label_no_ = a['category_id']
        img_no_ = a['image_id']
        bbox_ = a['bbox']
        seg_ = a['segmentation'][0]
        
        label_no.append(label_no_)
        img_no.append(img_no_)
        bbox.append(bbox_)
        seg.append(seg_)
        
    label_dict = {'label_no' : label_no, 'img_no' : img_no, 'bbox' : bbox}
    df = pd.DataFrame(label_dict)
    
    img_name = []
    for b in list(range(df['img_no'].unique().shape[0])):
        anno_image_name = anno['images'][b]['file_name'].split('/')[-1]
        img_name.append(anno_image_name)
    
    c = list(range(df['img_no'].unique().shape[0]))
    for d,e in zip(img_name, c):
        df.loc[(df['img_no'] == e, 'img_no')] = d
    df['labels'] = df['label_no']
    
    for f, g in zip(list(range(len(label_list_check_))), label_list_check_):
        df.loc[(df['label_no'] == f+1), 'label_no'] = g
        
    df['labels'] = df['labels'] - 1
    
    width = image_size[0]
    height = image_size[1]

    seg_x = []
    seg_y = []

    for i in df.seg.tolist():
        seg_x_ = []
        seg_y_ = []

        for a, b in enumerate(i):
            if a%2 == 0:
                seg_x_.append(b/width)
            if a%2 == 1:
                seg_y_.append(b/height)
        seg_x.append(seg_x_)
        seg_y.append(seg_y_)
    
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    w = []
    h = []
    
    for i in df.bbox.tolist():
        x1.append(i[0])
        y1.append(i[1])
        w.append(i[2])
        h.append(i[3])
        x2.append(i[0] + i[2])
        y2.append(i[1] + i[3])
        
    df['bbox_x_min'] = x1
    df['bbox_y_min'] = y1
    df['bbox_x_max'] = x2
    df['bbox_y_max'] = y2
    df['w'] = w
    df['h'] = h
    df['seg_x'] = seg_x
    df['seg_y'] = seg_y
    
    df['bbox_x_centre'] = df['bbox_x_min'] + df['w'] / 2
    df['bbox_y_centre'] = df['bbox_y_min'] + df['h'] / 2
    
    df['x_centre_yolo'] = df['bbox_x_centre'] / width
    df['y_centre_yolo'] = df['bbox_y_centre'] / height
    df['w_yolo'] = df['w'] / width
    df['y_yolo'] = df['h'] / height

    df.to_csv(annotation.replace('json', 'csv'))
    
    img_list = glob.glob(annotation.replace('annotations.json', 'images')+'/*.jpg')

    label_folders = ['labels_seg', 'labels_det']
    dataset_folders = ['data_annotated_coco_train', 'data_annotated_coco_valid', 'data_annotated_coco_test']

    for dataset_folder in dataset_folders:
        for label_folder in label_folders:
            if not os.path.exists(os.path.join(dataset_folder, label_folder)):
                os.mkdir(os.path.join(dataset_folder, label_folder))
    
    for j in img_list:
        with open(j.replace('images', 'labels_det')[:-4]+'.txt', 'wb') as k:
            anno = df[df['img_no'].str.contains(j.split('/')[-1])]
            
            anno_list = []
            
            for l in list(range(anno.shape[0])):
                x_centre_yolo = anno.iloc[l,:]['x_centre_yolo']
                y_centre_yolo = anno.iloc[l,:]['y_centre_yolo']
                w_yolo = anno.iloc[l,:]['w_yolo']
                y_yolo = anno.iloc[l,:]['y_yolo']
                labels = anno.iloc[l,:]['labels']
                
                anno_list.append([str(labels) + ' ' +  str(x_centre_yolo) + ' ' +  str(y_centre_yolo) + ' ' +  str(w_yolo) + ' ' +  str(y_yolo)])
                
            counting = 1
            
            for m in anno_list:
                k.write(('\n' + m[0]).encode('utf-8'))            
                
        k.close()
    
        with open(j.replace('images', 'labels_det')[:-4]+'.txt', 'r') as fin:
            data = fin.read().splitlines(True)
        with open(j.replace('images', 'labels_det')[:-4]+'.txt', 'w') as fout:
            fout.writelines(data[1:])

    for k in img_list:
        with open(j.replace('images', 'labels_seg')[:-4]+'.txt', 'wb') as k:
            anno = df[df['img_no'].str.contains(j.split('/')[-1])]
            
            segmentation_list = []
            
            for l in list(range(anno.shape[0])):
                seg_x_list = ast.literal_eval(anno.iloc[l,:]['seg_x'])
                seg_y_list = ast.literal_eval(anno.iloc[l,:]['seg_y'])
                labels = anno.iloc[l,:]['labels']
                for seg_x, seg_y in zip(seg_x_list, seg_y_list):
                    if seg_x < 0:
                        seg_x = 0.
                    if seg_y < 0:
                        seg_y = 0.
                    seg_list.append(str(seg_x))
                    seg_list.append(str(seg_y))
                seg = ' '.join(seg_list)

                segmentation_list.append(seg)
                                
            counting = 1
            
            for m in segmentation_list:
                k.write(('\n' + m[0]).encode('utf-8'))            
                
        k.close()
    
        with open(j.replace('images', 'labels_seg')[:-4]+'.txt', 'r') as fin:
            data = fin.read().splitlines(True)
        with open(j.replace('images', 'labels_seg')[:-4]+'.txt', 'w') as fout:
            fout.writelines(data[1:])


def yolov5_check():
    if os.path.exists('yolov5'):
        print('yolov5 files are already existed')
    else:
    	os.system('git clone https://github.com/ultralytics/yolov5')
 
def img_label_preprocessing(img_format='jpg', label_format='json', change_label_name=False, split_rate=False, FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test'], FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']):
    checking_datafile(img_format=img_format, label_format=label_format)
    print('')
    label_name_change(change_label_name=change_label_name)
    print('')
    label_name_check()
    print('')
    split_train_valid_test(split_rate=split_rate, img_format=img_format, label_format=label_format)
    print('')
    
    for folders in FOLDERS:
        print('-------------', folders.split('_')[-1], '-------------')
        print('start to create datasets in coco_form')
        
        labelme2coco(folders, labels='./labels.txt', viz=False)
        print('complete')
        print('')
        
    counting_labels(FOLDERS = FOLDERS)
    print('')
    
def yolo_config_preprocessing(annotation = 'annotations.json', image_size=(3840,2160), FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']):
    for folders_coco in FOLDERS_COCO:
        print('-------------', folders_coco.split('_')[-1], '-------------')
        print('start to create datasets in yolo_form')
        
        coco2yolo(annotation=os.path.join(folders_coco, 'annotations.json'), image_size = (3840,2160))
        print('complete')
    
def data_preprocessing(label2coco = True, coco2yolo = True, img_format='jpg', label_format='json', change_label_name=False, split_rate=False, FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test'], FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test'], annotation = 'annotations.json', image_size=(3840,2160)):
    print('')
    if label2coco == True:
        img_label_preprocessing(img_format=img_format, label_format=label_format, change_label_name=change_label_name, split_rate=split_rate, FOLDERS = FOLDERS, FOLDERS_COCO = FOLDERS_COCO)
    else:
        global label_name
        
        with open('labels.txt', 'r') as label:
            labels = label.readlines()
            label_list = []
            for a in labels:
                label_list.append(a.rstrip())
            label_list.sort()
        label_name = label_list
        
    checking_datafile(img_format=img_format, label_format=label_format)
    counting_labels(FOLDERS = FOLDERS)
    print('data files checked')
    print('')
    
    print('train : ', len(glob.glob('./data_dataset_coco_train/images/*jpg')))
    print('valid : ', len(glob.glob('./data_dataset_coco_valid/images/*jpg')))
    print('test : ', len(glob.glob('./data_dataset_coco_test/images/*jpg')))
    print('')
    print('coco form datasets checked')
    
    if coco2yolo == True:
        yolo_config_preprocessing(annotation = annotation, image_size=image_size)
    else:
    	for folders_coco in FOLDERS_COCO:
    		_img = len(glob.glob(os.path.join(folders_coco, 'images', '*.')+img_format))
    		_txt = len(glob.glob(os.path.join(folders_coco, 'labels', '*.')+'txt'))
    		
    		if _img == _txt:
    			print(f'yolo form annotations already existed... {folders_coco} ... checked... complete')
    		assert _img == _txt, 'check againg yolo form annotations'
