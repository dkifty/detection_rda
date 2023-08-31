import os.path as osp
import os, glob
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def vid2frame(vid_path, save_path, name, frame):
    vid = cv2.VideoCapture(vid_path) 
    frame_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print('FRAME NUM: %d' % frame_num)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('FPS: %d' % round(fps))
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    count = 0  
    savecnt = 0
    
    while(vid.isOpened()):
        ret, frame = vid.read()
        
        if not ret:
            continue

        if np.mod(count,60) == 0:
        # if count < 995:
            cv2.imwrite(osp.join(save_path, name+'_{0:05d}.jpg'.format(count)), frame)
            print('Process: %d/' % count+ '%d' % frame_num)
            savecnt += 1
            
        if vid.get(cv2.CAP_PROP_POS_FRAMES) >= (frame_num):
            break

        count += 1
        # print(count)

    vid.release()
    # cv2.destroyAllWindows()

    print('%d/' % (savecnt) + '%d frames are saved' % frame_num)
    
def v2f(folder_name, formating, frame):
    
    captured_path = osp.join(os.getcwd(),'captured')
    if not os.path.exists(captured_path):
        os.mkdir(osp.join(captured_path))
    
                 
    if len(folder_name.split('/')) == 1:
        if not osp.exists(osp.join(captured_path, folder_name)):
            os.mkdir(osp.join(captured_path, folder_name))
        
    else:
        for len_folder in range(len(folder_name.split('/'))):
            if not osp.exists(osp.join(captured_path, folder_name.split('/')[len_folder])):
                os.mkdir(osp.join(captured_path, folder_name.split('/')[len_folder]))
            captured_path = osp.join(captured_path, folder_name.split('/')[len_folder])
        
    for a in glob.glob(osp.join(os.getcwd(), 'VIDEO', folder_name, '*.'+ formating)):
        file_name = a.split('/')[-1].split('.')[0]
        DIR = osp.join(os.getcwd(), folder_name)
        DIR_save = osp.join(os.getcwd(), 'captured', folder_name)
        file = osp.join(DIR_save.replace('captured', 'VIDEO'), file_name+'.'+formating)
        
        print(file_name)
        vid2frame(file, DIR_save, file_name, frame)
