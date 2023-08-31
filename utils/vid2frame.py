import os.path as osp
import os, glob
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def vid2frame(vid_path, save_path, name, frame):
    '''
    Video to frames

    Params: 동영상 경로, 프레임 저장 경로, 저장 프레임 이름
    Return:
    '''

    vid = cv2.VideoCapture(vid_path)                # 동영상 로드
    frame_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)   # 전체 프레임 수
    print('FRAME NUM: %d' % frame_num)
    fps = vid.get(cv2.CAP_PROP_FPS)                 # 현재 FPS
    print('FPS: %d' % round(fps))

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)             # 현재 프레임 번호를 0으로 설정

    count = 0   # 프레임 카운팅용 변수
    savecnt = 0 # 저장 카운팅용 변수
    while(vid.isOpened()):      # 동영상이 정상적으로 불러와졌으면 loop 동작
        ret, frame = vid.read() # 프레임 읽기

        # print ('%d' % count + '%s' % ret)
        # assert ret,'False frame %d/' % count + '%d' % frame_num

        # 리턴이 없으면 해당 루프 무시
        if not ret:
            continue

        # cv2.imshow('video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # 1초 마다 프레임 영상 저장 (FPS가 15이기 때문에 count와 15 mod 연산)
        # if np.mod(count,5) == 0:
        #     cv2.imwrite(osp.join(save_path, 'frame_2107231100_{0:05d}.png'.format(count)), frame)
        #     print('Process: %d/' % count+ '%d' % frame_num)
        #     savecnt += 1

        if np.mod(count,60) == 0:
        # if count < 995:
            cv2.imwrite(osp.join(save_path, name+'_{0:05d}.jpg'.format(count)), frame)
            print('Process: %d/' % count+ '%d' % frame_num)
            savecnt += 1

        # 현재 프레임 번호가 총 프레임 수보다 크면 루프 탈출
        if vid.get(cv2.CAP_PROP_POS_FRAMES) >= (frame_num):
            break

        count += 1
        # print(count)

    vid.release()   # 비디오 해제
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
