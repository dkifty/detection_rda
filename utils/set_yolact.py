#!/usr/bin/env python
import os 
import sys

def yolact_check():
    if os.path.exists('yolact'):
        print('yolact files are already existed')
        
    else:
        os.system('git clone https://github.com/dbolya/yolact.git')
        os.chdir('./yolact/external')
        os.system('git clone https://github.com/jinfagang/DCNv2_latest.git')
        os.chdir('./DCNv2_latest')
        os.system('python3 setup.py build develop')
        os.chdir('./../../..')
