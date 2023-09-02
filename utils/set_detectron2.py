#!/usr/bin/env python
import os 
import sys

def detectron2_check():
    if os.path.exists('detectron2'):
        import torch, detectron2
        os.system('nvcc --version')
        TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
        CUDA_VERSION = torch.__version__.split("+")[-1]
        print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
        print("detectron2:", detectron2.__version__)
        print('detectron2 files are already existed')
    else:
        os.system('python -m pip install pyyaml==5.1')
        
        os.system('git clone https://github.com/facebookresearch/detectron2.git')
        os.system('python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html')
        os.system('pip install opencv-python')
        
        import distutils.core
        dist = distutils.core.run_setup("detectron2/setup.py")
        for a in dist.install_requires:
        	os.system('python -m pip install '+a)
        sys.path.insert(0, os.path.abspath('detectron2/detectron2'))
        
        with open('detectron2/detectron2/evaluation/coco_evaluation.py', 'r') as f:
            lines = f.readlines()
        lines[369] = lines[369].replace('precisions[:, :, idx, 0, -1]','precisions[0, :, idx, 0, -1]')
        with open('detectron2/detectron2/evaluation/coco_evaluation.py', 'w') as f:
            for line in lines:
                f.write(line)
        
        print('')        
        import torch, detectron2
        os.system('nvcc --version')
        TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
        CUDA_VERSION = torch.__version__.split("+")[-1]
        print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
        print("detectron2:", detectron2.__version__)
        print('detectron2 files are already existed')
        
        
