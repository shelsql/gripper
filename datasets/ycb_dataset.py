import sys
import os
import json
import glob
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class BOPYCBDataset(Dataset):
    def __init__(self,
                 dataset_location = "/root/autodl-tmp/shiqian/datasets/bop/ycbv"):
        self.dataset_location = dataset_location
        self.video_dirs = glob.glob(self.dataset_location + "/test/*")
        print('Found %d videos in %s' % (len(self.video_dirs), self.dataset_location))
        self.all_rgbs = []
        self.all_masks = []
        self.all_gts = []
        self.all_cams = []
        for vid_dir in self.video_dirs:
            scene_gt = json.loads(open(vid_dir + "/scene_gt.json").read())
            scene_cam = json.loads(open(vid_dir + "/scene_camera.json").read())
        
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass