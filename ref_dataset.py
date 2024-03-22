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

class ReferenceDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/code/gripper/rendered_franka",
                 use_augs=False,
                 num_views=64,
                 strides=[1,2]
                 ):
        super().__init__()
        print("Loading reference view dataset...")
        self.N = num_views
        self.dataset_location = dataset_location
        self.rgb_paths = glob.glob(dataset_location + "/*png")
        self.camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
        print("Found %d views in %s" % (len(self.rgb_paths), self.dataset_location))
    
        
    def __getitem__(self, index):
        
        rgbs = []
        depths = []
        masks = []
        c2ws = []
        
        camera_intrinsic = json.loads(open(self.camera_intrinsic_path).read())
        
        for glob_rgb_path in self.rgb_paths:
            path = glob_rgb_path[:-8]
        
            rgb_path = path + "_rgb.png"
            depth_path = path + "_depth1.exr"
            mask_path = path + "_id1.exr"
            c2w_path = path + "_c2w.npy"
            
            #print(rgb_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            c2w = np.load(c2w_path)
            
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            c2ws.append(c2w)
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        c2ws = np.stack(c2ws, axis = 0)
        print(depths.shape)
        sample = {
            "rgbs": rgbs,
            "depths": depths,
            "masks": masks,
            "c2ws": c2ws,
            "intrinsics": camera_intrinsic
        }
        
        return sample
    def __len__(self):
        return 1