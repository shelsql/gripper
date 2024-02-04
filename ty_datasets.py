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

class PoseDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/datasets/Ty_data",
                 use_augs=False
                 ):
        super().__init__()
        print("Loading pose estimation dataset...")
        self.dataset_location = dataset_location
        self.video_dirs = glob.glob(dataset_location + "/*")
        print("Found %d videos in %s" % (len(self.video_dirs), self.dataset_location))
        self.all_paths = []
        self.all_intrinsics = []
        for video in self.video_dirs:
            camera_intrinsic_path = os.path.join(video, "train_camera_intrinsic.json")
            camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
            for img_path in glob.glob(video + "/*.png"):
                path_name = img_path[:-4]
                self.all_paths.append(path_name)
                self.all_intrinsics.append(camera_intrinsic)
        
        
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
            
    def __getitem__(self, index):
        path = self.all_paths[index]
        intrinsics = self.all_intrinsics[index]
        
        rgb_path = path + ".png"
        depth_path = path + ".exr"
        mask_path = path + "_mask.exr"
        kpts_path = path + ".json"
        npy_path = path + ".npy"
        
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]   
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        kpts = json.loads(open(kpts_path).read())
        npy = np.load(open(npy_path))
        sample = {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "kpts": kpts,
            "npy": npy
        }
        return
    def __len__(self):
        return len(self.all_paths)
        

class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/datasets/Ty_data",
                 use_augs=False,
                 S=32,
                 ):
        super().__init__()
        print("Loading pose tracking dataset...")
        self.dataset_location = dataset_location
        self.video_dirs = glob.glob(dataset_location + "/*")
        print("Found %d videos in %s" % (len(self.video_dirs), self.dataset_location))
    def __getitem__(self, index):
        return
    def __len__(self):
        return