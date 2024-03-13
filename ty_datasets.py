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
            camera_intrinsic_path = os.path.join(video, "_camera_settings.json")
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
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        kpts = json.loads(open(kpts_path).read())
        npy = np.load(npy_path)
        sample = {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "kpts": kpts,
            "npy": npy,
            "intrinsics": intrinsics
        }
        return sample
    
    def __len__(self):
        return len(self.all_paths)
        

class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/datasets/Ty_data",
                 use_augs=False,
                 S=32,
                 strides=[1,2]
                 ):
        super().__init__()
        print("Loading pose tracking dataset...")
        self.S = S
        self.dataset_location = dataset_location
        self.video_dirs = glob.glob(dataset_location + "/*")
        print("Found %d videos in %s" % (len(self.video_dirs), self.dataset_location))
        
        self.img_paths = {}
        for video in self.video_dirs:
            self.img_paths[video] = [path[:-4] for path in sorted(glob.glob(video + "/*.png"))]
        
        self.all_videos = []
        self.all_full_idx = []
        
        for video in self.video_dirs:
            for stride in strides:
                S_local = len(self.img_paths[video])
                for ii in range(0, max(S_local-self.S*stride,1), 1):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    self.all_videos.append(video)
                    self.all_full_idx.append(full_idx)
        
    def __getitem__(self, index):
        video_dir = self.all_videos[index]
        full_idx = self.all_full_idx[index]
        camera_intrinsic_path = os.path.join(video_dir, "_camera_settings.json")
        camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
        
        print(video_dir, full_idx)
        
        rgbs = []
        depths = []
        masks = []
        mask_preds = []
        ours_009_masks = []
        ours_masks = []
        kptss = []
        npys = []
        for idx in full_idx:
            path = self.img_paths[video_dir][idx]
        
            rgb_path = path + ".png"
            depth_path = path + ".exr"
            mask_path = path + "_mask.exr"
            kpts_path = path + ".json"
            npy_path = path + ".npy"
            
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            kpts = json.loads(open(kpts_path).read())
            npy = np.load(npy_path)
            
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            kptss.append(kpts)
            npys.append(npy)
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        npys = np.stack(npys, axis = 0)
        sample = {
            "rgbs": rgbs,
            "depths": depths,
            "masks": masks,
            "kptss": kptss,
            "npys": npys,
            "intrinsics": camera_intrinsic
        }
        
        return sample
    def __len__(self):
        return len(self.all_videos)