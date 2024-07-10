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
from utils.quaternion_utils import quaternion_to_matrix
from utils.spd import transform_pointcloud
import joblib

class RealDataset(Dataset):
    def __init__(self,
                 dataset_location = "./real_data"
                 ):
        self.dataset_location = dataset_location
        self.videos = glob.glob(dataset_location + "/data_sample*")
        self.model_pc = np.loadtxt("/root/autodl-tmp/shiqian/code/render/hephaestus/sampled_2048.txt")
        print("Found %d videos in directory %s" % (len(self.videos), self.dataset_location))
        
    def __getitem__(self, index):
        vid_dir = self.videos[index]
        rgbs = []
        masks = []
        depths = []
        g2cs = []
        camera_intrinsics = json.loads(open(vid_dir + "/camera_intrinsics.json").read())["intrinsics"]
        camera_K = np.array([
            [camera_intrinsics["fx"], 0, camera_intrinsics["cx"]],
            [0, camera_intrinsics["fy"], camera_intrinsics["cy"]],
            [0, 0, 1]
        ])
        for frame in glob.glob(vid_dir + "/data*"):
            rgb_path = frame + "/rgb.png"
            depth_path = frame + "/depth.png"
            mask_path = frame + "/mask.png"
            g2r_path = frame + "/pose.npy"
            r2c_path = frame + "/cam_pose.txt"
            
            rgb = rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis] / 1000.0
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)[:,:,np.newaxis]
            print(mask.shape)
            g2r = np.load(g2r_path)
            
            r2c_q = np.loadtxt(r2c_path)[:4]
            r2c_t = np.loadtxt(r2c_path)[4:]
            r2c_r = quaternion_to_matrix(torch.tensor(r2c_q).unsqueeze(0))[0]
            r2c = np.eye(4)
            r2c[:3,:3] = r2c_r
            r2c[:3,3] = r2c_t
            r2c = np.linalg.inv(r2c)
            g2c = np.dot(r2c, g2r)
            
            '''
            model_camspace = transform_pointcloud(self.model_pc, g2c)
            model_camspace = np.dot(model_camspace, camera_K.T)
            model_camspace = model_camspace[:,:2] / model_camspace[:,2:]
            model_camspace = model_camspace.astype(np.int32)
            max_x = min(np.max(model_camspace[:,0]) + 20, 640)
            max_y = min(np.max(model_camspace[:,1]) + 20, 480)
            min_x = max(np.min(model_camspace[:,0]) - 20, 0)
            min_y = max(np.min(model_camspace[:,1]) - 20, 0)
            mask[min_y:max_y, min_x:max_x] = 1
            '''
            
            
            rgbs.append(rgb)
            masks.append(mask)
            depths.append(depth)
            g2cs.append(g2c)
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        depths = np.stack(depths)
        g2cs = np.stack(g2cs)
        sample = {
            "rgb": rgbs,
            "depth": depths,
            "mask": masks, # TODO: Implement mask
            "pose": g2cs,
            "intrinsics": camera_intrinsics,
        }
        return sample
        
    def __len__(self):
        return len(self.videos)
        