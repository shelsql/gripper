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
import joblib

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class YCBVDataset(Dataset):
    def __init__(
        self,
        dataset_location="/root/autodl-tmp/shiqian/datasets/BOP/ycbv",
        obj_id = 1,
    ):
        self.dataset_location = dataset_location
        self.obj_id = obj_id
        self.video_dirs = glob.glob(dataset_location + "/test/*")
        print('Found %d videos in %s' % (len(self.video_dirs), dataset_location))
        self.all_video_clips = []
        for vid_dir in self.video_dirs:
            vid_clip = {}
            vid_clip["vid_id"] = vid_dir.split("/")[-1]
            vid_clip["img_ids"] = []
            vid_clip["gt_poses"] = []
            scene_gt = json.loads(open(vid_dir + "/scene_gt.json").read())
            scene_cam = json.loads(open(vid_dir + "/scene_camera.json").read())
            for img_id in scene_gt.keys():
                frame_info = scene_gt[img_id]
                for obj_info in frame_info:
                    if obj_info['obj_id'] == obj_id:
                        vid_clip["img_ids"].append(img_id)
                        o2c_pose = np.eye(4)
                        o2c_pose[:3,:3] = np.array(obj_info['cam_R_m2c']).reshape(3,3)
                        o2c_pose[:3,3] = np.array(obj_info['cam_t_m2c'])
                        vid_clip["gt_poses"].append(o2c_pose)
                        break
            if len(vid_clip["img_ids"]) > 0:
                self.all_video_clips.append(vid_clip)
                
    def __getitem__(self, index):
        vid_clip = self.all_video_clips[index]
        vid_id = vid_clip["vid_id"]
        img_ids = vid_clip["img_ids"]
        gt_poses = vid_clip["gt_poses"]
        rgb_dir = f"{self.dataset_location}/test/{vid_id}/rgb"
        depth_dir = f"{self.dataset_location}/test/{vid_id}/rgb"
        mask_dir = f"{self.dataset_location}/test/{vid_id}/rgb"
        return imgs, gt_poses
        
    def __len__(self):
        return len(self.all_video_clips)