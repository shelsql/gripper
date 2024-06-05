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
            vid_clip["obj_ids_in_img"] = []
            scene_gt = json.loads(open(vid_dir + "/scene_gt.json").read())
            scene_cam = json.loads(open(vid_dir + "/scene_camera.json").read())
            for img_id in scene_gt.keys():
                frame_info = scene_gt[img_id]
                for i in range(len(frame_info)):
                    obj_info = frame_info[i]
                    if obj_info['obj_id'] == obj_id:
                        vid_clip["img_ids"].append(img_id)
                        o2c_pose = np.eye(4)
                        o2c_pose[:3,:3] = np.array(obj_info['cam_R_m2c']).reshape(3,3)
                        o2c_pose[:3,3] = np.array(obj_info['cam_t_m2c']) / 1000.0
                        vid_clip["gt_poses"].append(o2c_pose)
                        vid_clip["obj_ids_in_img"].append(i)
                        break
            if len(vid_clip["img_ids"]) > 0:
                self.all_video_clips.append(vid_clip)
        print('Found %d clips for object %d' % (len(self.all_video_clips), obj_id))
                
    def __getitem__(self, index):
        vid_clip = self.all_video_clips[index]
        vid_id = vid_clip["vid_id"]
        img_ids = vid_clip["img_ids"]
        gt_poses = vid_clip["gt_poses"]
        obj_ids_in_img = vid_clip["obj_ids_in_img"]
        rgb_dir = f"{self.dataset_location}/test/{vid_id}/rgb"
        depth_dir = f"{self.dataset_location}/test/{vid_id}/depth"
        mask_dir = f"{self.dataset_location}/test/{vid_id}/mask_visib"
        cam_intrinsic_list = json.loads(open(f"{self.dataset_location}/test/{vid_id}/scene_camera.json").read())
        cam_K = cam_intrinsic_list[list(cam_intrinsic_list.keys())[0]]['cam_K']
        intrinsics = {
            "fx": cam_K[0],
            "fy": cam_K[4],
            "cx": cam_K[2],
            "cy": cam_K[5],
        }
        print("Video id: %s  Obj id: %d" % (vid_id, self.obj_id))
        rgbs = []
        depths = []
        masks = []
        for i in range(len(img_ids)):
            img_id = int(img_ids[i])
            obj_id_in_img = obj_ids_in_img[i]
            rgb = cv2.cvtColor(cv2.imread(rgb_dir + "/%.6d.png" % img_id), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_dir + "/%.6d.png" % img_id, cv2.IMREAD_ANYDEPTH)[:,:,np.newaxis].astype(float) / 10000.0
            mask = cv2.imread(mask_dir + "/%.6d_%.6d.png" % (img_id, obj_id_in_img))[:,:,0:1]
            #print("rgb stats:", rgb.shape,rgb.dtype, rgb.max(), rgb.min(), rgb.mean())
            #print("depth stats:", depth.shape,depth.dtype, depth.max(), depth.min(), depth.mean())
            #print("mask stats:", mask.shape,mask.dtype, mask.max(), mask.min(), mask.mean())
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        gt_poses = np.stack(gt_poses, axis = 0)
        
        sample = {
            "rgb": rgbs,
            "depth": depths,
            "mask": masks,
            "pose": gt_poses,
            "intrinsics": intrinsics,
        }
        return sample
        
    def __len__(self):
        return len(self.all_video_clips)