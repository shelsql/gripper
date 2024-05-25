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
                 features=23
                 ):
        super().__init__()
        print("Loading pose estimation dataset...")
        self.dataset_location = dataset_location
        self.features = features
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
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,np.newaxis]
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
        mask = (mask >= 9).astype(int)
        kpts = json.loads(open(kpts_path).read())
        npy = np.load(npy_path)
        
        depth /= 1000.0
        
        gripper_info = kpts[0]['keypoints'][8]
        gripper_t = torch.tensor(gripper_info["location_wrt_cam"]).numpy()
        gripper_r = torch.tensor(gripper_info["R2C_mat"]).numpy()
        gripper_rt = np.zeros((4,4))
        gripper_rt[:3, :3] = gripper_r
        gripper_rt[:3, 3] = gripper_t
        gripper_rt[3, 3] = 1  # obj2cam
        c2w = np.linalg.inv(gripper_rt)
        obj_pose = np.eye(4)        
        sample = {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "kpts": kpts,
            "npy": npy,
            "c2w": c2w,
            "obj_pose": obj_pose,
            "intrinsics": intrinsics
        }
        return sample
    
    def __len__(self):
        return len(self.all_paths)
        

class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/datasets/Ty_data",
                 seqlen=32,
                 features=19
                 ):
        super().__init__()
        print("Loading pose tracking dataset...")
        self.seqlen = seqlen
        self.features = features
        self.dataset_location = dataset_location
        self.video_dirs = glob.glob(dataset_location + "/*")
        print("Found %d videos in %s" % (len(self.video_dirs), self.dataset_location))
        
        self.img_paths = {}
        for video in self.video_dirs:
            self.img_paths[video] = [path[:-4] for path in sorted(glob.glob(video + "/*.png"))]
        
        self.all_videos = []
        for video in self.video_dirs:
            vid = {}
            vid["dir"] = video
            filelist = open("%s/camera_static_frames.txt" % video).readlines()
            vid["filelist"] = filelist
            
            self.all_videos.append(vid)
            
            
            
    def __getitem__(self, index):
        video = self.all_videos[index]
        video_dir = video["dir"]
        imgid_list = video["filelist"]
        camera_intrinsic_path = os.path.join(video_dir, "_camera_settings.json")
        camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
        camera_intrinsic = camera_intrinsic['camera_settings'][0]['intrinsic_settings']
        
        print(video_dir, imgid_list)
        
        rgbs = []
        depths = []
        masks = []
        kptss = []
        npys = []
        c2ws = []
        obj_poses = []
        if self.features > 0:
            feats = []
        for id in imgid_list:
            path = video_dir + "/" + id[:-1]
        
            rgb_path = path + ".png"
            depth_path = path + ".exr"
            mask_path = path + "_mask.exr"
            kpts_path = path + ".json"
            npy_path = path + ".npy"
            
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,np.newaxis]
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            mask = (mask >= 9).astype(int)
            kpts = json.loads(open(kpts_path).read())
            npy = np.load(npy_path)
            
            depth = depth / 1000.0
            
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            kptss.append(kpts)
            npys.append(npy)
            
            gripper_info = kpts[0]['keypoints'][8]
            gripper_t = torch.tensor(gripper_info["location_wrt_cam"]).numpy()
            gripper_r = torch.tensor(gripper_info["R2C_mat"]).numpy()
            gripper_rt = np.zeros((4,4))
            gripper_rt[:3, :3] = gripper_r
            gripper_rt[:3, 3] = gripper_t
            gripper_rt[3, 3] = 1  # obj2cam
            c2w = np.linalg.inv(gripper_rt)
            c2ws.append(c2w)
            obj_poses.append(np.eye(4))
            if self.features > 0:
                feat_path = path + "_feats_%.2d.npy" % self.features
                feat = np.load(feat_path)
                feats.append(feat)
            
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        npys = np.stack(npys, axis = 0)
        c2ws = np.stack(c2ws, axis = 0)
        obj_poses = np.stack(obj_poses, axis = 0)
        if self.features > 0:
            feats = np.stack(feats, axis = 0)
        else:
            feats = None
        print(rgbs.shape, depths.shape, masks.shape)
        sample = {
            "rgb": rgbs,
            "depth": depths,
            "mask": masks,
            "kpts": kptss,
            "npy": npys,
            "c2w": c2ws,
            "obj_pose": obj_poses,
            "feat": feats,
            "intrinsics": camera_intrinsic,
            "path":path
        }
        
        return sample
    def __len__(self):
        return len(self.video_dirs)