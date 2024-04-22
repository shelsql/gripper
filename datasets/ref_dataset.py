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
                 num_views=64,
                 features=23
                 ):
        super().__init__()
        print("Loading reference view dataset...")
        self.N = num_views
        self.features = features
        self.dataset_location = dataset_location
        self.rgb_paths = glob.glob(dataset_location + "/*png")
        self.camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
        print("Found %d views in %s" % (len(self.rgb_paths), self.dataset_location))
    
        
    def __getitem__(self, index):
        
        rgbs = []
        depths = []
        masks = []
        c2ws = []
        obj_poses = []
        if self.features > 0:
            feats = []
        
        camera_intrinsic = json.loads(open(self.camera_intrinsic_path).read())
        
        for glob_rgb_path in self.rgb_paths:
            path = glob_rgb_path[:-8]
        
            rgb_path = path + "_rgb.png"
            depth_path = path + "_depth1.exr"
            mask_path = path + "_id1.exr"
            c2w_path = path + "_c2w.npy"
            obj_pose_path = path + "_objpose.npy"
            
            #print(rgb_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            c2w = np.load(c2w_path)
            obj_pose = np.load(obj_pose_path)
            if self.features > 0:
                feat_path = path + "_feats_%.2d.npy" % self.features
                feat = np.load(feat_path)
                feats.append(feat)
            
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            c2ws.append(c2w)
            obj_poses.append(obj_pose)
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        c2ws = np.stack(c2ws, axis = 0)
        obj_poses = np.stack(obj_poses, axis = 0)
        if self.features > 0:
            feats = np.stack(feats, axis = 0)
        else:
            feats = None
        #print(depths.shape)
        sample = {
            "rgbs": rgbs,
            "depths": depths,
            "masks": masks,
            "c2ws": c2ws,
            "obj_poses": obj_poses,
            "feats": feats,
            "intrinsics": camera_intrinsic
        }
        
        return sample
    def __len__(self):
        return 1
    
    
class SimTestDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/code/gripper/test_views/franka_69.4_64",
                 use_augs=False,
                 features=23
                 ):
        super().__init__()
        print("Loading reference view dataset...")
        self.dataset_location = dataset_location
        self.features = features
        self.rgb_paths = glob.glob(dataset_location + "/*png")
        self.camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
        print("Found %d views in %s" % (len(self.rgb_paths), self.dataset_location))
    
        
    def __getitem__(self, index):
        
        camera_intrinsic = json.loads(open(self.camera_intrinsic_path).read())
        
        path = self.rgb_paths[index][:-8]
    
        rgb_path = path + "_rgb.png"
        depth_path = path + "_depth1.exr"
        mask_path = path + "_id1.exr"
        c2w_path = path + "_c2w.npy"
        obj_pose_path = path + "_objpose.npy"
        
        #print(rgb_path)
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
        c2w = np.load(c2w_path)
        obj_pose = np.load(obj_pose_path)
        if self.features > 0:
            feat_path = path + "_feats_%.2d.npy" % self.features
            feat = np.load(feat_path)
            
        #print(depths.shape)
        sample = {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "c2w": c2w,
            "obj_pose": obj_pose,
            "feat": feat,
            "intrinsics": camera_intrinsic
        }
        
        return sample
    def __len__(self):
        return len(self.rgb_paths)
    

class SimTrackDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/code/gripper/test_views/franka_69.4_1024",
                 use_augs=False,
                 seqlen=64,
                 features=23
                 ):
        super().__init__()
        print("Loading reference view dataset...")
        self.S = seqlen
        self.features = features
        self.dataset_location = dataset_location
        self.rgb_paths = glob.glob(dataset_location + "/*png")
        self.camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
        
        total_frames = len(self.rgb_paths)
        print("Found %d frames in %s" % (total_frames, self.dataset_location))
        self.all_full_idx = []
        for ii in range(0, max(total_frames-self.S+1,1), self.S):
            # print('bbox_areas[%d]' % ii, bbox_areas[ii])
            full_idx = ii + np.arange(self.S)
            full_idx = [ij for ij in full_idx if ij < total_frames]
            if len(full_idx) == self.S:
                self.all_full_idx.append(full_idx)
        print("Found %d clips of length %d in %s" % (len(self.all_full_idx), self.S, self.dataset_location))
    
        
    def __getitem__(self, index):
        print('dataid',index)
        full_idx = self.all_full_idx[index]
        glob_paths = [self.rgb_paths[i] for i in full_idx]
        rgbs = []
        depths = []
        masks = []
        c2ws = []
        obj_poses = []
        if self.features > 0:
            feats = []
        
        camera_intrinsic = json.loads(open(self.camera_intrinsic_path).read())
        
        for glob_rgb_path in glob_paths:
            path = glob_rgb_path[:-8]
        
            rgb_path = path + "_rgb.png"
            depth_path = path + "_depth1.exr"
            mask_path = path + "_id1.exr"
            c2w_path = path + "_c2w.npy"
            obj_pose_path = path + "_objpose.npy"
            
            #print(rgb_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            c2w = np.load(c2w_path)
            obj_pose = np.load(obj_pose_path)
            if self.features > 0:
                feat_path = path + "_feats_%.2d.npy" % self.features
                feat = np.load(feat_path)
                feats.append(feat)
            
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            c2ws.append(c2w)
            obj_poses.append(obj_pose)
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        c2ws = np.stack(c2ws, axis = 0)
        obj_poses = np.stack(obj_poses, axis = 0)
        if self.features > 0:
            feats = np.stack(feats, axis = 0)
        else:
            feats = None
        #print(depths.shape)
        sample = {
            "rgb": rgbs,
            "depth": depths,
            "mask": masks,
            "c2w": c2ws,
            "obj_pose": obj_poses,
            "feat": feats,
            "intrinsics": camera_intrinsic
        }
        
        return sample
    def __len__(self):
        return len(self.all_full_idx)
    

class SimLargeDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/code/render/final_20240419",
                 features=23
                 ):
        super().__init__()
        print("Loading rendered dataset...")
        self.features = features
        self.dataset_location = dataset_location
        #self.subdirs = glob.glob(dataset_location + "/*")
        #Change back to this later
        self.subdirs = glob.glob(dataset_location + "/*panda")
        print("Found %d subdirs in %s" % (len(self.subdirs), self.dataset_location))
        self.videos = glob.glob(dataset_location + "/*panda/0*")
        
        print("Found %d videos in %s" % (len(self.videos), self.dataset_location))
        
    def __getitem__(self, index):
        print('dataid',index)
        video_path = self.videos[index]
        glob_paths = glob.glob(video_path + "/*rgb.png")
        rgbs = []
        depths = []
        masks = []
        c2ws = []
        obj_poses = []
        if self.features > 0:
            feats = []
        
        camera_intrinsic_path = video_path + "/camera_intrinsics.json"
        camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
        
        for glob_rgb_path in glob_paths:
            path = glob_rgb_path[:-8]
        
            rgb_path = path + "_rgb.png"
            depth_path = path + "_depth1.exr"
            mask_path = path + "_id1.exr"
            c2w_path = path + "_c2w.npy"
            obj_pose_path = path + "_objpose.npy"
            
            #print(rgb_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
            c2w = np.load(c2w_path)
            obj_pose = np.load(obj_pose_path)
            if self.features > 0:
                feat_path = path + "_feats_%.2d.npy" % self.features
                feat = np.load(feat_path)
                feats.append(feat)
            
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            c2ws.append(c2w)
            obj_poses.append(obj_pose)
        rgbs = np.stack(rgbs, axis = 0)
        depths = np.stack(depths, axis = 0)
        masks = np.stack(masks, axis = 0)
        c2ws = np.stack(c2ws, axis = 0)
        obj_poses = np.stack(obj_poses, axis = 0)
        if self.features > 0:
            feats = np.stack(feats, axis = 0)
        else:
            feats = None
        #print(depths.shape)
        sample = {
            "rgb": rgbs,
            "depth": depths,
            "mask": masks,
            "c2w": c2ws,
            "obj_pose": obj_poses,
            "feat": feats,
            "intrinsics": camera_intrinsic
        }
        
        return sample
    def __len__(self):
        return len(self.videos)
    