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
from matcher import Dinov2Matcher
from utils.spd import get_2dbboxes
import torch.nn.functional as F

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

dataset_location = "./render_64views"
rgb_paths = glob.glob(dataset_location + "/*png")
camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"


rgbs = []
depths = []
masks = []
c2ws = []
obj_poses = []

camera_intrinsic = json.loads(open(camera_intrinsic_path).read())

for glob_rgb_path in tqdm(rgb_paths):
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
    obj_pose = np.linalg.inv(obj_pose)
    
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

#print(rgbs.shape, depths.shape, masks.shape, c2ws.shape, obj_poses.shape)
rgbs = torch.tensor(rgbs).float().permute(0, 3, 1, 2) # N, 3, H, W
depths = torch.tensor(depths).float().permute(0, 3, 1, 2) # N, 1, H, W
masks = torch.tensor(masks).float().permute(0, 3, 1, 2) # N, 1, H, W
c2ws = torch.tensor(c2ws).float() # N, 4, 4
w2os = torch.tensor(obj_poses).float()

cam_to_obj = torch.bmm(w2os, c2ws) # N, 4, 4

N = rgbs.shape[0]
device = "cuda:0"
img_size = 448

print(cam_to_obj.shape)


transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])

rgbs = rgbs / 255.0 # N, 3, H, W
bboxes = get_2dbboxes(masks[:,0]) # B, 4
cropped_rgbs = torch.zeros((N, 3, img_size, img_size), device = device)
cropped_masks = torch.zeros((N, 1, img_size, img_size), device = device)
for b in range(N):
    y_min, x_min, y_max, x_max = bboxes[b,0], bboxes[b,1], bboxes[b,2], bboxes[b,3]
    cropped_rgb = rgbs[b:b+1, :, y_min:y_max, x_min:x_max]
    cropped_mask = masks[b:b+1, :, y_min:y_max, x_min:x_max]
    cropped_rgb = F.interpolate(cropped_rgb, size=(img_size, img_size), mode="bilinear")
    cropped_mask = F.interpolate(cropped_mask, size=(img_size, img_size), mode="nearest")
    cropped_rgbs[b:b+1] = cropped_rgb
    cropped_masks[b:b+1] = cropped_mask
cropped_rgbs = transform(cropped_rgbs)
cropped_rgbs = cropped_rgbs.to(device)
bboxes = torch.tensor(bboxes, device = device)

print(cropped_rgbs.shape, cropped_masks.shape, bboxes.shape)