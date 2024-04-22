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

import hickle as hkl

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def save_features(
    dataset_location = "./ref_views/powerdrill_69.4_840",
    layer = 23,
    device = "cuda:0"
):
    print("Saving features from layer %d to %s" % (layer, dataset_location))
    print("Using device %s" % device)
    rgb_paths = glob.glob(dataset_location + "/*png")

    rgbs = []
    masks = []
    for glob_rgb_path in tqdm(rgb_paths):
        path = glob_rgb_path[:-7]

        rgb_path = path + "rgb.png"
        mask_path = path + "id1.exr"
        
        #print(rgb_path)
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
        #mask = (mask >= 9).astype(int)
        
        rgbs.append(rgb)
        masks.append(mask)
    rgbs = np.stack(rgbs, axis = 0)
    masks = np.stack(masks, axis = 0)

    #print(rgbs.shape, depths.shape, masks.shape, c2ws.shape, obj_poses.shape)
    rgbs = torch.tensor(rgbs).float().permute(0, 3, 1, 2) # N, 3, H, W
    masks = torch.tensor(masks).float().permute(0, 3, 1, 2) # N, 1, H, W

    N, C, H, W = rgbs.shape
    img_size = 448

    transform = transforms.Compose([
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])

    rgbs = rgbs / 255.0 # N, 3, H, W
    bboxes = get_2dbboxes(masks[:,0]) # B, 4
    print(bboxes[:10])
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
    cropped_rgbs[cropped_masks.repeat(1,3,1,1) == 0] = 0 # Mask out background
    bboxes = torch.tensor(bboxes, device = device)

    #rgb_0 = cropped_rgbs[20].permute(1, 2, 0).cpu().numpy()
    #rgb_0 = (rgb_0 - np.min(rgb_0)) / (np.max(rgb_0) - np.min(rgb_0))
    #rgb_0 *= 255.0
    #cv2.imwrite("match_vis/masked_rgb.png", rgb_0)

    print(cropped_rgbs.shape, cropped_masks.shape, bboxes.shape)

    dinov2 = torch.hub.load(repo_or_dir="../dinov2",source="local", model="dinov2_vitl14_reg", pretrained=False)
    dinov2.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
    dinov2.to(device)
    dinov2.eval()

    batch_size = 128
    num_batches = (N + batch_size - 1) // batch_size
    all_tokens = torch.zeros((N, 1024, 32, 32), device = device)

    with torch.inference_mode():
        for i in tqdm(range(num_batches)):
            start_idx = i*batch_size
            end_idx = min(N, (i+1)*batch_size)
            image_batch = cropped_rgbs[start_idx:end_idx]
            tokens = dinov2.get_intermediate_layers(image_batch, n=[layer])[0]
            B, N_tokens, C = tokens.shape
            assert(N_tokens == img_size*img_size / 196)
            tokens = tokens.permute(0,2,1).reshape(B, C, img_size//14, img_size//14)
            all_tokens[start_idx:end_idx] = tokens
        #print(tokens.shape)
        #print(tokens.max(), tokens.min(), tokens.mean())
        
    print(all_tokens.shape)
    print("Saving features...")
    for i in tqdm(range(N)):
        path = rgb_paths[i][:-7]
        feat_path = path + "feats_%.2d.npy" % layer
        np.save(feat_path, all_tokens[i].cpu().numpy())

    #torch.save(data_dict, "./dataset_exports/ref_840.pth")
    #hkl.dump(data_dict, "./dataset_exports/ref_64.hkl")
'''
layers = [11, 15]
data_dirs = ["./ref_views/powerdrill_69.4_840",
                "./ref_views/franka_69.4_840",
                "./test_views/powerdrill_69.4_1024",
                "./test_views/franka_69.4_1024",]
device = "cuda:0"
for data_dir in data_dirs:
    for layer in layers:
        save_features(data_dir, layer, device)
'''
for data_dir in glob.glob("/root/autodl-tmp/shiqian/code/render/final_20240419/*panda/0*"):
    save_features(data_dir, 19, "cuda:2")