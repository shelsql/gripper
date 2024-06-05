import argparse
import sys
import os
import json
import glob
import time

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from utils.spd import get_2dbboxes,depth_map_to_pointcloud, create_3dmeshgrid,transform_batch_pointcloud_torch,calculate_2d_projections,depth_map_to_pointcloud_tensor
import torch.nn.functional as F
from models.uni3d import create_uni3d

# import hickle as hkl

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def idx_to_2d_coords(idxs, bboxes):
    N, D = idxs.shape  # imgidx, feat x, feat y
    assert (D == 3)
    coords = torch.zeros_like(idxs)
    coords[:, 0] = idxs[:, 0]
    batch_bboxes = bboxes[idxs[:, 0]]
    # Turn token idx into coord within 448*448 box
    coords[:, 1:3] = idxs[:, 1:3] * 14 + 14 / 2
    # Turn coord within 448*448 box to coord on full image
    coords[:, 1:3] = (coords[:, 1:3] / 448.0 * (batch_bboxes[:, 2:4] - batch_bboxes[:, 0:2])) + batch_bboxes[:, 0:2]
    return coords

def coords_2d_to_3d(coords, depth_maps, intrinsics, c2os):
    N, D = coords.shape
    assert (D == 3)

    # Unpack intrinsic matrix
    fx = intrinsics['fx']#.item()
    fy = intrinsics['fy']#.item()
    cx = intrinsics['cx']#.item()
    cy = intrinsics['cy']#.item()

    # print("depth_maps", depth_maps.shape)
    depths = depth_maps[coords[:, 0].int(), coords[:, 1].int(), coords[:, 2].int()]
    # print("depths", depths.shape)
    cam_space_x = (coords[:, 2] - cx) * depths / fx
    cam_space_y = (coords[:, 1] - cy) * depths / fy
    cam_space_z = depths
    cam_space_coords = torch.stack([cam_space_x, cam_space_y, cam_space_z], axis=1)  # N, 3

    # c2os = c2os[coords[:, 0].int()]
    # print("c2os:", c2os.shape)
    # obj_space_coords = transform_batch_pointcloud_torch(cam_space_coords, c2os)
    # print(obj_space_coords.shape)

    ref_ids = coords[:, 0:1]
    ids_and_coords_3d = torch.concat([ref_ids, cam_space_coords], axis=1)

    return ids_and_coords_3d

def save_features(
    dataset_location = "./ref_views/powerdrill_69.4_840",
    layer = 23,
    device = "cuda:0",
    use_3d_feature=True,
    get_2d_feature=True,
        uni3d_layer=23,
uni3d_use_color=False
):
    print("Saving features from layer %d to %s" % (layer, dataset_location))
    print("Using device %s" % device)

    camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
    camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
    intrinsic_matrix = torch.zeros((3, 3),device=device)
    intrinsic_matrix[0, 0] = camera_intrinsic['fx']
    intrinsic_matrix[1, 1] = camera_intrinsic['fy']
    intrinsic_matrix[0, 2] = camera_intrinsic['cx']
    intrinsic_matrix[1, 2] = camera_intrinsic['cy']
    intrinsic_matrix[2, 2] = 1

    rgb_paths = glob.glob(dataset_location + "/00[01]*png")

    rgbs = []
    masks = []
    depths = []
    for glob_rgb_path in tqdm(rgb_paths):
        path = glob_rgb_path[:-7]

        rgb_path = path + "rgb.png"
        mask_path = path + "id1.exr"
        depth_path = path + "depth1.exr"

        #print(rgb_path)
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0:1]
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:1]
        #mask = (mask >= 9).astype(int)
        
        rgbs.append(rgb)
        masks.append(mask)
        depths.append(depth)
    rgbs = np.stack(rgbs, axis = 0)
    masks = np.stack(masks, axis = 0)
    depths = np.stack(depths, axis=0)

    #print(rgbs.shape, depths.shape, masks.shape, c2ws.shape, obj_poses.shape)
    rgbs = torch.tensor(rgbs).float().permute(0, 3, 1, 2) # N, 3, H, W
    masks = torch.tensor(masks).float().permute(0, 3, 1, 2) # N, 1, H, W
    depths = torch.tensor(depths).float().permute(0, 3, 1, 2)
    rgbs[masks.repeat(1, 3, 1, 1) == 0] = 0
    N, C, H, W = rgbs.shape
    img_size = 448

    transform = transforms.Compose([
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])

    rgbs = rgbs / 255.0 # N, 3, H, W
    bboxes = get_2dbboxes(masks[:,0]) # B, 4
    #print(bboxes[:10])
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
    # cropped_rgbs[cropped_masks.repeat(1,3,1,1) == 0] = 0 # Mask out background 在外面细粒度的mask过了
    bboxes = torch.tensor(bboxes, device = device)

    #rgb_0 = cropped_rgbs[20].permute(1, 2, 0).cpu().numpy()
    #rgb_0 = (rgb_0 - np.min(rgb_0)) / (np.max(rgb_0) - np.min(rgb_0))
    #rgb_0 *= 255.0
    #cv2.imwrite("match_vis/masked_rgb.png", rgb_0)

    print(cropped_rgbs.shape, cropped_masks.shape, bboxes.shape)





    if use_3d_feature:
        with torch.inference_mode():
            for i in tqdm(range(rgbs.shape[0])):


                # # step1 先从深度图还原点云，获得512个feature及该group中心坐标
                # 点云要先中心化，但是投影回去的时候不要忘了再把中心位置加回去，不然patch找自己的geo feature时会找歪         重要！！

                depth_tmp = depths[i][0].to(device)
                mask_tmp = masks[i][0].to(device)
                time0 = time.time()
                point_cloud = depth_map_to_pointcloud_tensor(depth_tmp,mask_tmp, intrinsic_matrix)

                time1 = time.time()
                # print('map_depth_time:',time1-time0)
                point_center = point_cloud.mean(0)
                point_cloud = point_cloud - point_center
                if uni3d_use_color:
                    rgb = rgbs[i][:,masks[i][0].bool()].transpose(-1,-2).to(device)
                else:
                    rgb = torch.ones_like(point_cloud)

                geo_feat, geo_center = uni3d.encode_pc(torch.cat([point_cloud.unsqueeze(0), rgb.unsqueeze(0)], dim=-1), layer=uni3d_layer)

                geo_feat = geo_feat / geo_feat.norm(dim=-1,keepdim=True)        # cat之前都除以norm
                geo_center = geo_center + point_center

                projected_coordinates = intrinsic_matrix @ geo_center[0].transpose(-1,-2)
                projected_coordinates = (projected_coordinates[:2, :] / projected_coordinates[2, :]).transpose(-1,-2)
                proj_2d = torch.zeros_like(projected_coordinates)
                proj_2d[:,0] = projected_coordinates[:,1]
                proj_2d[:, 1] = projected_coordinates[:, 0]

                # step2 把512个点云投到2d，每个patch直接用最近的那个作为自己的geo feature（为mask之内的点找到最近的那个点云，把它的feature拿来用，如果直接这么用的话，可能没落在mask内部；所以得取该patch中的所有mask内部的点的点云feature的均值？）
                test_idx = create_3dmeshgrid(1, 32, 32, device)
                batch_feat_masks = F.interpolate(cropped_masks[i:i+1], size=(32, 32), mode="nearest")   # 在32，32的mask
                test_idx = test_idx[batch_feat_masks[:, 0] > 0]  # 287?,3 这个是在32*32中的坐标
                test_2d_coords = idx_to_2d_coords(test_idx, bboxes[i:i+1])  # 这个是在360,640中的坐标

                dis = torch.norm(test_2d_coords[:,1:].unsqueeze(1)-proj_2d.unsqueeze(0) ,dim=-1)    # 307?,512
                cor_id = dis.min(dim=1).indices # 307？,范围为（0~512）

                geo_feat_image = torch.zeros((1024,32,32),device=device)
                geo_feat_image[:,test_idx[:,1],test_idx[:,2]] = geo_feat[0,1:,:][cor_id].transpose(-1,-2)   # 1024,32,32 mask之外是全0
                path = rgb_paths[i][:-7]
                if uni3d_use_color:
                    uni3d_path = path + f"feats_uni3d{uni3d_layer}_colored.npy"
                else:
                    uni3d_path = path + f"feats_uni3d{uni3d_layer}_nocolor.npy"
                # np.save(uni3d_path, geo_feat_image.cpu().numpy())





                image_batch = cropped_rgbs[i:i+1]
                tokens = dinov2.get_intermediate_layers(image_batch, n=[layer])[0]
                B, N_tokens, C = tokens.shape
                assert (N_tokens == img_size * img_size / 196)
                tokens = tokens.permute(0, 2, 1).reshape(B, C, img_size // 14, img_size // 14)
                tokens = tokens / tokens.norm(dim=1, keepdim=True)      # 注意这里是新加的，cat之前要都除以norm



                dino_path = path + f"feats_dino{layer}.npy"
                np.save(dino_path,tokens[0].cpu().numpy())







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
# for data_dir in glob.glob("/root/autodl-tmp/shiqian/datasets/reference_views/*"):

device="cuda:0"
dinov2 = torch.hub.load(repo_or_dir="../dinov2", source="local", model="dinov2_vitl14_reg", pretrained=False)
dinov2.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
dinov2.to(device)
dinov2.eval()

uni3d = create_uni3d().to(device)
checkpoint = torch.load("checkpoints/model_large.pt", map_location='cpu')
sd = checkpoint['module']
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
uni3d.load_state_dict(sd)

# dino_layer = 19
# uni3d_layer = 19
uni3d_use_color = False

parser = argparse.ArgumentParser()
parser.add_argument('--dino_layer', default=19,type=int)
parser.add_argument('--uni3d_layer', default=19,type=int)
cfg = parser.parse_args()
gripper_list = ['panda','kinova','robotiq2f140','robotiq2f85','robotiq3f','shadowhand'] # #,
dino_layer = cfg.dino_layer
uni3d_layer = cfg.uni3d_layer
for gripper in gripper_list:
    save_features(f'/home/data/tianshuwu/data/ref_1920/{gripper}', dino_layer, device,use_3d_feature=True,get_2d_feature=True, uni3d_layer =uni3d_layer,uni3d_use_color=uni3d_use_color)
# for gripper in gripper_list:
#     save_features(f'/root/autodl-tmp/shiqian/code/render/reference_more/{gripper}', 19, "cuda:0",use_3d_feature=True,get_2d_feature=True, uni3d_layer =uni3d_layer,uni3d_use_color=False)
# path = '/root/autodl-tmp/shiqian/datasets/final_20240419/panda'
# subdirs = glob.glob(path + "/videos/*")
# videos = []
# for subdir in subdirs:
#     videos.extend(sorted(glob.glob(subdir + "/0*"))[:10])
# for video in videos:
#     save_features(video, 19, "cuda:0",use_3d_feature=True,get_2d_feature=True,uni3d_layer =uni3d_layer)
