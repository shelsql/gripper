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
from utils.spd import get_2dbboxes, depth_map_to_pointcloud, create_3dmeshgrid, transform_batch_pointcloud_torch, \
    calculate_2d_projections, depth_map_to_pointcloud_tensor
import torch.nn.functional as F


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
    fx = intrinsics['fx']  # .item()
    fy = intrinsics['fy']  # .item()
    cx = intrinsics['cx']  # .item()
    cy = intrinsics['cy']  # .item()

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
        # dataset_location="./ref_views/powerdrill_69.4_840",
        # camera_intrinsic_path,
rgb_path,
mask_path,
# depth_path,
save_folder,
        layer=23,
        device="cuda:0",
        use_3d_feature=True,
        get_2d_feature=True,
        uni3d_layer=23,
        uni3d_use_color=False,
        real=False
):
    # if not real:
    #     camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
    #     intrinsic_matrix = torch.zeros((3, 3), device=device)
    #     intrinsic_matrix[0, 0] = camera_intrinsic['fx']
    #     intrinsic_matrix[1, 1] = camera_intrinsic['fy']
    #     intrinsic_matrix[0, 2] = camera_intrinsic['cx']
    #     intrinsic_matrix[1, 2] = camera_intrinsic['cy']
    #     intrinsic_matrix[2, 2] = 1
    # else:
    #     pass
    # rgb_paths = glob.glob(dataset_location + "/*png")

    rgbs = []
    masks = []
    # depths = []



    # print(rgb_path)
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:1]
    # depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :] # real和sim不一样
    mask = (mask >= 0.5).astype(int)

    rgbs.append(rgb)
    masks.append(mask)
    # depths.append(depth)
    rgbs = np.stack(rgbs, axis=0)
    masks = np.stack(masks, axis=0)
    # depths = np.stack(depths, axis=0)

    # print(rgbs.shape, depths.shape, masks.shape, c2ws.shape, obj_poses.shape)
    rgbs = torch.tensor(rgbs).float().permute(0, 3, 1, 2)  # N, 3, H, W
    masks = torch.tensor(masks).float().permute(0, 3, 1, 2)  # N, 1, H, W
    bboxes = get_2dbboxes(masks[:, 0])  # B, 4

    y_min, x_min, y_max, x_max = bboxes[0, 0], bboxes[0, 1], bboxes[0, 2], bboxes[0, 3]

    cropped_rgb = rgbs[0:0 + 1, :, y_min:y_max, x_min:x_max]
    image_array = np.transpose(cropped_rgb[0].cpu().numpy(), (1, 2, 0))
    # 将值缩放到0-255并转换为uint8类型
    image_array = (image_array).astype(np.uint8)
    # 创建Image对象
    image = Image.fromarray(image_array)
    image.save(f'{save_folder}/image_withground.png')

    # depths = torch.tensor(depths).unsqueeze(-1).float().permute(0, 3, 1, 2)
    rgbs[masks.repeat(1, 3, 1, 1) == 0] = 0
    N, C, H, W = rgbs.shape
    img_size = 448

    transform = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
    ])

    rgbs = rgbs / 255.0  # N, 3, H, W

    # print(bboxes[:10])
    cropped_rgbs = torch.zeros((N, 3, img_size, img_size), device=device)
    cropped_masks = torch.zeros((N, 1, img_size, img_size), device=device)




    for b in range(N):
        y_min, x_min, y_max, x_max = bboxes[b, 0], bboxes[b, 1], bboxes[b, 2], bboxes[b, 3]

        cropped_rgb = rgbs[b:b + 1, :, y_min:y_max, x_min:x_max]
        cropped_mask = masks[b:b + 1, :, y_min:y_max, x_min:x_max]
        cropped_rgb = F.interpolate(cropped_rgb, size=(img_size, img_size), mode="bilinear")
        cropped_mask = F.interpolate(cropped_mask, size=(img_size, img_size), mode="nearest")
        cropped_rgbs[b:b + 1] = cropped_rgb
        cropped_masks[b:b + 1] = cropped_mask


    np.save(f'{save_folder}/cropped_rgb.npy', cropped_rgbs[0].cpu().numpy())

    image_array = np.transpose(cropped_rgbs[0].cpu().numpy(), (1, 2, 0))
    # 将值缩放到0-255并转换为uint8类型
    image_array = (image_array * 255).astype(np.uint8)
    # 创建Image对象
    image = Image.fromarray(image_array)
    image.save(f'{save_folder}/image_noground.png')


    np.save(f'{save_folder}/cropped_mask.npy', cropped_masks[0].cpu().numpy())
    np.save(f'{save_folder}/bbox.npy', bboxes[0])

    cropped_rgbs = transform(cropped_rgbs)
    cropped_rgbs = cropped_rgbs.to(device)
    # cropped_rgbs[cropped_masks.repeat(1,3,1,1) == 0] = 0 # Mask out background 在外面细粒度的mask过了
    bboxes = torch.tensor(bboxes, device=device)



    # rgb_0 = cropped_rgbs[20].permute(1, 2, 0).cpu().numpy()
    # rgb_0 = (rgb_0 - np.min(rgb_0)) / (np.max(rgb_0) - np.min(rgb_0))
    # rgb_0 *= 255.0
    # cv2.imwrite("match_vis/masked_rgb.png", rgb_0)

    print(cropped_rgbs.shape, cropped_masks.shape, bboxes.shape)


    with torch.inference_mode():
        for i in tqdm(range(rgbs.shape[0])):


            image_batch = cropped_rgbs[i:i + 1]
            tokens = dinov2.get_intermediate_layers(image_batch, n=[layer])[0]
            B, N_tokens, C = tokens.shape
            assert (N_tokens == img_size * img_size / 196)
            tokens = tokens.permute(0, 2, 1).reshape(B, C, img_size // 14, img_size // 14)
            tokens = tokens / tokens.norm(dim=1, keepdim=True)  # 注意这里是新加的，cat之前要都除以norm

            np.save(f'{save_folder}/feature{layer}.npy', tokens[0].cpu().numpy())

    # torch.save(data_dict, "./dataset_exports/ref_840.pth")
    # hkl.dump(data_dict, "./dataset_exports/ref_64.hkl")


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

device = "cuda:0"
dinov2 = torch.hub.load(repo_or_dir="../dinov2", source="local", model="dinov2_vitl14_reg", pretrained=False)
dinov2.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
dinov2.to(device)
dinov2.eval()



# dino_layer = 19
# uni3d_layer = 19
uni3d_use_color = False

parser = argparse.ArgumentParser()
parser.add_argument('--dino_layer', default=15, type=int)
parser.add_argument('--uni3d_layer', default=-1, type=int)
cfg = parser.parse_args()
gripper_list = ['panda']  # #,
dino_layer = cfg.dino_layer
uni3d_layer = cfg.uni3d_layer
for gripper in gripper_list:
    save_features(
        # camera_intrinsic_path='/home/data/tianshuwu/data/ref_960/panda/camera_intrinsics.json',
                  rgb_path='/home/data/tianshuwu/data/ref_960/panda/003268_rgb.png',
                  mask_path='/home/data/tianshuwu/data/ref_960/panda/003268_id1.exr',
                  # depth_path='/home/data/tianshuwu/data/Ty_data/6_D415_left_1/000448.exr',
                  save_folder='/home/data/tianshuwu/code/gripper/tmp',

                  layer=dino_layer, device=device, use_3d_feature=True,
                  get_2d_feature=True, uni3d_layer=uni3d_layer, uni3d_use_color=uni3d_use_color)