import os
import numpy as np
from tqdm import tqdm
import shutil
list_name_list = ['list_40x6.txt','list_80x12.txt','list_80x24.txt']
ref_name_list = ['ref_240','ref_960','ref_1920']
for i in range (len(list_name_list)):
    for gripper_name in ['kinova','robotiq2f140','robotiq2f85','robotiq3f','shadowhand']:
        with open(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/{list_name_list[i]}') as f:
            if not os.path.exists(f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}'):
                os.makedirs(f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}')
            data_list = [int(line.strip('\n')) for line in f.readlines()]
            for data_id in tqdm(data_list):
                shutil.copy(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/{str(data_id).zfill(6)}_c2w.npy',
                            f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}/{str(data_id).zfill(6)}_c2w.npy')
                shutil.copy(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/{str(data_id).zfill(6)}_depth1.exr',
                            f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}/{str(data_id).zfill(6)}_depth1.exr')
                shutil.copy(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/{str(data_id).zfill(6)}_id1.exr',
                            f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}/{str(data_id).zfill(6)}_id1.exr')
                shutil.copy(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/{str(data_id).zfill(6)}_objpose.npy',
                            f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}/{str(data_id).zfill(6)}_objpose.npy')
                shutil.copy(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/{str(data_id).zfill(6)}_rgb.png',
                            f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}/{str(data_id).zfill(6)}_rgb.png')
            shutil.copy(f'/root/autodl-tmp/shiqian/code/render/refs_320x24/{gripper_name}/camera_intrinsics.json',
                        f'/root/autodl-tmp/tianshuwu/data/{ref_name_list[i]}/{gripper_name}/camera_intrinsics.json')