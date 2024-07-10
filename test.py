import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
import pickle

import torch
import time
import numpy as np


# def depth_map_to_pointcloud_tensor(depth_map, mask, intrinsics,x,y):
#     start_time = time.time()
#     H, W = depth_map.shape
#
#     # Unpack intrinsic matrix
#     fx = intrinsics['fx']
#     fy = intrinsics['fy']
#     cx = intrinsics['cx']
#     cy = intrinsics['cy']
#
#     time0 = time.time()
#     # Create grid of pixel coordinates
#
#     time1 = time.time()
#
#     # Apply mask
#     mask = mask.bool()
#     x_masked = x[mask]
#     y_masked = y[mask]
#     z_masked = depth_map[mask]
#
#     time2 = time.time()
#     # Convert pixel coordinates to camera coordinates
#     X = (x_masked - cx) * z_masked / fx
#     Y = (y_masked - cy) * z_masked / fy
#     Z = z_masked
#
#     time3 = time.time()
#     # Stack into point cloud
#     pointcloud = torch.stack((X, Y, Z), dim=-1)
#
#     end_time = time.time()
#
#     # Print timing
#     print('meshgrid time: {:.6f} seconds'.format(time1 - time0))
#     print('masking time: {:.6f} seconds'.format(time2 - time1))
#     print('conversion time: {:.6f} seconds'.format(time3 - time2))
#
#     print('time0-start {:.6f} seconds'.format(time0 - start_time))
#     print('end_time - time3: {:.6f} seconds'.format(end_time - time3))
#     print('total time: {:.6f} seconds'.format(end_time - start_time))
#     return pointcloud
#
#
# # Example usage
# for i in range(100):
#     depth_map = torch.randn(200, 300, device='cuda:1')
#     mask = torch.zeros(200, 300, dtype=torch.bool, device='cuda:1')
#     mask[110:160, 150:200] = 1  # Small valid region
#
#     intrinsics = {'fx': 500.0, 'fy': 500.0, 'cx': 150.0, 'cy': 100.0}
#     H, W = depth_map.shape
#     y, x = torch.meshgrid(torch.arange(H, device=mask.device), torch.arange(W, device=mask.device), indexing='ij')
#     pointcloud = depth_map_to_pointcloud_tensor(depth_map, mask, intrinsics,x,y)
#     time0 = time.time()
#     pointcloud = pointcloud.to('cuda:0')
#     time1 = time.time()


# print('move to gpu: {:.6f} seconds'.format(time1 - time0))

f_read = open('/home/data/tianshuwu/code/gripper/results/key16/panda_5_time_dick.pkl', 'rb')
dict2 = pickle.load(f_read)
move_data_to_gpu_time = np.array(dict2['move data to gpu time']).mean()
extract_feature_time = np.array(dict2['extract feature time']).mean()
select_refs_time = np.array(dict2['select ref time']).mean()
match_time = np.array(dict2['match time']).mean()
pnp_time = np.array(dict2['pnp time']).mean()
prepare_time = np.array(dict2['prepare time']).mean()
ceres_time = np.array(dict2['ceres time']).mean()
total_time = np.array(dict2['total time']).mean()
mask_time = np.array(dict2['mask time']).mean()
print('move_data_to_gpu_time',move_data_to_gpu_time)
print('extract_feature_time',extract_feature_time)
print('mask_time',mask_time)
print('select_refs_time',select_refs_time)
print('match_time',match_time)
print('pnp_time',pnp_time)
print('prepare_time',prepare_time)
print('ceres_time',ceres_time)
print('total_time',total_time)
run_model_time = np.array(dict2['run model time']).mean()
interpolate_time = np.array(dict2['interpolate time']).mean()
create3dmesh_time = np.array(dict2['3dmesh time']).mean()


print('run model time',run_model_time)
print('interpolate_time',interpolate_time)
print('create3dmesh_time',create3dmesh_time)

print(total_time -ceres_time-prepare_time-pnp_time-match_time-select_refs_time-extract_feature_time-move_data_to_gpu_time)



