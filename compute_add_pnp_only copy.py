import os
import pickle

import numpy as np
from utils.spd import read_pointcloud,transform_pointcloud, save_pointcloud
from utils.quaternion_utils import *
import torch
from utils.metrics import compute_auc_all
result_dict = {
    'add_10cm':[],
    'adds_10cm':[],
    'add_1cm':[],
    'adds_1cm':[]
}
results_total = []
# 优化时
results = np.loadtxt(f'results/tmp/15_pnponly.txt')   # 不是pnp only 时
results_total.append(results)

results_total = np.concatenate(results_total,axis=0)
#q_preds = results_total[:,:4]     # 1024,4
#t_preds = np.clip(np.nan_to_num(results_total[:,4:7],nan=100),-100,100)    # 1024,3
pred_poses = results_total[:,:16].reshape(-1,4,4)    # 1024,4,4
gt_poses = results_total[:,16:].reshape(-1,4,4)    # 1024,4,4

# pnp only 的
# results = np.loadtxt(f'results/{fold_name}/{gripper_name}_pnponly.txt')
# pred_poses = results[:,:16].reshape(-1,4,4)
# gt_poses = np.linalg.inv(results[:,16:].reshape(-1,4,4))
# q_preds = matrix_to_quaternion(torch.tensor(pred_poses[:,:3,:3])).numpy()
# t_preds = np.clip(np.nan_to_num(pred_poses[:,:3,3],nan=100),-100,100)

gripper_pointcloud = read_pointcloud(f"/root/autodl-tmp/shiqian/code/render/YCB-Video/models/035_power_drill/sampled_2048.txt")   # 8192,3

x = np.linspace(0,0.1,1000)
y = np.zeros(1000)
for i in range(len(pred_poses)):
    #pred_pointcloud = quaternion_apply(torch.tensor(q_preds[i]),torch.tensor(gripper_pointcloud)) + torch.tensor(t_preds[i])
    pred_pointcloud = torch.tensor(transform_pointcloud(gripper_pointcloud,pred_poses[i]))
    gt_pointcloud = torch.tensor(transform_pointcloud(gripper_pointcloud,gt_poses[i]))
    dis = torch.mean(torch.norm(gt_pointcloud - pred_pointcloud,dim=1))
    print("%d: %.4f" % (i, dis))
    #print(q_preds[i])
    #print(pred_pointcloud[:10])
    #print(gt_pointcloud[:10])
    save_pointcloud(pred_pointcloud, "pointclouds/add_pred_%.3d.txt" % i)
    save_pointcloud(gt_pointcloud, "pointclouds/add_gt_%.3d.txt" % i)

    mask = x>np.array(dis)
    y[mask] += 1/len(pred_poses)

area = np.trapz(y, x) / 0.1
print(area)


preds = pred_poses
metrics = compute_auc_all(preds,gt_poses,gripper_pointcloud)
print(metrics)
result_dict['add_10cm'].append(metrics[0])
result_dict['adds_10cm'].append(metrics[1])
metrics = compute_auc_all(preds,gt_poses,gripper_pointcloud,0.01,0.0001)
print(metrics)
result_dict['add_1cm'].append(metrics[0])
result_dict['adds_1cm'].append(metrics[1])

print(round(np.array(result_dict['add_10cm']).mean(),4),round(np.array(result_dict['adds_10cm']).mean(),4))
print(round(np.array(result_dict['add_1cm']).mean(),4),round(np.array(result_dict['adds_1cm']).mean(),4))
