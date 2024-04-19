import os
import numpy as np
from utils.spd import read_pointcloud,transform_pointcloud
from utils.quaternion_utils import *
import torch
from utils.metrics import compute_auc_all
from itertools import combinations
# def point_cloud_diameter(point_cloud):
#     distances = [np.linalg.norm(a - b) for a, b in combinations(point_cloud, 2)]
#     return np.max(distances)
# dia = point_cloud_diameter(gripper_pointcloud)
# dia = 0.21452632541743355

feat_layer = 19
S = 32
results = np.loadtxt(f'results/memory32_key8_iter500_use_depth_no_full_no_adjust.txt')   # 1024,23
q_preds = results[:,:4]     # 1024,4
t_preds = results[:,4:7]    # 1024,3
gt_poses = results[:,7:23].reshape(-1,4,4)    # 1024,4,4
gripper_pointcloud = read_pointcloud("./pointclouds/gripper.txt")   # 8192,3

x = np.linspace(0,0.1,1000)
y = np.zeros(1000)
for i in range(len(q_preds)):
    pred_pointcloud = quaternion_apply(torch.tensor(q_preds[i]),torch.tensor(gripper_pointcloud)) + torch.tensor(t_preds[i])
    gt_pointcloud = torch.tensor(transform_pointcloud(gripper_pointcloud,np.linalg.inv(gt_poses[i])))
    dis = torch.mean(torch.norm(gt_pointcloud - pred_pointcloud,dim=1))

    mask = x>np.array(dis)
    y[mask] += 1/len(q_preds)

area = np.trapz(y, x) / 0.1
print(area)


preds = np.zeros((q_preds.shape[0],4,4))
preds[:,:3,:3] = np.array(quaternion_to_matrix(torch.tensor(q_preds)))
preds[:,:3,3] = t_preds
preds[:,3,3] = 1
metrics = compute_auc_all(preds,np.linalg.inv(gt_poses),gripper_pointcloud)
print(metrics)





for i in range(gt_poses.shape[0]):
    gt_poses[i] = np.linalg.inv(gt_poses[i])
    # preds[i] = np.linalg.inv(preds[i])


r_errors = []
t_errors = []
for i in range(gt_poses.shape[0]):
    R1 = gt_poses[i,:3, :3]/np.cbrt(np.linalg.det(gt_poses[i,:3, :3]))
    T1 = gt_poses[i,:3, 3]

    R2 = preds[i,:3, :3]/np.cbrt(np.linalg.det(preds[i,:3, :3]))
    T2 = preds[i,:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100

    theta = min(theta, np.abs(180 - theta))
    r_errors.append(theta)
    t_errors.append(shift)

num_samples = len(r_errors)
r_errors = np.array(r_errors)
t_errors = np.array(t_errors)

thresholds = [
    (5, 2),
    (5, 5),
    (10, 2),
    (10, 5),
    (10, 10)
]

print("Average R_error: %.2f Average T_error: %.2f" % (np.mean(r_errors), np.mean(t_errors)))
print("Median R_error: %.2f Median T_error: %.2f" % (np.median(r_errors), np.median(t_errors)))

for r_thres, t_thres in thresholds:
    good_samples = np.sum(np.logical_and(r_errors < r_thres, t_errors < t_thres))
    acc = (good_samples / num_samples) * 100.0
    print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))