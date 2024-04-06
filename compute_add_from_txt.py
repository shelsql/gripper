import os
import numpy as np
from utils.spd import read_pointcloud,transform_pointcloud
from utils.quaternion_utils import *
import torch

from itertools import combinations
# def point_cloud_diameter(point_cloud):
#     distances = [np.linalg.norm(a - b) for a, b in combinations(point_cloud, 2)]
#     return np.max(distances)
# dia = point_cloud_diameter(gripper_pointcloud)
# dia = 0.21452632541743355

feat_layer = 19
S = 32
results = np.loadtxt(f'results/layer{feat_layer}_seq{S}_refinea.txt')   # 1024,23
q_preds = results[:,:4]     # 1024,4
t_preds = results[:,4:7]    # 1024,3
gt_poses = results[:,7:].reshape(-1,4,4)    # 1024,4,4
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


