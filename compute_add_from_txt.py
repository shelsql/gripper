import os
import numpy as np
from utils.spd import read_pointcloud,transform_pointcloud
from utils.quaternion_utils import *
import torch
from utils.metrics import compute_auc_all

gripper_name = 'panda'
results_total = []
for i in range(1,31):
    results = np.loadtxt(f'results/单帧both/{gripper_name}_{i}.txt')   # 1024,23
    results_total.append(results)

results_total = np.concatenate(results_total,axis=0)
q_preds = results_total[:,:4]     # 1024,4
t_preds = results_total[:,4:7]    # 1024,3
gt_poses = results_total[:,7:23].reshape(-1,4,4)    # 1024,4,4

# results = np.loadtxt(f'results/memory/{gripper_name}_pnponly.txt')
# pred_poses = results[:,:16].reshape(-1,4,4)
# gt_poses = np.linalg.inv(results[:,16:].reshape(-1,4,4))
# q_preds = matrix_to_quaternion(torch.tensor(pred_poses[:,:3,:3])).numpy()
# t_preds = pred_poses[:,:3,3]

gripper_pointcloud = read_pointcloud(f"/root/autodl-tmp/shiqian/datasets/final_20240419/{gripper_name}/model/sampled_4096.txt")   # 8192,3

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
metrics = compute_auc_all(preds,np.linalg.inv(gt_poses),gripper_pointcloud,0.01,0.0001)
print(metrics)





for i in range(gt_poses.shape[0]):
    gt_poses[i] = np.linalg.inv(gt_poses[i])
    # preds[i] = np.linalg.inv(preds[i])

r_count = 0
t_count = 0
r_errors = []
t_errors = []
for i in range(gt_poses.shape[0]):
    R1 = gt_poses[i,:3, :3]/np.cbrt(np.linalg.det(gt_poses[i,:3, :3]))
    T1 = gt_poses[i,:3, 3]

    R2 = preds[i,:3, :3]/np.cbrt(np.linalg.det(preds[i,:3, :3]))
    T2 = preds[i,:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos(np.clip((np.trace(R) - 1)/2,-1,1)) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100

    r_errors.append(theta)
    t_errors.append(shift)
    if theta>10:
        r_count +=1
    if shift>5:
        t_count +=1

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