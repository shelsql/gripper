import os
import pickle

import numpy as np
from utils.spd import read_pointcloud,transform_pointcloud
from utils.quaternion_utils import *
import torch
from utils.metrics import compute_auc_all
import matplotlib.pyplot as plt

fold_name = 'finalfinal/layer19'
gripper_name_list = ['realpanda','panda','robotiq2f140','robotiq2f85','kinova','shadowhand','robotiq3f',]
result_dict = {
    # 'add_10cm':[],
    # 'adds_10cm':[],
    'add_1cm':[],
    'adds_1cm':[]
}
for gripper_name in gripper_name_list:
    results_total = []
    # 优化时

    if gripper_name == 'realpanda':
        for i in range(1,5):#[1,2,5,9]:#:
            results = np.loadtxt(f'results/{fold_name}realpca/panda_{i}.txt')   # 不是pnp only 时
            results_total.append(results)
        gripper_pointcloud = read_pointcloud(f"/home/data/tianshuwu/data/final_20240419/panda/model/sampled_2048.txt")   # 8192,3

    else:
        for i in range(1,31):#[1,2,5,9]:#:
            results = np.loadtxt(f'results/{fold_name}pca/{gripper_name}_{i}.txt')   # 不是pnp only 时
            results_total.append(results)
        gripper_pointcloud = read_pointcloud(f"/home/data/tianshuwu/data/final_20240419/{gripper_name}/model/sampled_2048.txt")   # 8192,3


    results_total = np.concatenate(results_total,axis=0)
    q_preds = results_total[:,:4]     # 1024,4
    t_preds = np.clip(np.nan_to_num(results_total[:,4:7],nan=100),-100,100)    # 1024,3
    gt_poses = results_total[:,7:23].reshape(-1,4,4)    # 1024,4,4

    # pnp only 的
    # results = np.loadtxt(f'results/{fold_name}/{gripper_name}_pnponly_full.txt')
    # pred_poses = results[:,:16].reshape(-1,4,4)
    # gt_poses = np.linalg.inv(results[:,16:].reshape(-1,4,4))
    # q_preds = matrix_to_quaternion(torch.tensor(pred_poses[:,:3,:3])).numpy()
    # t_preds = np.clip(np.nan_to_num(pred_poses[:,:3,3],nan=100),-100,100)


    # gripper_pointcloud = read_pointcloud("/home/data/tianshuwu/code/gripper/results/tmp/sampled_2048.txt")

    # x = np.linspace(0,0.1,1000)
    # y = np.zeros(1000)
    # for i in range(len(q_preds)):
    #     pred_pointcloud = quaternion_apply(torch.tensor(q_preds[i]),torch.tensor(gripper_pointcloud)) + torch.tensor(t_preds[i])
    #     gt_pointcloud = torch.tensor(transform_pointcloud(gripper_pointcloud,np.linalg.inv(gt_poses[i])))
    #     dis = torch.mean(torch.norm(gt_pointcloud - pred_pointcloud,dim=1))
    #
    #     mask = x>np.array(dis)
    #     y[mask] += 1/len(q_preds)


    preds = np.zeros((q_preds.shape[0],4,4))
    preds[:,:3,:3] = np.array(quaternion_to_matrix(torch.tensor(q_preds)))
    preds[:,:3,3] = t_preds
    preds[:,3,3] = 1

    # metrics = compute_auc_all(preds,np.linalg.inv(gt_poses),gripper_pointcloud)
    # print(metrics)
    # result_dict['add_10cm'].append(metrics[0])
    # result_dict['adds_10cm'].append(metrics[1])

    metrics = compute_auc_all(preds,np.linalg.inv(gt_poses),gripper_pointcloud,0.01,0.0001)
    result_dict['add_1cm'].append(metrics[0]*100)
    result_dict['adds_1cm'].append(metrics[1]*100)

    print(gripper_name,'add/adds')
    print(f'{round(metrics[0]*100, 2)}/{round(metrics[1]*100, 2)}')

# print(round(np.array(result_dict['add_10cm']).mean(),4),round(np.array(result_dict['adds_10cm']).mean(),4))
print('mean', round(np.array(result_dict['add_1cm']).mean(),2),round(np.array(result_dict['adds_1cm']).mean(),2))



    # 5deg 10cm metrics:
    # for i in range(gt_poses.shape[0]):
    #     gt_poses[i] = np.linalg.inv(gt_poses[i])
    #     # preds[i] = np.linalg.inv(preds[i])
    #
    # r_count = 0
    # t_count = 0
    # r_errors = []
    # t_errors = []
    # for i in range(gt_poses.shape[0]):
    #     R1 = gt_poses[i,:3, :3]/np.cbrt(np.linalg.det(gt_poses[i,:3, :3]))
    #     T1 = gt_poses[i,:3, 3]
    #
    #     R2 = preds[i,:3, :3]/np.cbrt(np.linalg.det(preds[i,:3, :3]))
    #     T2 = preds[i,:3, 3]
    #
    #     R = R1 @ R2.transpose()
    #     theta = np.arccos(np.clip((np.trace(R) - 1)/2,-1,1)) * 180/np.pi
    #     shift = np.linalg.norm(T1-T2) * 100
    #
    #     r_errors.append(theta)
    #     t_errors.append(shift)
    #     if theta>10:
    #         r_count +=1
    #     if shift>5:
    #         t_count +=1
    #
    # num_samples = len(r_errors)
    # r_errors = np.array(r_errors)
    # t_errors = np.array(t_errors)
    #
    # thresholds = [
    #     (180,1),
    #     (5, 2),
    #     (5, 5),
    #     (10, 2),
    #     (10, 5),
    #     (10, 10)
    # ]
    # print("Average R_error: %.2f Average T_error: %.2f" % (np.mean(r_errors), np.mean(t_errors)))
    # print("Median R_error: %.2f Median T_error: %.2f" % (np.median(r_errors), np.median(t_errors)))
    #
    # for r_thres, t_thres in thresholds:
    #     good_samples = np.sum(np.logical_and(r_errors < r_thres, t_errors < t_thres))
    #     acc = (good_samples / num_samples) * 100.0
    #     print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))

    # time cost statics
    # f_read = open(f'results/{fold_name}/{gripper_name}_30_time_dick.pkl', 'rb')
    # dict2 = pickle.load(f_read)
    # move_data_to_gpu_time = np.array(dict2['move data to gpu time']).mean()
    # extract_feature_time = np.array(dict2['extract feature time']).mean()
    # select_refs_time = np.array(dict2['select ref time']).mean()
    # match_time = np.array(dict2['match time']).mean()
    # pnp_time = np.array(dict2['pnp time']).mean()
    # prepare_time = np.array(dict2['prepare time']).mean()
    # ceres_time = np.array(dict2['ceres time']).mean()
    # total_time = np.array(dict2['total time']).mean()
    # mask_time = np.array(dict2['mask time']).mean()
    # print('move_data_to_gpu_time',move_data_to_gpu_time)
    # print('extract_feature_time',extract_feature_time)
    # print('mask_time',mask_time)
    # print('select_refs_time',select_refs_time)
    # print('match_time',match_time)
    # print('pnp_time',pnp_time)
    # print('prepare_time',prepare_time)
    # print('ceres_time',ceres_time)
    # print('total_time',total_time)
    # run_model_time = np.array(dict2['run model time']).mean()
    # interpolate_time = np.array(dict2['interpolate time']).mean()
    # create3dmesh_time = np.array(dict2['3dmesh time']).mean()