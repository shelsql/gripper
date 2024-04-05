import time
import numpy as np
import utils.improc
import random
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
from tqdm import tqdm

from datasets.ty_datasets import PoseDataset, TrackingDataset
from datasets.ref_dataset import ReferenceDataset, SimTestDataset, SimTrackDataset
from torch.utils.data import DataLoader
from matcher import Dinov2Matcher

from utils.spd import sample_points_from_mesh, depth_map_to_pointcloud,compute_RT_errors
from utils.spd import save_pointcloud, transform_pointcloud, get_2dbboxes
from utils.spd import image_coords_to_camera_space, read_pointcloud
from utils.geometric_vision import solve_pnp_ransac, solve_pnp
import cv2
import pickle

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


def run_model(d, refs, pointcloud, matcher, device, dname, step,vis_dict=None, sw=None):
    metrics = {}
    rgbs = torch.Tensor(d['rgb'])[0].float().permute(0, 3, 1, 2).to(device) # B, C, H, W
    depths = torch.Tensor(d['depth'])[0].float().permute(0, 3, 1, 2).to(device)
    masks = torch.Tensor(d['mask'])[0].float().permute(0, 3, 1, 2).to(device)
    c2ws = d['c2w'][0] # B, 4, 4
    o2ws = d['obj_pose'][0] # B, 4, 4
    intrinsics = d['intrinsics']
    
    test_camera_K = np.zeros((3,3))
    test_camera_K[0,0] = intrinsics['fx']
    test_camera_K[1,1] = intrinsics['fy']
    test_camera_K[0,2] = intrinsics['cx']
    test_camera_K[1,2] = intrinsics['cy']
    test_camera_K[2,2] = 1
    #print(test_camera_K)


    raw_r_errors = []
    raw_t_errors = []
    S = rgbs.shape[0]

    matches_3ds, rt_matrixs, test_camera_Ks, gt_poses = [], [], [], []  # 需要传出去的

    for i in range(S):
        start_time = time.time()
        frame = {
            'rgb': d['rgb'][0,i:i+1],
            'depth': d['depth'][0,i:i+1],
            'mask': d['mask'][0,i:i+1],
            'feat': d['feat'][0,i:i+1],
            'intrinsics': d['intrinsics']
        }
        matches_3d_list = matcher.match_batch(frame, step=i,N_select=10)  # N, 6i
        #print(matches_3d[::10])
        # if matches_3d is None:
        #     print("No matches")
        #     rt_matrix = np.eye(4)
        #     continue
        matches_3d_multi_view = []
        rt_matrix_multi_view = []
        for matches_3d in matches_3d_list:
            matches = matches_3d    # 这样写是对的，不用clone
            matches[:,[1,2]] = matches[:,[2,1]]
            pnp_retval, translation, rt_matrix, inlier = solve_pnp_ransac(matches[:,3:6].cpu().numpy(), matches[:,1:3].cpu().numpy(), camera_K=test_camera_K)
            matches_3d_multi_view.append(matches_3d[inlier.reshape(-1)])
            rt_matrix_multi_view.append(rt_matrix)

        # 10个views都用（可能有重复），选inlier最多的rt_matrix
        n = 0
        select_id = 0
        for idx in range(len(matches_3d_multi_view)):
            if matches_3d_multi_view[idx].shape[0] > n:
                n = matches_3d_multi_view[idx].shape[0]
                select_id = idx



        matches_3ds.append(torch.cat(matches_3d_multi_view,dim=0))
        rt_matrixs.append(rt_matrix_multi_view[select_id])
        test_camera_Ks.append(test_camera_K)
        gt_cam_to_obj = np.dot(np.linalg.inv(o2ws[i]), c2ws[i])
        gt_obj_to_cam = np.linalg.inv(gt_cam_to_obj)
        gt_pose = gt_cam_to_obj
        gt_poses.append(gt_pose)

        #print("pnp_retval:", pnp_retval)
        # if not pnp_retval:
        #     print("No PnP result")
        #     rt_matrix = np.eye(4)
            

    #     R1 = gt_obj_to_cam[:3, :3]/np.cbrt(np.linalg.det(gt_obj_to_cam[:3, :3]))
    #     T1 = gt_obj_to_cam[:3, 3]
    #
    #     R2 = rt_matrix[:3, :3]/np.cbrt(np.linalg.det(rt_matrix[:3, :3]))
    #     T2 = rt_matrix[:3, 3]
    #
    #     R = R1 @ R2.transpose()
    #     theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    #     shift = np.linalg.norm(T1-T2) * 100
    #
    #     theta = min(theta, np.abs(180 - theta))
    #     raw_r_errors.append(theta)
    #     raw_t_errors.append(shift)
    #     print("R_error: %.2f T_error: %.2f time:%.2f" % (theta, shift, time.time() - start_time))
    #
    # raw_r_errors = np.array(raw_r_errors)
    # raw_t_errors = np.array(raw_t_errors)
    # print("Single frame metrics,without optimization")
    # print("Average R error: %.2f Average T error: %.2f" % (np.mean(raw_r_errors), np.mean(raw_t_errors)))
    # #save_pointcloud(matches_3d[:,3:].cpu().numpy(), "./pointclouds/matched_3d_pts.txt")
    #
    # thresholds = [
    #     (5, 2),
    #     (5, 5),
    #     (10, 2),
    #     (10, 5),
    #     (10, 10)
    # ]
    # for r_thres, t_thres in thresholds:
    #     good_samples = np.sum(np.logical_and(raw_r_errors < r_thres, raw_t_errors < t_thres))
    #     acc = (good_samples / S) * 100.0
    #     print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))

    gt_poses = np.array(gt_poses)
    # if vis_dict != None:
    #     vis_dict['rgbs'] = np.array(rgbs.cpu()).tolist()
    #     vis_dict['gt_poses'] = np.stack(gt_poses).tolist()
    #     vis_dict['matches_3ds'] = [tmp.tolist() for tmp in matches_3ds]

    return matches_3ds,np.stack(rt_matrixs,axis=0),test_camera_Ks,gt_poses

def main(
        dname='sim',
        exp_name='debug',
        B=1, # batchsize
        S=32, # seqlen
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        is_training=True,
        log_dir='./logs_match',
        ref_dir='/root/autodl-tmp/shiqian/code/gripper/ref_views/franka_69.4_840',
        test_dir='/root/autodl-tmp/shiqian/code/gripper/test_views/franka_69.4_1024',
        optimize=False,
        feat_layer=19, # Which layer of features from dinov2 to take
        max_iters=3,
        log_freq=1,
        device_ids=[2],
        record_vis=True,

):
    
    # The idea of this file is to test DinoV2 matcher and multi frame optimization on Blender rendered data
    device = 'cuda:%d' % device_ids[0]
    
    exp_name = 'test_on_sim_seq'

    
    ## autogen a descriptive name
    model_name = "%d_%d" % (B, S)
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % dname
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    test_dataset = SimTrackDataset(dataset_location=test_dir, seqlen=S, features=feat_layer)
    ref_dataset = ReferenceDataset(dataset_location=ref_dir, num_views=840, features=feat_layer)
    test_dataloader = DataLoader(test_dataset, batch_size=B, shuffle=shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=shuffle)
    iterloader = iter(test_dataloader)
    # Load ref images and init Dinov2 Matcher
    refs = next(iter(ref_dataloader))
    ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3) # B, S, C, H, W
    ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3)
    ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3)
    ref_images = torch.concat([ref_rgbs[0], ref_depths[0,:,0:1], ref_masks[0,:,0:1]], axis = 1)
    print(ref_images.shape)

    global_step = 0
    
    gripper_path = "/root/autodl-tmp/shiqian/code/gripper/franka_hand_obj/franka_hand.obj"
    
    pointcloud_path = "./pointclouds/gripper.txt"
    if not os.path.exists(pointcloud_path):
        gripper_pointcloud = sample_points_from_mesh(gripper_path, fps = True, n_pts=8192)
    else:
        gripper_pointcloud = read_pointcloud(pointcloud_path)
    
    matcher = Dinov2Matcher(ref_dir=ref_dir, refs=refs,
                            model_pointcloud=gripper_pointcloud,
                            feat_layer = feat_layer,
                            device=device)

    q_preds,t_preds,gt_poses_for_result = [],[],[]
    while global_step < max_iters: 
        print("Iteration {}".format(global_step))
        matches_3ds, rt_matrixs, test_camera_Ks, gt_poses = [], [], [], []
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0
        
        read_start_time = time.time()
        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=8,
            scalar_freq=int(log_freq/4),
            just_gif=True)

        sample = next(iterloader)
        read_time = time.time()-read_start_time
        iter_read_time += read_time
        if record_vis:
            vis_dict = {}
        else:
            vis_dict = None
        if sample is not None:
            matches_3ds,rt_matrixs,test_camera_Ks,gt_poses = run_model(sample, refs, gripper_pointcloud, matcher, device, dname, global_step, vis_dict,sw=sw_t)
        else:
            print('sampling failed')
        qt_pred_for_vis_seq = []
        for view_id in range(S):      # seq中的每一个view，都算一遍结果，从而得到整个数据集所有view的统计结果
            views_idx_for_opt = fps_optimize_views_from_test(gt_poses, select_numbers=S // 4,start_idx=view_id)   # 对于当前view，用fps选出几个用来辅助优化的view

            q_pred,t_pred = optimize_reproject([matches_3ds[i] for i in views_idx_for_opt],
                                               [rt_matrixs[i] for i in views_idx_for_opt],
                                               [test_camera_Ks[i] for i in views_idx_for_opt],
                                               [gt_poses[i] for i in views_idx_for_opt],
                                               qt_pred_for_vis_seq,
                                               device=device,
                                                dtype=matches_3ds[0][0].dtype)
            q_preds.append(q_pred)
            t_preds.append(t_pred)
            gt_poses_for_result.append(gt_poses[view_id])
        if vis_dict != None:
            vis_dict['qt_preds'] = qt_pred_for_vis_seq
            iter_time = time.time()-iter_start_time
            if not os.path.exists(f'vis_results/layer{feat_layer}_seq{S}'):
                os.makedirs(f'vis_results/layer{feat_layer}_seq{S}')
            with open(f'vis_results/layer{feat_layer}_seq{S}/{global_step}_10views.pkl','wb') as f:
                pickle.dump(vis_dict,f)



    q_preds = torch.stack(q_preds,dim=0)
    t_preds = torch.stack(t_preds, dim=0)
    gt_poses_for_result = torch.tensor(np.stack(gt_poses_for_result,axis=0),device=device)
    r_errors,t_errors = compute_results(q_preds, t_preds,gt_poses_for_result)
    results = np.concatenate(([r_errors],[t_errors]),axis=0)
    # if not os.path.exists(f'results'):
    #     os.makedirs(f'results')
    # np.savetxt(f'results/layer{feat_layer}_seq{S}.txt',results)

    # num_samples = len(r_errors)
    # r_errors = np.array(r_errors)
    # t_errors = np.array(t_errors)

    thresholds = [
        (5, 2),
        (5, 5),
        (10, 2),
        (10, 5),
        (10, 10)
    ]

    # print("Average R_error: %.2f Average T_error: %.2f" % (np.mean(r_errors), np.mean(t_errors)))
    #
    # for r_thres, t_thres in thresholds:
    #     good_samples = np.sum(np.logical_and(r_errors < r_thres, t_errors < t_thres))
    #     acc = (good_samples / num_samples) * 100.0
    #     print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))


    writer_t.close()
from utils.quaternion_utils import *

def optimize_reproject(matches_3d_multi_view, rt_matrixs_multi_view, test_camera_Ks, gt_poses,qt_pred_for_vis_seq,device,dtype):
    '''删除垃圾点的迭代方法，优化的是输入list的第一帧，后面的帧仅辅助
    # matches_3ds: list[tensor(342,6),...] different shape
    # other: list[np.array(4,4)or(3,3)], same shape'''




    optimizer = torch.optim.Adam([{'params': q_pred, 'lr': 1e-2}, {'params': t_pred, 'lr': 1e-3}], lr=1e-4)
    start_time = time.time()
    iteration = 0
    loss_change = 1
    loss_last = 0
    qt_pred_for_vis_frame = []
    while iteration < 200 and abs(loss_change) > 1e-4:
        qt_pred_for_vis_frame.append((q_pred.tolist(), t_pred.tolist()))
        optimizer.zero_grad()
        reproj_dis_list = []
        for i in range(len(matches_3ds)):
            fact_2d = matches_3ds[i][:, 1:3].clone()  # 342,2
            # use rotation matrix to change:
            pred_3d = quaternion_apply(matrix_to_quaternion(torch.inverse(gt_poses[i])[:3, :3]),
                                       matches_3ds[i][:, 3:]) + torch.inverse(gt_poses[i])[:3, 3]
            pred_3d = quaternion_apply(matrix_to_quaternion(gt_poses[0][:3, :3]), pred_3d) + gt_poses[0][:3, 3]
            pred_3d = quaternion_apply(q_pred, pred_3d) + t_pred

            proj_2d = (torch.matmul(test_camera_Ks[i], pred_3d.transpose(1, 0)) / pred_3d.transpose(1, 0)[2,
                                                                                  :]).transpose(1, 0)[:, :2]  # 220,2
            reproj_dis = torch.norm(fact_2d - proj_2d, dim=1)  # 220,
            reproj_dis_list.append(reproj_dis)

            _ = torch.mean(torch.norm(fact_2d - proj_2d, dim=1))
            # _unit = torch.norm(q_pred)

        reproj_dis_list = torch.cat(reproj_dis_list, dim=0)
        mean = reproj_dis_list.mean()
        std = reproj_dis_list.std()
        within_3sigma = (reproj_dis_list >= mean - 3 * std) & (reproj_dis_list <= mean + 3 * std)
        reproj_dis_list = reproj_dis_list[within_3sigma]    # 删除垃圾点

        loss = 1e4 * (1 - torch.norm(q_pred)) ** 2 + torch.mean(reproj_dis_list)

        # print(iteration, loss.item())

        loss.backward()
        optimizer.step()

        loss_change = loss - loss_last
        loss_last = loss
        iteration += 1
    # 每次只挑30帧左右可视化
    if len(qt_pred_for_vis_frame) > 27:
        first_qt = qt_pred_for_vis_frame.pop(0)
        last_qt = qt_pred_for_vis_frame.pop()
        step = len(qt_pred_for_vis_frame) // 25
        qt_pred_for_vis_frame = [qt_pred_for_vis_frame[i*step] for i in range(25)]
        qt_pred_for_vis_frame.insert(0, first_qt)
        qt_pred_for_vis_frame.append(last_qt)
        qt_pred_for_vis_seq.append(qt_pred_for_vis_frame)
    end_time = time.time()
    print("Time", end_time - start_time)
    return q_pred, t_pred

def fps_optimize_views_from_test(poses, select_numbers=16,start_idx=0):
    dist_mat = np.zeros((poses.shape[0],poses.shape[0]))
    for i,pose1 in enumerate(poses):  # TODO batch形式
        for j,pose2 in enumerate(poses):
            dist_mat[i,j],_ = compute_RT_errors(pose1,pose2)
    select_views = np.zeros((select_numbers,), dtype=int)
    view_idx = start_idx
    dist_to_set = dist_mat[:,view_idx]
    for i in range(select_numbers):
        select_views[i] = view_idx
        dist_to_set = np.minimum(dist_to_set,dist_mat[:,view_idx])
        view_idx = np.argmax(dist_to_set)

    return select_views

def compute_results(q_preds,t_preds,gt_poses):
    r_preds = quaternion_to_matrix(q_preds) # n.3.3

    rt_preds = torch.zeros_like(gt_poses)   # n,4,4
    rt_preds[:,:3,:3] = r_preds
    rt_preds[:,:3,3] = t_preds
    rt_preds[:,3,3] = 1


    pose_preds = torch.inverse(rt_preds)

    r_errors = []
    t_errors = []
    pose_preds = pose_preds.detach().cpu().numpy()
    gt_poses = gt_poses.detach().cpu().numpy()

    for i in range(gt_poses.shape[0]):
        R1 = gt_poses[i,:3, :3]/np.cbrt(np.linalg.det(gt_poses[i,:3, :3]))
        T1 = gt_poses[i,:3, 3]

        R2 = pose_preds[i,:3, :3]/np.cbrt(np.linalg.det(pose_preds[i,:3, :3]))
        T2 = pose_preds[i,:3, 3]

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

    return r_errors, t_errors
if __name__ == '__main__':
    Fire(main)