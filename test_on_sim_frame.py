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

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


def run_model(d, refs, pointcloud, matcher, device, dname, step, sw=None):
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
    gt_poses = []
    for i in range(S):
        start_time = time.time()
        frame = {
            'rgb': d['rgb'][0,i:i+1],
            'depth': d['depth'][0,i:i+1],
            'mask': d['mask'][0,i:i+1],
            'feat': d['feat'][0,i:i+1],
            'intrinsics': d['intrinsics']
        }
        matches_3d_list = matcher.match_batch(frame, i)  # N, 6i
        #print(matches_3d[::10])
        matches_3d = torch.cat(matches_3d_list, dim=0)
        matches = matches_3d
        matches[:,[1,2]] = matches[:,[2,1]]
        #print(matches)
        #save_pointcloud(matches[:,3:].cpu().numpy(), "./pointclouds/matched_3d_pts.txt")
        pnp_retval, translation, rt_matrix, inlier = solve_pnp_ransac(matches[:,3:6].cpu().numpy(), matches[:,1:3].cpu().numpy(), camera_K=test_camera_K)
        #pnp_retval, translation, rt_matrix = solve_pnp_ransac(pts_3d, pts_2d[:,::-1].astype(float), camera_K=test_camera_K)
        
        #print("pnp_retval:", pnp_retval)
        if not pnp_retval:
            print("No PnP result")
            rt_matrix = np.eye(4)
            
        gt_cam_to_obj = np.dot(np.linalg.inv(o2ws[i]), c2ws[i])
        gt_obj_to_cam = np.linalg.inv(gt_cam_to_obj)
        gt_pose = gt_cam_to_obj
        gt_poses.append(gt_pose)
        R1 = gt_obj_to_cam[:3, :3]/np.cbrt(np.linalg.det(gt_obj_to_cam[:3, :3]))
        T1 = gt_obj_to_cam[:3, 3]
        
        flip_180 = np.array([[-1,0,0],
                             [0,-1,0],
                             [0,0,1]])
        flipped_R = np.dot(gt_obj_to_cam[:3,:3], flip_180)
        R1_flipped = flipped_R/np.cbrt(np.linalg.det(flipped_R))

        R2 = rt_matrix[:3, :3]/np.cbrt(np.linalg.det(rt_matrix[:3, :3]))
        T2 = rt_matrix[:3, 3]

        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
        shift = np.linalg.norm(T1-T2) * 100
        
        R_flip = R1_flipped @ R2.transpose()
        theta_flip = np.arccos((np.trace(R_flip) - 1)/2) * 180/np.pi

        #theta = min(theta, np.abs(180 - theta))
        theta = min(theta, theta_flip)
        raw_r_errors.append(theta)
        raw_t_errors.append(shift)
        print("R_error: %.2f T_error: %.2f time:%.2f" % (theta, shift, time.time() - start_time))
        
    raw_r_errors = np.array(raw_r_errors)
    raw_t_errors = np.array(raw_t_errors)
    print("Single frame metrics")
    print("Average R error: %.2f Average T error: %.2f" % (np.mean(raw_r_errors), np.mean(raw_t_errors)))
    #save_pointcloud(matches_3d[:,3:].cpu().numpy(), "./pointclouds/matched_3d_pts.txt")    

    thresholds = [
        (5, 2),
        (5, 5),
        (10, 2),
        (10, 5),
        (10, 10)
    ]
    for r_thres, t_thres in thresholds:
        good_samples = np.sum(np.logical_and(raw_r_errors < r_thres, raw_t_errors < t_thres))
        acc = (good_samples / S) * 100.0
        print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))
        
    metrics = {
        'r_errors': raw_r_errors,
        't_errors': raw_t_errors
    }
        
    return metrics

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
        feat_layer=23, # Which layer of features from dinov2 to take
        max_iters=32,
        log_freq=1,
        device_ids=[0],
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
    test_dataset = TrackingDataset(features=feat_layer)
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
    
    all_r_errors = []
    all_t_errors = []
    while global_step < max_iters: # Num of test images
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

        if sample is not None:
            metrics = run_model(sample, refs, gripper_pointcloud, matcher, device, dname, global_step, sw=sw_t)
        else:
            print('sampling failed')
        iter_time = time.time()-iter_start_time
        # TODO 优化后的结果
        seq_r_error = metrics['r_errors']
        seq_t_error = metrics['t_errors']
        all_r_errors.append(seq_r_error)
        all_t_errors.append(seq_t_error)

        print('%s; step %06d/%d; itime %.2f; R_error %.2f; T_error %.2f' % (
                model_name, global_step, max_iters, iter_time, np.mean(seq_r_error), np.mean(seq_t_error)))


    # num_samples = len(r_errors)
    all_r_errors = np.array(all_r_errors).ravel()
    all_t_errors = np.array(all_t_errors).ravel()
    num_samples = all_r_errors.shape[0]

    thresholds = [
        (5, 2),
        (5, 5),
        (10, 2),
        (10, 5),
        (10, 10)
    ]

    print("Metrics")
    print("Average R_error: %.2f Average T_error: %.2f" % (np.mean(all_r_errors), np.mean(all_t_errors)))
    
    for r_thres, t_thres in thresholds:
        good_samples = np.sum(np.logical_and(all_r_errors < r_thres, all_t_errors < t_thres))
        acc = (good_samples / num_samples) * 100.0
        print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))


    writer_t.close()
from utils.quaternion_utils import *

def optimize_reproject(matches_3ds, rt_matrixs, test_camera_Ks, gt_poses):
    '''删除垃圾点的迭代方法'''
    # matches_3ds: list[tensor(342,6),...] different shape
    # other: list[np.array(4,4)or(3,3)], same shape

    rt_matrixs = torch.tensor(rt_matrixs, device=matches_3ds[0].device, dtype=matches_3ds[0].dtype)  # b,4,4
    test_camera_Ks = torch.tensor(test_camera_Ks, device=matches_3ds[0].device, dtype=matches_3ds[0].dtype)  # b,3,3
    gt_poses = torch.tensor(gt_poses, device=matches_3ds[0].device, dtype=matches_3ds[0].dtype)  # b,4,4
    q_pred = torch.tensor(matrix_to_quaternion(rt_matrixs[0][:3, :3]), requires_grad=True)  # 4
    t_pred = torch.tensor(rt_matrixs[0][:3, 3], requires_grad=True)  # 3
    optimizer = torch.optim.Adam([{'params': q_pred, 'lr': 2e-2}, {'params': t_pred, 'lr': 2e-2}], lr=1e-4)
    start_time = time.time()
    iteration = 0
    loss_change = 1
    loss_last = 0
    while iteration < 400 and abs(loss_change) > 1e-7:
        optimizer.zero_grad()
        # reproj_loss = 0
        reproj_dis_list = []
        for i in range(len(matches_3ds)):
            fact_2d = matches_3ds[i][:, 1:3].clone()  # 342,2
            fact_2d[:, [0, 1]] = fact_2d[:, [1, 0]]
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
            _unit = torch.norm(q_pred)

        reproj_dis_list = torch.cat(reproj_dis_list, dim=0)
        mean = reproj_dis_list.mean()
        std = reproj_dis_list.std()
        within_3sigma = (reproj_dis_list >= mean - 1 * std) & (reproj_dis_list <= mean + 1 * std)
        reproj_dis_list = reproj_dis_list[within_3sigma]

        loss = 1e4 * (1 - torch.norm(q_pred)) ** 2 + torch.mean(reproj_dis_list)

        print(iteration, loss.item())

        loss.backward()
        optimizer.step()

        loss_change = loss - loss_last
        loss_last = loss
        iteration += 1

    end_time = time.time()
    print("Time", end_time - start_time)
    return q_pred, t_pred

def fps_optimize_views_from_test(poses, select_numbers=16):
    dist_mat = np.zeros((poses.shape[0],poses.shape[0]))
    for i,pose1 in enumerate(poses):  # TODO batch形式
        for j,pose2 in enumerate(poses):
            dist_mat[i,j],_ = compute_RT_errors(pose1,pose2)
    select_views = np.zeros((select_numbers,), dtype=int)
    view_idx = 0
    dist_to_set = dist_mat[:,view_idx]
    for i in range(select_numbers):
        select_views[i] = view_idx
        dist_to_set = np.minimum(dist_to_set,dist_mat[:,view_idx])
        view_idx = np.argmax(dist_to_set)

    return select_views

def compute_results(q_preds,t_preds,gt_poses):
    r_preds = quaternion_to_matrix(q_preds) # n.3.3

    rt_preds = torch.zeros_like(gt_poses)   # n,4,4
    rt_preds[:,:3,:3] = r_preds              #
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

    for r_thres, t_thres in thresholds:
        good_samples = np.sum(np.logical_and(r_errors < r_thres, t_errors < t_thres))
        acc = (good_samples / num_samples) * 100.0
        print("%.1f degree %.1f cm threshold: %.2f" % (r_thres, t_thres, acc))

if __name__ == '__main__':
    Fire(main)
