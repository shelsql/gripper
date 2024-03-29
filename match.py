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

from ty_datasets import PoseDataset, TrackingDataset
from ref_dataset import ReferenceDataset, SimTestDataset
from torch.utils.data import DataLoader
from matcher import Dinov2Matcher

from utils.spd import sample_points_from_mesh, depth_map_to_pointcloud
from utils.spd import save_pointcloud, transform_pointcloud, get_2dbboxes
from utils.spd import image_coords_to_camera_space, read_pointcloud
from utils.geometric_vision import solve_pnp_ransac, solve_pnp
import cv2

random.seed(11)
np.random.seed(11)
torch.manual_seed(11)

def run_model(d, refs, pointcloud, matcher, device, dname, step, sw=None):
    metrics = {}

    rgbs = torch.Tensor(d['rgb']).float().permute(0, 3, 1, 2).to(device) # B, C, H, W
    depths = torch.Tensor(d['depth']).float().permute(0, 3, 1, 2).to(device)
    masks = torch.Tensor(d['mask']).float().permute(0, 3, 1, 2).to(device)
    #kptss = d['kpts']
    #npys = d['npy']
    c2ws = d['c2w'] # B, 4, 4
    o2ws = d['obj_pose'] # B, 4, 4
    intrinsics = d['intrinsics']

    ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3).to(device) # B, S, C, H, W
    ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3).to(device)
    ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3).to(device)
    ref_c2ws = refs['c2ws']
    ref_intrinsics = refs['intrinsics']
    #print(kptss)
    #print(npys)
    
    '''
    gripper_info = kptss[0]['keypoints'][8]
    print(gripper_info)
    gripper_t = torch.tensor(gripper_info["location_wrt_cam"]).numpy()
    gripper_r = torch.tensor(gripper_info["R2C_mat"]).numpy()
    gripper_rt = np.zeros((4,4))
    gripper_rt[:3, :3] = gripper_r
    gripper_rt[:3, 3] = gripper_t
    gripper_rt[3, 3] = 1
    '''
    
    print(rgbs.shape, depths.shape, masks.shape)
    #masks = (masks >= 9).float()
    images = torch.concat([rgbs, depths[:,0:1], masks[:,0:1]], axis = 1)
    matches_3d = matcher.match_and_fuse(d)  # N, 6
    

    test_camera_K = np.zeros((3,3))
    #test_camera_K[0,0] = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    #test_camera_K[1,1] = intrinsics['camera_settings'][0]['intrinsic_settings']['fy']
    #test_camera_K[0,2] = intrinsics['camera_settings'][0]['intrinsic_settings']['cx']
    #test_camera_K[1,2] = intrinsics['camera_settings'][0]['intrinsic_settings']['cy']
    test_camera_K[0,0] = intrinsics['fx']
    test_camera_K[1,1] = intrinsics['fy']
    test_camera_K[0,2] = intrinsics['cx']
    test_camera_K[1,2] = intrinsics['cy']
    test_camera_K[2,2] = 1
    #print(test_camera_K)
    
    gt_cam_to_obj = np.dot(np.linalg.inv(o2ws[0]), c2ws[0])
    gt_obj_to_cam = np.linalg.inv(gt_cam_to_obj)
    #print("o2w:", o2ws[0])
    #print("c2w:", c2ws[0])
    #print("gt_cam_to_obj:", gt_cam_to_obj)
    #print("gt_cam_to_obj_inv", np.linalg.inv(gt_pose))
    
    #test_pc = depth_map_to_pointcloud(depths[0,0], ref_masks[0,0], intrinsics)
    
    #TODO cluster debug pose and PnP

    for i in range(rgbs.shape[0]):
        if matches_3d is None:
            print("No matches")
            rt_matrix = np.eye(4)
            continue
        #valid_pts_2d = torch.nonzero(masks[i,0] == 1)
        #print(valid_pts_2d.shape, valid_pts_2d)
        #exit()
        #pts_2d = valid_pts_2d[::50].cpu().numpy()
        #pts_3d = image_coords_to_camera_space(depths[0,0].cpu().numpy(), pts_2d, intrinsics)
        #pts_3d = transform_pointcloud(pts_3d, gt_pose)
        #print(pts_2d)
        #print(pts_3d)
        #marked_rgb = rgbs[0].permute(1,2,0).cpu().numpy()
        #marked_rgb[pts_2d[:10,0], pts_2d[:10,1]] = np.array([0,0,255])
        #cv2.imwrite("./match_vis/marked_2d_pts.png",marked_rgb)
        #save_pointcloud(pts_3d[:10], "./pointclouds/selected_pts.txt")
        matches = matches_3d[matches_3d[:,0] == i]
        matches[:,[1,2]] = matches[:,[2,1]]
        #print(matches)
        save_pointcloud(matches[:,3:].cpu().numpy(), "./pointclouds/matched_3d_pts.txt")
        pnp_retval, translation, rt_matrix = solve_pnp_ransac(matches[:,3:6].cpu().numpy(), matches[:,1:3].cpu().numpy(), camera_K=test_camera_K)
        #pnp_retval, translation, rt_matrix = solve_pnp_ransac(pts_3d, pts_2d[:,::-1].astype(float), camera_K=test_camera_K)
        
        print("pnp_retval:", pnp_retval)
        if not pnp_retval:
            print("No PnP result")
            rt_matrix = np.eye(4)
        #print(translation)
        #print("rt_matrix", rt_matrix)
        #print("rt_matrix_inv", np.linalg.inv(rt_matrix))
        test_pointcloud_cam = depth_map_to_pointcloud(depths[0,0], masks[0,0], intrinsics)
        save_pointcloud(test_pointcloud_cam, "./pointclouds/test_result_%.2d_cam.txt" % step)
        test_pointcloud_cam2obj = transform_pointcloud(test_pointcloud_cam, np.linalg.inv(rt_matrix))
        save_pointcloud(test_pointcloud_cam2obj, "./pointclouds/test_result_%.2d_cam2obj.txt" % step)
        test_pointcloud_obj2cam = transform_pointcloud(pointcloud, rt_matrix)
        save_pointcloud(test_pointcloud_obj2cam, "./pointclouds/test_result_%.2d_obj2cam.txt" % step)
        test_pointcloud_gt_obj2cam = transform_pointcloud(pointcloud, gt_obj_to_cam)
        save_pointcloud(test_pointcloud_gt_obj2cam, "./pointclouds/test_result_%.2d_gt_obj2cam.txt" % step)
        test_pointcloud_gt_cam2obj = transform_pointcloud(test_pointcloud_cam, gt_cam_to_obj)
        save_pointcloud(test_pointcloud_gt_cam2obj, "./pointclouds/test_result_%.2d_gt_cam2obj.txt" % step)
        #print("gripper_rt", gripper_rt)
    
    scene_pointcloud = depth_map_to_pointcloud(depths[0,0], None, intrinsics)
    save_pointcloud(scene_pointcloud / 1000.0, "pointclouds/scene.txt")
    #t_error = np.linalg.norm(gt_obj_to_cam[:3, 3] - rt_matrix[:3, 3])
    #r, _ = cv2.Rodrigues(np.dot(gt_obj_to_cam[:3, :3], np.linalg.inv(rt_matrix[:3, :3])))
    #r_error = np.linalg.norm(r)
    
    
    R1 = gt_obj_to_cam[:3, :3]/np.cbrt(np.linalg.det(gt_obj_to_cam[:3, :3]))
    T1 = gt_obj_to_cam[:3, 3]

    R2 = rt_matrix[:3, :3]/np.cbrt(np.linalg.det(rt_matrix[:3, :3]))
    T2 = rt_matrix[:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100
    
    theta = min(theta, np.abs(180 - theta))
    metrics = {
        "r_error": theta,
        "t_error": shift
    }
    #save_pointcloud(matches_3d[:,3:].cpu().numpy(), "./pointclouds/matched_3d_pts.txt")    

    return matches_3d,rt_matrix,test_camera_K,gt_pose

def main(
        dname='ty',
        exp_name='debug',
        B=1, # batchsize
        S=64, # seqlen
        rand_frames=False,
        crop_size=(256,448),
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=True, # dataset shuffling
        is_training=True,
        log_dir='./logs_match',
        max_iters=20,
        log_freq=1,
        device_ids=[0],
):
    device = 'cuda:%d' % device_ids[0]
    
    exp_name = 'vis_seq' # Visualize every type of data in the samples: what are the different masks?

    assert(crop_size[0] % 64 == 0)
    assert(crop_size[1] % 64 == 0)
    
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
    vis_dataset = PoseDataset()
    vis_dataset = SimTestDataset(dataset_location="/root/autodl-tmp/shiqian/code/gripper/render_large_fov", features=True)
    ref_dataset = ReferenceDataset(dataset_location="/root/autodl-tmp/shiqian/code/gripper/render_lowres", num_views=840, features=True)
    vis_dataloader = DataLoader(vis_dataset, batch_size=B, shuffle=shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=shuffle)
    iterloader = iter(vis_dataloader)
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
    
    matcher = Dinov2Matcher(refs=refs, model_pointcloud=gripper_pointcloud, device=device)
    
    matches_3ds,rt_matrixs,test_camera_Ks,gt_poses = [],[],[],[]
    r_errors = []
    t_errors = []

    while global_step < max_iters:

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
            matches_3d,rt_matrix,test_camera_K,gt_pose, metrics = run_model(sample, refs, gripper_pointcloud, matcher, device, dname, global_step, sw=sw_t)
            matches_3ds.append(matches_3d)
            rt_matrixs.append(rt_matrix)
            test_camera_Ks.append(test_camera_K)
            gt_poses.append(gt_pose)
        else:
            print('sampling failed')
                  
        iter_time = time.time()-iter_start_time
        
        r_error = metrics['r_error']
        t_error = metrics['t_error']
        r_errors.append(r_error)
        t_errors.append(t_error)
        
        print('%s; step %06d/%d; itime %.2f; R_error %.2f; T_error %.2f' % (
            model_name, global_step, max_iters, iter_time, r_error, t_error))
    
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

    optimize_reproject(matches_3ds,rt_matrixs,test_camera_Ks,gt_poses)
    writer_t.close()
from utils.quaternion_utils import *

def optimize_reproject(matches_3ds,rt_matrixs,test_camera_Ks,gt_poses):
    # TODO how to optimize?
    # matches_3ds: list[tensor(342,6),...] different shape
    # other: list[np.array(4,4)or(3,3)], same shape

    rt_matrixs = torch.tensor(rt_matrixs,device=matches_3ds[0].device, dtype=matches_3ds[0].dtype)          # b,4,4
    test_camera_Ks = torch.tensor(test_camera_Ks,device=matches_3ds[0].device, dtype=matches_3ds[0].dtype)  # b,3,3
    gt_poses = torch.tensor(gt_poses,device=matches_3ds[0].device, dtype=matches_3ds[0].dtype)              # b,4,4
    q_pred = torch.tensor(matrix_to_quaternion(rt_matrixs[0][:3,:3]),requires_grad=True)    # 4
    t_pred = torch.tensor(rt_matrixs[0][:3,3],requires_grad=True) # 3
    optimizer = torch.optim.Adam([{'params':q_pred,'lr':1e-2},{'params':t_pred,'lr':1e-2}],lr=1e-4 )
    start_time = time.time()
    for iteration in range(400):
        optimizer.zero_grad()
        loss = 0
        for i in range(len(matches_3ds)):
            fact_2d = matches_3ds[i][:, 1:3].clone()    # 342,2
            fact_2d[:,[0,1]] = fact_2d[:,[1,0]]     # TODO 非常的奇怪，是因为前面有地方把这个顺序调换了？
            # use rotation matrix to change:
            # transform_pointcloud(matches_3ds[i][:, 3:], gt_poses[i]@torch.inverse(gt_poses[0])@rt_pred)

            pred_3d = quaternion_apply(matrix_to_quaternion(torch.inverse(gt_poses[i])[:3, :3]), matches_3ds[i][:, 3:]) + torch.inverse(gt_poses[i])[:3, 3]
            pred_3d = quaternion_apply(matrix_to_quaternion(gt_poses[0][:3,:3]),pred_3d) + gt_poses[0][:3,3]
            pred_3d = quaternion_apply(q_pred,pred_3d) + t_pred


            proj_2d = (torch.matmul(test_camera_Ks[i],pred_3d.transpose(1, 0))/pred_3d.transpose(1, 0)[2,:]).transpose(1, 0)[:, :2]
            loss += torch.mean(torch.norm(fact_2d - proj_2d,dim=1)) + 1e2*(1-torch.norm(q_pred))**2
            _ = torch.mean(torch.norm(fact_2d - proj_2d,dim=1))
            _unit = torch.norm(q_pred)
        loss /= len(matches_3ds)
        loss.backward()
        optimizer.step()
        if iteration == 350 :
            pass
    end_time = time.time()
    print("Time", end_time - start_time)


if __name__ == '__main__':
    Fire(main)
