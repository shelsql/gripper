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

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

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
    
    gt_pose = np.dot(np.linalg.inv(o2ws[0]), c2ws[0])
    #print("o2w:", o2ws[0])
    #print("c2w:", c2ws[0])
    print("gt_cam_to_obj:", gt_pose)
    #print("gt_cam_to_obj_inv", np.linalg.inv(gt_pose))
    
    #test_pc = depth_map_to_pointcloud(depths[0,0], ref_masks[0,0], intrinsics)
    
    #TODO cluster debug pose and PnP

    for i in range(rgbs.shape[0]):
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
        
        #print(pnp_retval)
        #print(translation)
        #print("rt_matrix", rt_matrix)
        print("rt_matrix_inv", np.linalg.inv(rt_matrix))
        test_pointcloud_cam = depth_map_to_pointcloud(depths[0,0], masks[0,0], intrinsics)
        save_pointcloud(test_pointcloud_cam, "./pointclouds/test_result_%.2d_cam.txt" % step)
        test_pointcloud_cam2obj = transform_pointcloud(test_pointcloud_cam, np.linalg.inv(rt_matrix))
        save_pointcloud(test_pointcloud_cam2obj, "./pointclouds/test_result_%.2d_cam2obj.txt" % step)
        test_pointcloud_obj2cam = transform_pointcloud(pointcloud, rt_matrix)
        save_pointcloud(test_pointcloud_obj2cam, "./pointclouds/test_result_%.2d_obj2cam.txt" % step)
        #print("gripper_rt", gripper_rt)
    
    scene_pointcloud = depth_map_to_pointcloud(depths[0,0], None, intrinsics)
    save_pointcloud(scene_pointcloud / 1000.0, "pointclouds/scene.txt")
        
    #save_pointcloud(matches_3d[:,3:].cpu().numpy(), "./pointclouds/matched_3d_pts.txt")    

    return None

def main(
        dname='ty',
        exp_name='debug',
        B=1, # batchsize
        S=64, # seqlen
        rand_frames=False,
        crop_size=(256,448),
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        is_training=True,
        log_dir='./logs_match',
        max_iters=1,
        log_freq=1,
        device_ids=[1],
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
    vis_dataset = SimTestDataset(features=True)
    ref_dataset = ReferenceDataset(dataset_location="/root/autodl-tmp/shiqian/code/gripper/render_lowres", num_views=840,features=True)
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
        
        # if sample is not None:
        #     print("got the sample", torch.sum(sample['vis_g']))
        
        if sample is not None:
            _ = run_model(sample, refs, gripper_pointcloud, matcher, device, dname, global_step, sw=sw_t)
        else:
            print('sampling failed')
                  
        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
            
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
