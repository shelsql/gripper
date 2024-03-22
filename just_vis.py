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
from ref_dataset import ReferenceDataset
from torch.utils.data import DataLoader

from utils.spd import sample_points_from_mesh, depth_map_to_pointcloud, save_pointcloud, transform_pointcloud, get_2dbboxes

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

def run_model(d, refs, pointcloud, device, dname, sw=None):
    metrics = {}
    
    rgbs = torch.Tensor(d['rgbs']).float().permute(0, 1, 4, 2, 3) # B, S, C, H, W
    depths = torch.Tensor(d['depths']).float().unsqueeze(2)
    masks = torch.Tensor(d['masks']).float().permute(0, 1, 4, 2, 3)
    kptss = d['kptss']
    npys = d['npys']
    intrinsics = d['intrinsics']
    
    ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3) # B, S, C, H, W
    ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3)
    ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3)
    ref_c2ws = refs['c2ws']
    ref_obj_poses = refs['obj_poses']
    ref_intrinsics = refs['intrinsics']
    #print(kptss)
    #print(npys)
    
    print(ref_rgbs.shape, ref_masks.shape, ref_depths.shape)
    print(ref_depths[:,:,0].max(), ref_depths[:,:,0].min())
    
    gripper_masks = masks[:, :, 0, :, :] >= 9 # B, S, H, W
    gripper_bboxes = get_2dbboxes(gripper_masks)
    #print(gripper_bboxes)
    print("Calculating pointclouds...")
    
    gripper_mask = masks[:, :, 0, :, :] >= 10
    print(gripper_mask.shape)
    print(intrinsics)
    
    scene_pointcloud = depth_map_to_pointcloud(depths[0,5,0], gripper_mask[0,0], intrinsics['camera_settings'][0]['intrinsic_settings'])
    save_pointcloud(scene_pointcloud / 1000.0, "pointclouds/scene.txt")
    
    gripper_info = kptss[0][0]['keypoints'][8]
    print(gripper_info)
    gripper_t = torch.tensor(gripper_info["location_wrt_cam"]).numpy()
    gripper_r = torch.tensor(gripper_info["R2C_mat"]).numpy()
    gripper_rt = np.zeros((4,4))
    gripper_rt[:3, :3] = gripper_r
    gripper_rt[:3, 3] = gripper_t
    gripper_rt[3, 3] = 1
    transformed_gripper = transform_pointcloud(pointcloud, gripper_rt)
    save_pointcloud(transformed_gripper, "pointclouds/transformed_gripper.txt")
    
    flip_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    rot90_mat = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    save_pointcloud(pointcloud, "pointclouds/gripper.txt")
    #print(ref_depths.shape)
    for i in tqdm(range(64)):
        ref_pointcloud = depth_map_to_pointcloud(ref_depths[0,i,0], ref_masks[0,i,0], ref_intrinsics)
        save_pointcloud(ref_pointcloud , "pointclouds/ref_image_%.3d.txt" % i)
        
        transformed_ref = transform_pointcloud(ref_pointcloud, ref_c2ws[0,i])
        transformed_ref = transform_pointcloud(transformed_ref, np.linalg.inv(ref_obj_poses[0,i]))
        save_pointcloud(transformed_ref , "pointclouds/transformed_ref_%.3d.txt" % i)
        #transformed_g = transform_pointcloud(pointcloud, np.dot(flip_mat, np.linalg.inv(ref_c2ws[0,26])))
        #save_pointcloud(transformed_g , "transformed_g.txt")
    
    print("Pointclouds saved. Writing gifs...")
    exit()
    B, S, C, H, W = rgbs.shape
    assert(C==3)
    
    print(rgbs.shape, masks.shape, depths.shape)
    
    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_masks = utils.improc.preprocess_color(masks*255.0 / torch.max(masks).item())
        prep_gripper_masks = utils.improc.preprocess_color(gripper_masks.unsqueeze(2).float()*255.0)
        prep_rgbs_list = [prep_rgbs[:,i] for i in range(prep_rgbs.shape[1])]
        prep_depths_list = [depths[:,i] for i in range(depths.shape[1])]
        prep_masks_list = [prep_masks[:,i] for i in range(prep_masks.shape[1])]
        prep_gripper_masks_list = [prep_gripper_masks[:,i] for i in range(prep_gripper_masks.shape[1])]
        sw.summ_rgbs("input/rgbs", prep_rgbs_list)
        sw.summ_oneds("input/depths", prep_depths_list)
        sw.summ_rgbs("input/masks", prep_masks_list)
        sw.summ_oneds("input/gripper_masks", prep_gripper_masks_list)
        
        rgb_g_vis = []
        for si in range(0,S,1):
            #print(prep_rgbs[:,si].shape, gripper_bboxes[0:1,si:si+1])
            rgb_g_vis.append(sw.summ_boxlist2d('', prep_rgbs[:,si], gripper_bboxes[0:1,si:si+1], frame_id=si, frame_str=dname, only_return=True))
            # mask_vis = sw.summ_oned('', masks_g[0:1,si], norm=False, only_return=True)
            # mask_g_vis.append(mask_vis)
        # joint_vis_g = [torch.cat([rgb, mask], dim=-1) for (rgb,mask) in zip(rgb_g_vis, mask_g_vis)]
        # sw.summ_rgbs('0_inputs/boxes_on_rgb0_and_mask0_g', joint_vis_g)
        sw.summ_rgbs('input/boxes_on_rgbs', rgb_g_vis)

    return None

def main(
        dname='ty',
        exp_name='debug',
        B=1, # batchsize
        S=32, # seqlen
        rand_frames=False,
        crop_size=(256,448),
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        is_training=True,
        log_dir='./logs_just_vis',
        max_iters=10,
        log_freq=1,
        device_ids=[0],
):
    device = 'cpu:%d' % device_ids[0]
    
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
    vis_dataset = TrackingDataset()
    #vis_dataset = 
    ref_dataset = ReferenceDataset(dataset_location="./render_lowres")
    vis_dataloader = DataLoader(vis_dataset, batch_size=B, shuffle=shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=shuffle)
    iterloader = iter(vis_dataloader)
    refs = next(iter(ref_dataloader))

    global_step = 0
    
    gripper_path = "./franka_hand_obj/franka_hand.obj"
    gripper_pointcloud = sample_points_from_mesh(gripper_path, n_pts=8192)

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
            _ = run_model(sample, refs, gripper_pointcloud, device, dname, sw=sw_t)
        else:
            print('sampling failed')
                  
        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
            
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
