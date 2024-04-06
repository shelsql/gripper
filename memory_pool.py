from test_on_sim_seq import optimize_reproject, fps_optimize_views_from_test,compute_results,run_model
from utils.quaternion_utils import *
import numpy as np
from utils.spd import sample_points_from_mesh,compute_RT_errors
from datasets.ref_dataset import ReferenceDataset, SimTestDataset, SimTrackDataset
from torch.utils.data import DataLoader
import os
from matcher import Dinov2Matcher
import time
from utils.spd import read_pointcloud
import pickle
class MemoryPool():
    def __init__(self,max_number,key_number):
        self.matches_3ds = []
        self.rt_matrixs = []
        self.test_camera_Ks = []
        self.gt_poses = []
        self.max_number = max_number
        self.key_number = key_number
    def refine_new_frame(self,frame):
        # frame: matches_3d,gt_pose,rt_matrix,camera_Ks
        self.matches_3ds.insert(0,frame['matches_3d'])
        self.rt_matrixs.insert(0,frame['rt_matrix'])
        self.test_camera_Ks.insert(0,frame['test_camera_K'])
        self.gt_poses.insert(0,frame['gt_pose'])

        # 从内存池中fps挑选k帧出来辅助优化
        key_ids = fps_optimize_views_from_test(np.array(self.gt_poses),
                                               select_numbers=min(len(self.gt_poses), self.key_number),
                                               start_idx=0)
        if len(self.matches_3ds)<self.key_number:   # 当内存池不足key_number帧时，把其他关键帧也优化一下
            for frame_id in key_ids:
                views_idx_for_opt = fps_optimize_views_from_test(np.array(self.gt_poses),
                                                                 select_numbers=len(key_ids),
                                                                 start_idx=frame_id)
                q_pred, t_pred = optimize_reproject([self.matches_3ds[i] for i in views_idx_for_opt],
                                                    [self.rt_matrixs[i] for i in views_idx_for_opt],
                                                    [self.test_camera_Ks[i] for i in views_idx_for_opt],
                                                    [self.gt_poses[i] for i in views_idx_for_opt])
                r_pred = quaternion_to_matrix(q_pred)
                rt_pred = np.zeros((4,4))
                rt_pred[:3,:3] = r_pred.cpu().detach().numpy()
                rt_pred[:3, 3] = t_pred.cpu().detach().numpy()
                rt_pred[3, 3] = 1
                self.rt_matrixs[frame_id] = rt_pred
        else:
            views_idx_for_opt = fps_optimize_views_from_test(np.array(self.gt_poses),
                                                             select_numbers=len(key_ids),
                                                             start_idx=0)
            q_pred, t_pred = optimize_reproject([self.matches_3ds[i] for i in views_idx_for_opt],
                                                [self.rt_matrixs[i] for i in views_idx_for_opt],
                                                [self.test_camera_Ks[i] for i in views_idx_for_opt],
                                                [self.gt_poses[i] for i in views_idx_for_opt])
            r_pred = quaternion_to_matrix(q_pred)
            rt_pred = np.zeros((4, 4))
            rt_pred[:3, :3] = r_pred.cpu().detach().numpy()
            rt_pred[:3, 3] = t_pred.cpu().detach().numpy()
            rt_pred[3, 3] = 1
            self.rt_matrixs[0] = rt_pred

        # 如果内存池已满，则删除其中一帧
        # 以gt_pose中的旋转为依据
        if len(self.matches_3ds) > self.max_number:
            self.eliminate_one_frame()

    def eliminate_one_frame(self):
        dist_mat = np.zeros((len(self.gt_poses), len(self.gt_poses)))
        for i, pose1 in enumerate(self.gt_poses):  # TODO batch形式
            for j, pose2 in enumerate(self.gt_poses):
                dist_mat[i, j], _ = compute_RT_errors(pose1, pose2)
        distsum = np.sum(dist_mat,axis=1,keepdims=False)
        eliminate_id = np.argsort(distsum)[0]
        self.matches_3ds.pop(eliminate_id)
        self.rt_matrixs.pop(eliminate_id)
        self.test_camera_Ks.pop(eliminate_id)
        self.gt_poses.pop(eliminate_id)


def main(
        dname='sim',
        B=1,  # batchsize
        S=1,  # seqlen
        shuffle=False,  # dataset shuffling
        ref_dir='/root/autodl-tmp/shiqian/code/gripper/ref_views/franka_69.4_840',
        test_dir='/root/autodl-tmp/shiqian/code/gripper/test_views/franka_69.4_1024',
        feat_layer=19,  # Which layer of features from dinov2 to take
        max_iters=1024,
        device_ids=[2],
        refine_mode='a',
        max_memory=50,
        key_number=10
):
    # The idea of this file is to test DinoV2 matcher and multi frame optimization on Blender rendered data
    device = 'cuda:%d' % device_ids[0]
    memory_pool = MemoryPool(max_memory,key_number)

    test_dataset = SimTrackDataset(dataset_location=test_dir, seqlen=S, features=feat_layer)
    ref_dataset = ReferenceDataset(dataset_location=ref_dir, num_views=840, features=feat_layer)
    test_dataloader = DataLoader(test_dataset, batch_size=B, shuffle=shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=shuffle)
    iterloader = iter(test_dataloader)
    # Load ref images and init Dinov2 Matcher
    refs = next(iter(ref_dataloader))
    ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3)  # B, S, C, H, W
    ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3)
    ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3)
    ref_images = torch.concat([ref_rgbs[0], ref_depths[0, :, 0:1], ref_masks[0, :, 0:1]], axis=1)
    print(ref_images.shape)
    global_step = 0
    gripper_path = "/root/autodl-tmp/shiqian/code/gripper/franka_hand_obj/franka_hand.obj"
    pointcloud_path = "./pointclouds/gripper.txt"
    if not os.path.exists(pointcloud_path):
        gripper_pointcloud = sample_points_from_mesh(gripper_path, fps=True, n_pts=8192)
    else:
        gripper_pointcloud = read_pointcloud(pointcloud_path)

    matcher = Dinov2Matcher(ref_dir=ref_dir, refs=refs,
                            model_pointcloud=gripper_pointcloud,
                            feat_layer=feat_layer,
                            device=device)

    q_preds, t_preds, gt_poses_for_result = [], [], []
    while global_step < max_iters:
        matches_3ds, rt_matrixs, test_camera_Ks, gt_poses = [], [], [], []
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0

        read_start_time = time.time()

        sample = next(iterloader)
        read_time = time.time() - read_start_time
        iter_read_time += read_time

        if sample is not None:
            matches_3ds, rt_matrixs, test_camera_Ks, gt_poses = run_model(sample, refs, gripper_pointcloud, matcher,
                                                                          device, dname, refine_mode, global_step,
                                                                          )
        else:
            print('sampling failed')
        qt_pred_for_vis_seq = []
        print("Iteration {}".format(global_step))
        frame = {'matches_3d': matches_3ds[0], 'gt_pose': gt_poses[0],'rt_matrix': rt_matrixs[0],'test_camera_K':test_camera_Ks[0]}
        # frame: matches_3d,gt_pose,rt_matrix,camera_Ks
        memory_pool.refine_new_frame(frame)



if __name__ == '__main__':
    main()