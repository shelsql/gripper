
import torch
torch.autograd.set_detect_anomaly(True)

from utils.quaternion_utils import *
import numpy as np
from utils.spd import sample_points_from_mesh,compute_RT_errors
from datasets.ref_dataset import ReferenceDataset, SimTestDataset, SimTrackDataset, SimVideoDataset
from datasets.ty_datasets import TrackingDataset
from torch.utils.data import DataLoader
import os
from matcher import Dinov2Matcher
import time
from utils.spd import read_pointcloud,depth_map_to_pointcloud,pairwise_distances_torch,compute_R_errors_batch
import pickle
import random
from utils.geometric_vision import solve_pnp_ransac
import argparse
import scipy
import sys
sys.path.append('/root/autodl-tmp/tianshuwu/gripper/try_pybind11/build')
import try_pybind11
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
class MemoryPool():
    def __init__(self,cfg):
        self.cfg = cfg
        self.matches_3ds = []
        self.pred_poses = []
        self.test_camera_Ks = []
        self.gt_poses = []
        self.paths = []
        if self.cfg.use_depth:
            self.keypoint_from_depths = []
            self.depths = []
            if self.cfg.use_full_depth:
                self.fullpoint_from_depths = []

        self.initial_pose = None
        self.relative_poses = []        # init_gt 2 gt

        self.last_pred_pose = None
        self.last_rela_pose = None

        self.q_for_compute_result = []
        self.t_for_compute_result = []
        self.gt_for_compute_result = []
        self.q_before_opt = []
        self.t_before_opt = []

        self.max_number = cfg.max_number
        self.key_number = cfg.key_number
        self.gripper_point = torch.tensor(read_pointcloud(cfg.gripper_point_path),device=cfg.device)
        self.noise = None

    def refine_new_frame(self,frame,record_vis,global_step,vis_dict=None):
        # frame: matches_3d,gt_pose,rt_matrix,camera_Ks
        self.matches_3ds.insert(0,frame['matches_3d'])      # tensor(98,6) different shape
        self.pred_poses.insert(0,frame['rt_matrix'])        # np array
        self.test_camera_Ks.insert(0,frame['test_camera_K'])# np array
        self.gt_poses.insert(0,frame['gt_pose'])            # np array
        self.paths.insert(0,frame['path'])
        if self.initial_pose is None:
            self.initial_pose = frame['gt_pose']  # np array
        if self.cfg.noise == "none":
            self.relative_poses.insert(0, frame['gt_pose'] @ np.linalg.inv(self.initial_pose))
        elif self.cfg.noise == "random":
            noise_r = np.eye(3)
            random_rotation = scipy.spatial.transform.Rotation.from_rotvec(np.random.randn(3) * 0.1)    # 0.1=5度
            noise_r = random_rotation.apply(noise_r)
            noise_t = np.random.randn(3)*0.02
            random_rt = np.eye(4)
            random_rt[:3,:3] = noise_r
            random_rt[:3,3] = noise_t
            self.relative_poses.insert(0, random_rt @ frame['gt_pose'] @ np.linalg.inv(self.initial_pose))
        elif self.cfg.noise == "incremental":
            if self.noise is None:
                noise_r = np.eye(3)
                random_rotation = scipy.spatial.transform.Rotation.from_rotvec(np.random.randn(3) * 0.001)    # 0.1=5度
                noise_r = random_rotation.apply(noise_r)
                noise_t = np.random.randn(3)*0.0001
                random_rt = np.eye(4)
                random_rt[:3,:3] = noise_r
                random_rt[:3,3] = noise_t
                self.init_noise = random_rt
                self.noise = random_rt
            else:
                self.noise = self.init_noise @ self.noise
            self.relative_poses.insert(0, self.noise @ frame['gt_pose'] @ np.linalg.inv(self.initial_pose))


        # self.point_from_depth.insert(0,torch.tensor(frame['point_from_depth'],device=self.cfg.device))
        if self.cfg.use_depth:
            self.keypoint_from_depths.insert(0,frame['keypoint_from_depth'])    # np array
            self.depths.insert(0,frame['depth'])
            if self.cfg.use_full_depth:
                self.fullpoint_from_depths.insert(0,frame['fullpoint_from_depth'])  # np array


        # 从内存池中fps挑选k帧出来辅助优化
        qt_pred_for_vis_seq = [] if record_vis else None
        views_idx_for_opt = fps_optimize_views_from_test(np.array(self.pred_poses),     # 花费0.0007s
                                                         select_numbers=min(len(self.pred_poses), self.key_number),
                                                         start_idx=0)
        # q_pred, t_pred = self.c_optimize(views_idx_for_opt)
        # q_pred = torch.tensor(q_pred)
        # t_pred = torch.tensor(t_pred)
        if self.cfg.single_opt:
            q_pred, t_pred = self.optimize(np.array([0]), qt_pred_for_vis_seq=qt_pred_for_vis_seq)
            r_pred = quaternion_to_matrix(q_pred)
            rt_pred = np.zeros((4, 4))
            rt_pred[:3, :3] = r_pred.cpu().detach().numpy()
            rt_pred[:3, 3] = t_pred.cpu().detach().numpy()
            rt_pred[3, 3] = 1
            self.pred_poses[0] = rt_pred
        else:
            q_pred, t_pred = self.optimize(views_idx_for_opt,frame['rubbish_flag'],qt_pred_for_vis_seq)   # 0.05s

        r_pred = quaternion_to_matrix(q_pred)
        rt_pred = np.zeros((4, 4))
        rt_pred[:3, :3] = r_pred.cpu().detach().numpy()
        rt_pred[:3, 3] = t_pred.cpu().detach().numpy()
        rt_pred[3, 3] = 1
        self.pred_poses[0] = rt_pred

        if self.cfg.adjust:
            # q_preds, t_preds = self.bundle_adjust()
            qt_pred_for_vis_seq_adjust = [] if record_vis else None
            views_idx_for_opt = nearest_view_select(np.array(self.pred_poses),  # 花费0.0007s
                                                             select_numbers=min(len(self.pred_poses), self.key_number),
                                                             start_idx=0)
            q_preds, t_preds = self.adjust(views_idx_for_opt,qt_pred_for_vis_seq_adjust)
            r_preds = quaternion_to_matrix(q_preds)
            rt_preds = np.zeros((len(views_idx_for_opt), 4,4))
            rt_preds[:, :3, :3] = r_preds.cpu().detach().numpy()
            rt_preds[:, :3, 3] = t_preds.cpu().detach().numpy()
            rt_preds[:, 3, 3] = 1
            for i,view_id in enumerate(views_idx_for_opt):
                self.pred_poses[view_id] = rt_preds[i]

        if vis_dict != None:
            vis_dict['qt_preds'] = qt_pred_for_vis_seq
            if self.cfg.adjust:
                vis_dict['qt_preds_adjust'] = qt_pred_for_vis_seq_adjust
            if not os.path.exists(f'vis_results/memory_pool_tydata'):
                os.makedirs(f'vis_results/memory_pool_tydata')
            with open(f'vis_results/memory_pool_tydata/step{global_step}.pkl', 'wb') as f:
                pickle.dump(vis_dict, f)

        self.q_for_compute_result.append(matrix_to_quaternion(torch.tensor(self.pred_poses[0][:3,:3])).cpu().detach().numpy())
        self.t_for_compute_result.append(self.pred_poses[0][:3,3])
        self.gt_for_compute_result.append(self.gt_poses[0])

        # 不论高低精度，都记录上一帧的rela_pose和predpose
        self.last_pred_pose = rt_pred
        self.last_rela_pose = self.relative_poses[0]

        # 如果内存池已满，则删除其中一帧
        if frame['rubbish_flag']:
            self.eliminate_one_frame(0)
        if len(self.matches_3ds) > self.max_number:
            self.eliminate_one_frame()



    def eliminate_one_frame(self,eliminate_id=None):
        if eliminate_id == None:
            # 用pred_pose选择要删除的帧
            dist_mat = compute_R_errors_batch(np.array(self.pred_poses),np.array(self.pred_poses))
            # dist_mat = np.zeros((len(self.pred_poses), len(self.pred_poses)))
            # for i, pose1 in enumerate(self.pred_poses):  # TODO batch形式
            #     for j, pose2 in enumerate(self.pred_poses):
            #         dist_mat[i, j], _ = compute_RT_errors(pose1, pose2)
            distsum = np.sum(dist_mat,axis=1,keepdims=False)
            eliminate_id = np.argsort(distsum)[0]   # 删掉的是提供额外信息最少的帧

        self.matches_3ds.pop(eliminate_id)
        self.pred_poses.pop(eliminate_id)

        self.test_camera_Ks.pop(eliminate_id)
        self.paths.pop(eliminate_id)
        if self.cfg.use_depth:
            self.keypoint_from_depths.pop(eliminate_id)
            self.depths.pop(eliminate_id)
            if self.cfg.use_full_depth:
                self.fullpoint_from_depths.pop(eliminate_id)
        self.gt_poses.pop(eliminate_id)

        self.relative_poses.pop(eliminate_id)

    def eliminate_all_frames_and_compute_result(self,result_name):
        while len(self.matches_3ds) > 0:
            self.eliminate_one_frame()
        compute_results(torch.tensor(np.stack(self.q_for_compute_result)),
                        torch.tensor(np.stack(self.t_for_compute_result)),
                        torch.tensor(np.stack(self.gt_for_compute_result)))

        results = np.concatenate([np.stack(self.q_for_compute_result),
                             np.stack(self.t_for_compute_result),
                             np.stack(self.gt_for_compute_result,).reshape(-1, 16),
                                  # np.stack(self.q_before_opt),
                                  # np.stack(self.t_before_opt)
                                  ],axis=1)  # 1024,4+3+16(+4+3)
        self.q_for_compute_result = []
        self.t_for_compute_result = []
        self.gt_for_compute_result = []
        self.initial_pose = None
        self.last_pred_pose = None
        self.last_rela_pose = None
        if not os.path.exists(f'results'):
            os.makedirs(f'results')

        if not os.path.exists(f'results/memory'):
            os.makedirs(f'results/memory')
        if self.cfg.single_opt:
            result_name = result_name + 'single'
        if not self.cfg.use_depth:
            result_name = result_name + 'nodepth'
        np.savetxt(f'results/memory/{result_name}_new.txt', results)

    def optimize(self,views_idx_for_opt,rubbish_flag=False,qt_pred_for_vis_seq=None):
        '''删除垃圾点的迭代方法，优化的是输入list的第一帧，后面的帧仅辅助'''
        # matches_3ds: list[tensor(342,6),...] different shape
        # other: list[np.array(4,4)or(3,3)], same shape
        time0 = time.time()
        matches_3ds = [self.matches_3ds[i] for i in views_idx_for_opt]   # list[tensor(342,6),...] different shape
        if self.cfg.use_depth:
            keypoint_from_depths = [torch.tensor(self.keypoint_from_depths[i],dtype=self.cfg.dtype,device=self.cfg.device) for i in views_idx_for_opt]        # list[tensor(98,3),...] different shape

            if self.cfg.use_full_depth:
                fullpoint_from_depths = [torch.tensor(self.fullpoint_from_depths[i],dtype=self.cfg.dtype,device=self.cfg.device) for i in views_idx_for_opt]      # list[tensor(2728,3)]

        # pred_poses = torch.stack([torch.tensor(self.pred_poses[i],dtype=cfg.dtype,device=cfg.device) for i in views_idx_for_opt])            # b,4,4
        # q_preds = matrix_to_quaternion(pred_poses[:,:3,:3])
        # t_preds = pred_poses[:,:3,3]
        test_camera_Ks = torch.stack([torch.tensor(self.test_camera_Ks[i],dtype=self.cfg.dtype,device=self.cfg.device) for i in views_idx_for_opt])   # b,3,3


        # 因为是tracking，所以不管高低精度都用上一帧加相对位移作为初值（改一下，低精度干脆完全不用相对位移算了）
        # 注意gt要取逆，作用方式才和pred_pose一样 TODO 检查
        if self.cfg.init == 'rela':
            if self.last_pred_pose is not None:
                rt_init = self.last_pred_pose @ self.last_rela_pose @ np.linalg.inv(self.relative_poses[0])
            else:
                rt_init = self.pred_poses[0]
        # elif self.cfg.init == 'pnp':
        #     if rubbish_flag:
        #         rt_init = self.last_pred_pose
        #     else:
        #         rt_init = self.pred_poses[0]


        rt_init = torch.tensor(rt_init,device=self.cfg.device,dtype=self.cfg.dtype)
        q_pred = matrix_to_quaternion(rt_init[:3, :3])  # 4
        t_pred = rt_init[:3, 3]  # 3
        # 这段不要删，用来看优化前的效果
        # self.q_before_opt.append(q_pred.cpu().numpy())
        # self.t_before_opt.append(t_pred.cpu().numpy())
        time1 = time.time()

        if self.cfg.rela_mode == "gt":        # 高精度时，使用相对于数据集第一帧的pose 计算相对pose 来优化
            relative_poses_tmp = torch.stack([torch.tensor(self.relative_poses[i],dtype=self.cfg.dtype,device=self.cfg.device) for i in views_idx_for_opt])                # b,4,4
            q_relatives = matrix_to_quaternion(relative_poses_tmp[:, :3, :3])  # b,4
            t_relatives = relative_poses_tmp[:, :3, 3]                  # b,3
            q_rela = q_relatives[0]
            t_rela = t_relatives[0]
        else:                                   # 低精度时，使用pred_pose 计算相对pose 来优化
            relative_poses_tmp = torch.stack([torch.inverse(torch.tensor(self.pred_poses[i],dtype=self.cfg.dtype,device=self.cfg.device)) for i in views_idx_for_opt])                # b,4,4
            q_relatives = matrix_to_quaternion(relative_poses_tmp[:, :3, :3])  # b,4
            t_relatives = relative_poses_tmp[:, :3, 3]                  # b,3
            q_rela = q_relatives[0]
            t_rela = t_relatives[0]
        time2 = time.time()     # 20ms~30ms
        # step1: refine with keypoints
        # 也是先剔去离群点再优化，这样在c++中也方便实现
        # 以及在外部就把点云旋转好，进去的话只用左乘pred
        matches_3ds_within_3sigma = []
        keypoint_from_depths_within_3sigma = []
        gripper_rela = []   # 进去只用左乘pred
        for i in range(len(matches_3ds)):
            q_rela_tmp = quaternion_multiply(q_rela, quaternion_invert(q_relatives[i]))
            t_rela_tmp = quaternion_apply(q_rela_tmp, -t_relatives[i]) + t_rela
            q_pred_tmp = quaternion_multiply(q_pred,q_rela_tmp)
            t_pred_tmp = quaternion_apply(q_pred,t_rela_tmp) + t_pred

            fact_2d = matches_3ds[i][:, 1:3].clone()  # 342,2
            # pred_3dkey = quaternion_apply(quaternion_invert(q_relatives[i]),
            #                               matches_3ds[i][:, 3:]) - quaternion_apply(
            #     quaternion_invert(q_relatives[i]), t_relatives[i])
            # pred_3dkey = quaternion_apply(q_rela, pred_3dkey) + t_rela
            pred_3dkey = quaternion_apply(q_pred_tmp, matches_3ds[i][:, 3:]) + t_pred_tmp

            if self.cfg.use_depth:
                key3d_dis = torch.norm(keypoint_from_depths[i] - pred_3dkey, dim=-1)
                mean = key3d_dis.mean()
                std = key3d_dis.std()
                within_3sigma_3d = (key3d_dis >= mean-1*std) & (key3d_dis <= mean+1*std)
                if self.cfg.use_full_depth:
                    pred_3dfull = quaternion_apply(q_pred_tmp,self.gripper_point) + t_pred_tmp
                    chamfer_dis = chamfer_distance(fullpoint_from_depths[i],pred_3dfull)
                    mean = chamfer_dis.mean()
                    std = chamfer_dis.std()
                    within_3sigma_full = (chamfer_dis >= mean-1*std) & (chamfer_dis<=mean+1*std)
                    gripper_rela.append(quaternion_apply(q_rela_tmp,self.gripper_point) + t_rela_tmp)
                    fullpoint_from_depths[i] = fullpoint_from_depths[i][within_3sigma_full]

            proj_2d = (torch.matmul(test_camera_Ks[i], pred_3dkey.transpose(1, 0)) / pred_3dkey.transpose(1, 0)[2,
                                                                                     :]).transpose(1, 0)[:,:2]  # 220,2
            reproj_dis = torch.norm(fact_2d - proj_2d, dim=1)  # 220,
            mean = reproj_dis.mean()
            std = reproj_dis.std()
            within_3sigma_2d = (reproj_dis >= mean-1*std) & (reproj_dis <= mean+1*std)
            within_3sigma = (within_3sigma_2d & within_3sigma_3d) if self.cfg.use_depth else within_3sigma_2d
            matches_3ds_within_3sigma.append(matches_3ds[i][within_3sigma])
            # q_rela_tmp = quaternion_multiply(q_rela, quaternion_invert(q_relatives[i]))
            # t_rela_tmp = quaternion_apply(q_rela_tmp, -t_relatives[i]) + t_rela
            matches_3ds_within_3sigma[i][:, 3:] = quaternion_apply(q_rela_tmp,
                                                                   matches_3ds_within_3sigma[i][:, 3:]) + t_rela_tmp
            if self.cfg.use_depth:
                keypoint_from_depths_within_3sigma.append(keypoint_from_depths[i][within_3sigma])
        time3 = time.time()     # 15 ms
        if self.cfg.use_cpp:
            camera_Ks = torch.zeros((len(matches_3ds), 4),device=self.cfg.device)
            camera_Ks[:, [0, 1, 2, 3]] = test_camera_Ks[:, [0, 0, 1, 1], [0, 2, 1, 2]]
            # 在外部就把点云旋转好，进去的话只用左乘pred
            # q_rela_list, t_rela_list = [], []
            # for i in range(len(matches_3ds)):
            #     q_rela_tmp = quaternion_multiply(q_rela,quaternion_invert(q_relatives[i]))
            #     t_rela_tmp = quaternion_apply(q_rela_tmp,-t_relatives[i]) + t_rela
            #     matches_3ds_within_3sigma[i][:,3:] = quaternion_apply(q_rela_tmp,matches_3ds_within_3sigma[i][:,3:] ) + t_rela_tmp
            time4 = time.time()
            if self.cfg.use_depth:
                q_pred,t_pred = try_pybind11.optimize_step1_usedepth(
                    [tmp[:,1:3].tolist() for tmp in matches_3ds_within_3sigma],
                    [tmp[:,3:].tolist() for tmp in matches_3ds_within_3sigma],
                    camera_Ks.tolist(),
                    q_pred.tolist(),
                    t_pred.tolist(),
                    [tmp.tolist() for tmp in keypoint_from_depths_within_3sigma]
                )
                q_pred = torch.tensor(q_pred,device=self.cfg.device, dtype=self.cfg.dtype)
                t_pred = torch.tensor(t_pred,device=self.cfg.device,dtype=self.cfg.dtype)
            else:
                q_pred,t_pred = try_pybind11.optimize_step1_nodepth(
                    [tmp[:,1:3].tolist() for tmp in matches_3ds_within_3sigma],
                    [tmp[:,3:].tolist() for tmp in matches_3ds_within_3sigma],
                    camera_Ks.tolist(),
                    q_pred.tolist(),
                    t_pred.tolist(),
                )
                q_pred = torch.tensor(q_pred,device=self.cfg.device, dtype=self.cfg.dtype)
                t_pred = torch.tensor(t_pred,device=self.cfg.device,dtype=self.cfg.dtype)
            time5 = time.time()
            print(f'time0{time1-time0} time1{time2-time1} time2{time3-time2} time3{time4-time3} time4{time5-time4} ')
        else:
            q_pred = q_pred.requires_grad_()
            t_pred = t_pred.requires_grad_()
            start_time = time.time()
            iteration = 0
            loss_change = 1
            loss_last = 0
            qt_pred_for_vis_frame = []
            optimizer = torch.optim.Adam([q_pred, t_pred], lr=2e-3)
            while iteration < 100 and abs(loss_change) > 1e-2:
                if self.cfg.record_vis:
                    qt_pred_for_vis_frame.append((q_pred.tolist(), t_pred.tolist()))
                optimizer.zero_grad()
                reproj_dis_list = []
                key3d_dis_list = []
                for i in range(len(matches_3ds)):
                    # fact_2d = matches_3ds_within_3sigma[i][:, 1:3].clone()  # 342,2
                    # pred_3dkey = quaternion_apply(quaternion_invert(q_relatives[i]),matches_3ds_within_3sigma[i][:, 3:]) - quaternion_apply(quaternion_invert(q_relatives[i]),t_relatives[i])
                    # pred_3dkey = quaternion_apply(q_rela,pred_3dkey) + t_rela
                    pred_3dkey = quaternion_apply(q_pred,matches_3ds_within_3sigma[i][:, 3:]) + t_pred

                    if self.cfg.use_depth:
                        key3d_dis = torch.norm(keypoint_from_depths_within_3sigma[i] - pred_3dkey, dim=-1)
                        key3d_dis_list.append(key3d_dis)

                    proj_2d = (torch.matmul(test_camera_Ks[i], pred_3dkey.transpose(1, 0)) / pred_3dkey.transpose(1, 0)[2,
                                                                                             :]).transpose(1, 0)[:,
                              :2]  # 220,2
                    reproj_dis = torch.norm(matches_3ds_within_3sigma[i][:, 1:3] - proj_2d, dim=1)  # 220,
                    reproj_dis_list.append(reproj_dis)
                reproj_dis_list = torch.cat(reproj_dis_list, dim=0)
                q_loss = 1e5 * (1 - torch.norm(q_pred)) ** 2
                reproj_loss = torch.mean(reproj_dis_list)
                # c_loss = 1000*torch.mean(chamfer_dis_list)
                if self.cfg.use_depth:
                    key3d_dis_list = torch.cat(key3d_dis_list,dim=0)
                    key3d_loss = 1000 * torch.mean(key3d_dis_list)
                    loss = q_loss + reproj_loss + key3d_loss

                    # print(iteration, 'q_loss', q_loss.item(), 'reproj', reproj_loss.item(), 'key3d_loss', key3d_loss.item(),loss.item())
                else:
                    loss = q_loss + reproj_loss

                    # print(iteration, 'q_loss', q_loss.item(), 'reproj', reproj_loss.item(),  loss.item())

                loss.backward()
                optimizer.step()
                # scheduler.step()

                loss_change = loss - loss_last
                loss_last = loss
                iteration += 1
        # step2: refine with full depth image
        # 先剔去离群点再优化，而不是在优化过程中筛选

        if self.cfg.use_depth:
            if self.cfg.use_full_depth:
                # 也在外部旋转好再送进去,同时筛去离群点

                # fullpoint_from_depths_in3sigma_list = []
                # for i in range(len(matches_3ds)):
                #     pred_3dfull = quaternion_apply(quaternion_invert(q_relatives[i]),
                #                                    self.gripper_point) - quaternion_apply(
                #         quaternion_invert(q_relatives[i]), t_relatives[i])
                #     pred_3dfull = quaternion_apply(q_rela, pred_3dfull) + t_rela
                #     pred_3dfull = quaternion_apply(q_pred, pred_3dfull) + t_pred
                #     chamfer_dis = chamfer_distance(fullpoint_from_depths[i], pred_3dfull)
                #     mean = chamfer_dis.mean()
                #     std = chamfer_dis.std()
                #     within_3sigma = (chamfer_dis >= mean - 1 * std) & (chamfer_dis <= mean + 1 * std)
                #     fullpoint_from_depths_in3sigma_list.append(fullpoint_from_depths[i][within_3sigma])


                # 有问题
                # if self.cfg.use_cpp:
                #     q_pred, t_pred = try_pybind11.optimize_step2(
                #         q_pred.tolist(),
                #         t_pred.tolist(),
                #         [tmp.tolist() for tmp in fullpoint_from_depths],
                #         [tmp.tolist() for tmp in gripper_rela]
                #     )
                #     q_pred = torch.tensor(q_pred, device=self.cfg.device, dtype=self.cfg.dtype)
                #     t_pred = torch.tensor(t_pred, device=self.cfg.device, dtype=self.cfg.dtype)
                # else:
                q_pred = q_pred.requires_grad_()
                t_pred = t_pred.requires_grad_()

                optimizer = torch.optim.Adam([q_pred, t_pred], lr=2e-3)
                iteration = 0
                loss_change = 1
                loss_last = 0

                while iteration < 100 and abs(loss_change) > 5e-3:
                    if self.cfg.record_vis:
                        qt_pred_for_vis_frame.append((q_pred.tolist(), t_pred.tolist()))
                    optimizer.zero_grad()
                    chamfer_dis_list = []
                    for i in range(len(matches_3ds)):
                        # pred_3dfull = quaternion_apply(quaternion_invert(q_relatives[i]),
                        #                                self.gripper_point) - quaternion_apply(
                        #     quaternion_invert(q_relatives[i]), t_relatives[i])
                        # pred_3dfull = quaternion_apply(q_rela, pred_3dfull) + t_rela
                        pred_3dfull = quaternion_apply(q_pred, gripper_rela[i]) + t_pred

                        chamfer_dis = chamfer_distance(fullpoint_from_depths[i], pred_3dfull)
                        chamfer_dis_list.append(chamfer_dis)
                    chamfer_dis_list = torch.cat(chamfer_dis_list, dim=0)
                    # print(chamfer_dis_list.shape)

                    q_loss = 1e5 * (1 - torch.norm(q_pred)) ** 2
                    c_loss = 1000*torch.mean(chamfer_dis_list)

                    loss = q_loss + c_loss  # 问题出在：fullpoint_from_depths中的点数越来越多
                    # print(iteration, 'q_loss', q_loss.item(), 'c_loss', c_loss.item(), loss.item())
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    loss_change = loss.item() - loss_last
                    loss_last = loss.item()
                    iteration += 1

        if qt_pred_for_vis_seq is not None:
            # 每次只挑30帧左右可视化
            if not self.cfg.use_cpp:
                if len(qt_pred_for_vis_frame) > 27:
                    first_qt = qt_pred_for_vis_frame.pop(0)
                    last_qt = qt_pred_for_vis_frame.pop()
                    step = len(qt_pred_for_vis_frame) // 25
                    qt_pred_for_vis_frame = [qt_pred_for_vis_frame[i * step] for i in range(25)]
                    qt_pred_for_vis_frame.insert(0, first_qt)
                    qt_pred_for_vis_frame.append(last_qt)
                    qt_pred_for_vis_seq.append(qt_pred_for_vis_frame)
                else:
                    qt_pred_for_vis_seq.append(qt_pred_for_vis_frame)
                end_time = time.time()
                print("Time", end_time - start_time)
                print('num_of_iter', iteration)
        return q_pred, t_pred

    def bundle_adjust(self):

        # bundle adjust就不在外面转了

        matches_3ds = self.matches_3ds  # list[tensor(342,6),...] different shape
        if self.cfg.use_depth:
            keypoint_from_depths = [torch.tensor(tmp, dtype=self.cfg.dtype, device=self.cfg.device) for tmp in self.keypoint_from_depths]  # list[tensor(98,3),...] different shape
            depths = [torch.tensor(tmp, dtype=self.cfg.dtype, device=self.cfg.device) for tmp in self.depths]
        pred_poses = torch.stack(
            [torch.tensor(tmp, dtype=self.cfg.dtype, device=self.cfg.device) for tmp in self.pred_poses])  # b,4,4
        q_preds = matrix_to_quaternion(pred_poses[:, :3, :3]).requires_grad_()
        t_preds = pred_poses[:, :3, 3].requires_grad_()
        test_camera_Ks = torch.stack([torch.tensor(tmp, dtype=self.cfg.dtype, device=self.cfg.device) for tmp in
                                      self.test_camera_Ks])  # b,3,3

        matches_3ds_within_3sigma = []
        keypoint_from_depths_within_3sigma = []
        for i in range(len(matches_3ds)):
            q_pred_tmp = q_preds[i]
            t_pred_tmp = t_preds[i]

            fact_2d = matches_3ds[i][:, 1:3].clone()  # 342,2
            # pred_3dkey = quaternion_apply(quaternion_invert(q_relatives[i]),
            #                               matches_3ds[i][:, 3:]) - quaternion_apply(
            #     quaternion_invert(q_relatives[i]), t_relatives[i])
            # pred_3dkey = quaternion_apply(q_rela, pred_3dkey) + t_rela
            pred_3dkey = quaternion_apply(q_pred_tmp, matches_3ds[i][:, 3:]) + t_pred_tmp

            if self.cfg.use_depth:
                key3d_dis = torch.norm(keypoint_from_depths[i] - pred_3dkey, dim=-1)
                mean = key3d_dis.mean()
                std = key3d_dis.std()
                within_3sigma_3d = (key3d_dis >= mean-1*std) & (key3d_dis <= mean+1*std)

            proj_2d = (torch.matmul(test_camera_Ks[i], pred_3dkey.transpose(1, 0)) / pred_3dkey.transpose(1, 0)[2,
                                                                                     :]).transpose(1, 0)[:,:2]  # 220,2
            reproj_dis = torch.norm(fact_2d - proj_2d, dim=1)  # 220,
            mean = reproj_dis.mean()
            std = reproj_dis.std()
            within_3sigma_2d = (reproj_dis >= mean-1*std) & (reproj_dis <= mean+1*std)
            within_3sigma = (within_3sigma_2d & within_3sigma_3d) if self.cfg.use_depth else within_3sigma_2d
            matches_3ds_within_3sigma.append(matches_3ds[i][within_3sigma])

            if self.cfg.use_depth:
                keypoint_from_depths_within_3sigma.append(keypoint_from_depths[i][within_3sigma])

        if self.cfg.use_cpp:
            camera_Ks = torch.zeros((len(matches_3ds), 4),device=self.cfg.device)
            camera_Ks[:, [0, 1, 2, 3]] = test_camera_Ks[:, [0, 0, 1, 1], [0, 2, 1, 2]]
            q_preds,t_preds = try_pybind11.adjust(
                [tmp[:,1:3].tolist() for tmp in matches_3ds_within_3sigma],
                [tmp[:, 3:].tolist() for tmp in matches_3ds_within_3sigma],
                camera_Ks.tolist(),
                q_preds.tolist(),
                t_preds.tolist()
            )
            q_preds = torch.tensor(q_preds,device=self.cfg.device,dtype=self.cfg.dtype).reshape(-1,4)
            t_preds = torch.tensor(t_preds, device=self.cfg.device, dtype=self.cfg.dtype).reshape(-1,3)
        else:
            optimizer = torch.optim.Adam([ q_preds, t_preds], lr=2e-3)
            start_time = time.time()
            iteration = 0
            loss_change = 1
            loss_last = 0
            # step1: refine with keypoints
            while iteration < 100 and abs(loss_change) > 1e-2:
                optimizer.zero_grad()
                reproj_dis_list = []
                key3d_dis_list = []
                for i in range(len(matches_3ds)):
                    fact_2d = matches_3ds[i][:,1:3].clone()
                    pred_3dkey = quaternion_apply(q_preds[i],matches_3ds[i][:,3:]) + t_preds[i]
                    proj_2d = (torch.matmul(test_camera_Ks[i], pred_3dkey.transpose(1, 0)) / pred_3dkey.transpose(1, 0)[2,:]).transpose(1, 0)[:,:2]  # 220,2
                    reproj_dis = torch.norm(fact_2d - proj_2d, dim=1)  # 220,
                    reproj_dis_list.append(reproj_dis)
                    if self.cfg.use_depth:
                        key3d_dis = torch.norm(keypoint_from_depths[i] - pred_3dkey,dim=-1)
                        key3d_dis_list.append(key3d_dis)

                reproj_dis_list = torch.cat(reproj_dis_list, dim=0)
                mean = reproj_dis_list.mean()
                std = reproj_dis_list.std()
                within_3sigma = (reproj_dis_list >= mean - 1 * std) & (reproj_dis_list <= mean + 1 * std)
                reproj_dis_list = reproj_dis_list[within_3sigma]  # 删除垃圾点

                if self.cfg.use_depth:
                    key3d_dis_list = torch.cat(key3d_dis_list, dim=0)
                    mean = key3d_dis_list.mean()
                    std = key3d_dis_list.std()
                    within_3sigma = (key3d_dis_list >= mean - 1 * std) & (key3d_dis_list <= mean + 1 * std)
                    key3d_dis_list = key3d_dis_list[within_3sigma]

                q_loss = 1e5 * (1 - torch.norm(q_preds,dim=-1).mean()) ** 2
                reproj_loss = torch.mean(reproj_dis_list)

                if self.cfg.use_depth:
                    key3d_loss = 1000 * torch.mean(key3d_dis_list)
                    loss = q_loss + reproj_loss + key3d_loss
                    # print(iteration, 'q_loss', q_loss.item(), 'reproj', reproj_loss.item(), 'key3d_loss', key3d_loss.item(),loss.item())
                else:
                    loss = q_loss + reproj_loss
                    # print(iteration, 'q_loss', q_loss.item(), 'reproj', reproj_loss.item(),  loss.item())

                loss.backward()
                optimizer.step()
                # scheduler.step()

                loss_change = loss - loss_last
                loss_last = loss
                iteration += 1
        return q_preds, t_preds

    def adjust(self,views_idx_for_opt,qt_pred_for_vis_seq_adjust):

        matches_3ds = [self.matches_3ds[i] for i in views_idx_for_opt]  # list[tensor(342,6),...] different shape
        if self.cfg.use_depth:
            keypoint_from_depths = [torch.tensor(self.keypoint_from_depths[i], dtype=self.cfg.dtype, device=self.cfg.device) for i in views_idx_for_opt]  # list[tensor(98,3),...] different shape
            depths = [torch.tensor(self.depths[i], dtype=self.cfg.dtype, device=self.cfg.device) for i in views_idx_for_opt]
        pred_poses = torch.stack(
            [torch.tensor(self.pred_poses[i], dtype=self.cfg.dtype, device=self.cfg.device) for i in views_idx_for_opt])  # b,4,4
        q_preds = matrix_to_quaternion(pred_poses[:, :3, :3]).requires_grad_()
        t_preds = pred_poses[:, :3, 3].requires_grad_()
        test_camera_Ks = torch.stack([torch.tensor(self.test_camera_Ks[i], dtype=self.cfg.dtype, device=self.cfg.device) for i in views_idx_for_opt])  # b,3,3
        self.q_before_opt.append(q_preds[0].detach().cpu().numpy())
        self.t_before_opt.append(t_preds[0].detach().cpu().numpy())
        matches_3ds_within_3sigma = []
        keypoint_from_depths_within_3sigma = []
        for i in range(len(matches_3ds)):
            q_pred_tmp = q_preds[i]
            t_pred_tmp = t_preds[i]

            fact_2d = matches_3ds[i][:, 1:3].clone()  # 342,2
            # pred_3dkey = quaternion_apply(quaternion_invert(q_relatives[i]),
            #                               matches_3ds[i][:, 3:]) - quaternion_apply(
            #     quaternion_invert(q_relatives[i]), t_relatives[i])
            # pred_3dkey = quaternion_apply(q_rela, pred_3dkey) + t_rela
            pred_3dkey = quaternion_apply(q_pred_tmp, matches_3ds[i][:, 3:]) + t_pred_tmp

            if self.cfg.use_depth:
                key3d_dis = torch.norm(keypoint_from_depths[i] - pred_3dkey, dim=-1)
                mean = key3d_dis.mean()
                std = key3d_dis.std()
                within_3sigma_3d = (key3d_dis >= mean-1*std) & (key3d_dis <= mean+1*std)

            proj_2d = (torch.matmul(test_camera_Ks[i], pred_3dkey.transpose(1, 0)) / pred_3dkey.transpose(1, 0)[2,
                                                                                     :]).transpose(1, 0)[:,:2]  # 220,2
            reproj_dis = torch.norm(fact_2d - proj_2d, dim=1)  # 220,
            mean = reproj_dis.mean()
            std = reproj_dis.std()
            within_3sigma_2d = (reproj_dis >= mean-1*std) & (reproj_dis <= mean+1*std)
            within_3sigma = (within_3sigma_2d & within_3sigma_3d) if self.cfg.use_depth else within_3sigma_2d
            matches_3ds_within_3sigma.append(matches_3ds[i][within_3sigma])

            if self.cfg.use_depth:
                keypoint_from_depths_within_3sigma.append(keypoint_from_depths[i][within_3sigma])

        if self.cfg.use_cpp:
            camera_Ks = torch.zeros((len(matches_3ds), 4),device=self.cfg.device)
            camera_Ks[:, [0, 1, 2, 3]] = test_camera_Ks[:, [0, 0, 1, 1], [0, 2, 1, 2]]
            q_preds,t_preds = try_pybind11.adjust(
                # [tmp[:,1:3].tolist() for tmp in matches_3ds_within_3sigma],
                [tmp[:, 3:].tolist() for tmp in matches_3ds_within_3sigma],
                camera_Ks.tolist(),
                q_preds.tolist(),
                t_preds.tolist(),
                [tmp.tolist() for tmp in keypoint_from_depths_within_3sigma],
                [tmp.reshape(-1).tolist() for tmp in depths],

            )
            q_preds = torch.tensor(q_preds,device=self.cfg.device,dtype=self.cfg.dtype).reshape(-1,4)
            t_preds = torch.tensor(t_preds, device=self.cfg.device, dtype=self.cfg.dtype).reshape(-1,3)
        else:
            # q_pred = q_preds[0].clone().requires_grad_()
            # t_pred = t_preds[0].clone().requires_grad_()
            optimizer = torch.optim.Adam([q_preds, t_preds], lr=1e-3)
            iteration = 0
            qt_change = 1
            qt_last = torch.cat([q_preds, t_preds],dim=-1).clone().detach()


            qt_pred_for_vis_frame = []
            while iteration < 50 and abs(qt_change) > 1e-5:
                if self.cfg.record_vis:
                    qt_pred_for_vis_frame.append((q_preds[0].tolist(), t_preds[0].tolist()))
                if len(matches_3ds_within_3sigma) == 1:
                    break
                optimizer.zero_grad()
                dis_list = []

                for i in range(len(matches_3ds_within_3sigma)):
                    for j in range(len(matches_3ds_within_3sigma)):

                        if i==j:
                            continue

                        keypoint_from_depth_i =keypoint_from_depths_within_3sigma[i]

                        key3d_debug = quaternion_apply(quaternion_invert(q_preds[i]),keypoint_from_depth_i.clone().detach()) - quaternion_apply(quaternion_invert(q_preds[i]),t_preds[i]) # TODO 问题出在这个和key3d差别较大，为什么？太特么奇怪了

                        key3d = matches_3ds_within_3sigma[i][:,3:]

                        key3d1 = quaternion_apply(q_preds[j],key3d) + t_preds[j]  # debug 只换这里

                        proj_2d = (torch.matmul(test_camera_Ks[i], key3d1.transpose(1, 0)) / key3d1.transpose(1, 0)[2,:]).transpose(1, 0)[:, :2]

                        keypoint_from_depth_j = torch.zeros_like(matches_3ds_within_3sigma[i][:, 3:6])

                        tmp = depths[j][torch.clip(proj_2d[:,1].to(dtype=torch.int32),0,359), torch.clip(proj_2d[:,0].to(dtype=torch.int32),0,639)]
                        depth_mask = tmp>0
                        keypoint_from_depth_j[:, 2] = tmp.clone()
                        keypoint_from_depth_j[:, 0] = (proj_2d[:,0] - test_camera_Ks[i][0, 2]) * tmp.clone() /test_camera_Ks[i][0, 0]
                        keypoint_from_depth_j[:, 1] = (proj_2d[:,1] - test_camera_Ks[i][ 1, 2]) * tmp.clone() /  test_camera_Ks[i][1, 1]

                        key3d2 = quaternion_apply(quaternion_invert(q_preds[j]),keypoint_from_depth_j) - quaternion_apply(quaternion_invert(q_preds[j]),t_preds[j])

                        key3d3 = quaternion_apply(q_preds[i],key3d2) + t_preds[i]

                        dis = torch.norm((key3d3 - keypoint_from_depth_i)[[depth_mask]],dim=-1)
                        print(len(dis))

                        # mean = torch.mean(dis)
                        # std = torch.std(dis)
                        # sigma_mask = (dis >= mean - std) & (dis <= mean + std)
                        # dis_one_loss=torch.mean(dis[sigma_mask])
                        # dis_one_loss.backward()

                        dis_list.append(dis)


                dis_list = torch.cat(dis_list,dim=0)
                mean = torch.mean(dis_list)
                std = torch.std(dis_list)
                sigma_mask = (dis_list >= mean - std) & (dis_list <= mean + std)
                q_loss = 1e5 * (1 - torch.norm(q_preds,dim=-1).mean()) ** 2
                dis_loss = torch.mean(dis_list[sigma_mask])
                loss = q_loss + dis_loss
                # print(iteration, 'q_loss', q_loss.item(), 'c_loss', c_loss.item(), loss.item())
                loss.backward()
                optimizer.step()
                # scheduler.step()
                qt_now = torch.cat([q_preds, t_preds],dim=-1).clone().detach()
                qt_change = torch.mean(qt_now - qt_last)
                qt_last = qt_now
                iteration += 1
            print("iteration of adjust", iteration)
            if self.cfg.record_vis:
                qt_pred_for_vis_seq_adjust.append(qt_pred_for_vis_frame)

        # q_preds[0] = q_pred.detach()
        # t_preds[0] = t_pred.detach()
        return q_preds, t_preds


def run_model(d, matcher,cfg ,vis_dict=None):
    metrics = {}
    rgbs = torch.Tensor(d['rgb'])[0].float().permute(0, 3, 1, 2).to(cfg.device)  # B, C, H, W
    depths = torch.Tensor(d['depth'])[0].float().permute(0, 3, 1, 2).to(cfg.device)
    masks = torch.Tensor(d['mask'])[0].float().permute(0, 3, 1, 2).to(cfg.device)
    c2ws = d['c2w'][0]  # B, 4, 4
    o2ws = d['obj_pose'][0]  # B, 4, 4
    intrinsics = d['intrinsics']

    test_camera_K = np.zeros((3, 3))
    test_camera_K[0, 0] = intrinsics['fx']
    test_camera_K[1, 1] = intrinsics['fy']
    test_camera_K[0, 2] = intrinsics['cx']
    test_camera_K[1, 2] = intrinsics['cy']
    test_camera_K[2, 2] = 1
    # print(test_camera_K)
    # if cfg.use_depth:
    #     point_from_depth = depth_map_to_pointcloud(depths[0][0], masks[0][0], intrinsics)
    #
    # else:
    #     point_from_depth =None

    raw_r_errors = []
    raw_t_errors = []
    S = rgbs.shape[0]

    matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths = [], [], [], [], [],[]  # 需要传出去的

    for i in range(S):
        start_time = time.time()
        frame = {
            'rgb': d['rgb'][0, i:i + 1],
            'depth': d['depth'][0, i:i + 1],
            'mask': d['mask'][0, i:i + 1],
            'feat': d['feat'][0, i:i + 1],
            'intrinsics': d['intrinsics']
        }
        matches_3d_list = matcher.match_batch(frame, step=i, N_select=cfg.view_number, refine_mode=cfg.refine_mode)  # N, 6i
        # print(matches_3d[::10])
        # if matches_3d is None:
        #     print("No matches")
        #     rt_matrix = np.eye(4)
        #     continue

        rubbish_flag = False

        if cfg.refine_mode == 'a':
            matches_3d_multi_view = []
            rt_matrix_multi_view = []
            for matches_3d in matches_3d_list:
                matches = matches_3d  # 这样写是对的，不用clone
                matches[:, [1, 2]] = matches[:, [2, 1]]
                pnp_retval, translation, rt_matrix, inlier = solve_pnp_ransac(matches[:, 3:6].cpu().numpy(),
                                                                              matches[:, 1:3].cpu().numpy(),
                                                                              camera_K=test_camera_K)
                if inlier is not None:
                    matches_3d_multi_view.append(matches_3d[inlier.reshape(-1)])
                    rt_matrix_multi_view.append(rt_matrix)

            # 10个views都用（可能有重复），选inlier最多的rt_matrix
            n = 0
            select_id = 0
            for idx in range(len(matches_3d_multi_view)):
                if matches_3d_multi_view[idx].shape[0] > n:
                    n = matches_3d_multi_view[idx].shape[0]
                    select_id = idx

            if len(matches_3d_multi_view) == 1:
                matches_3ds.append(matches_3d_multi_view)
                rt_matrixs.append(rt_matrix_multi_view[select_id])
            elif len(matches_3d_multi_view) == 0:
                matches_3ds.append(torch.cat(matches))
                rt_matrixs.append(np.eye(4))
            else:
                matches_3ds.append(torch.cat(matches_3d_multi_view, dim=0))
                rt_matrixs.append(rt_matrix_multi_view[select_id])
            if n<30:    # 这个没啥用
                # rubbish_flag = True
                pass

        elif cfg.refine_mode == 'b':
            matches_3d = torch.cat(matches_3d_list, dim=0)
            matches = matches_3d
            matches[:, [1, 2]] = matches[:, [2, 1]]
            pnp_retval, translation, rt_matrix, inlier = solve_pnp_ransac(matches[:, 3:6].cpu().numpy(),
                                                                          matches[:, 1:3].cpu().numpy(),
                                                                          camera_K=test_camera_K)
            matches_3ds.append(matches_3d[inlier.reshape(-1)])
            rt_matrixs.append(rt_matrix)

        elif cfg.refine_mode == 'c':
            matches_3d = matches_3d_list[0]
            matches = matches_3d
            matches[:, [1, 2]] = matches[:, [2, 1]]
            pnp_retval, translation, rt_matrix, inlier = solve_pnp_ransac(matches[:, 3:6].cpu().numpy(),
                                                                          matches[:, 1:3].cpu().numpy(),
                                                                          camera_K=test_camera_K)
            if inlier is not None and len(inlier) >=10:
                matches_3ds.append(matches_3d[inlier.reshape(-1)])
                rt_matrixs.append(rt_matrix)
            else:
                matches_3ds.append(matches_3d)
                rt_matrixs.append(np.eye(4))
                rubbish_flag = True

        test_camera_Ks.append(test_camera_K)
        # depthss.append(depths)
        gt_cam_to_obj = np.dot(np.linalg.inv(o2ws[i]), c2ws[i])
        gt_obj_to_cam = np.linalg.inv(gt_cam_to_obj)
        gt_pose = gt_cam_to_obj
        gt_poses.append(gt_pose)

        if cfg.use_depth:
            keypoint_from_depth = torch.zeros_like(matches_3ds[i][:,3:6])
            keypoint_from_depth[:, 2] = depths[i][0][
                matches_3ds[i][:, 2].to(dtype=torch.int32), matches_3ds[i][:, 1].to(dtype=torch.int32)]
            keypoint_from_depth[:, 0] = (matches_3ds[i][:, 1] - test_camera_Ks[i][0, 2]) * keypoint_from_depth[:, 2] / \
                                        test_camera_Ks[i][0, 0]
            keypoint_from_depth[:, 1] = (matches_3ds[i][:, 2] - test_camera_Ks[i][1, 2]) * keypoint_from_depth[:, 2] / \
                                        test_camera_Ks[i][1, 1]
            keypoint_from_depths.append(keypoint_from_depth.cpu().numpy())
            if cfg.use_full_depth:
                fullpoint_from_depth = depth_map_to_pointcloud(depths[0,i], masks[0,i], intrinsics)
                fullpoint_from_depths.append(fullpoint_from_depth)
            else:
                fullpoint_from_depths.append(None)

        else:
            keypoint_from_depths.append(None)
            fullpoint_from_depths.append(None)

        # print("pnp_retval:", pnp_retval)
        # if not pnp_retval:
        #     print("No PnP result")
        #     rt_matrix = np.eye(4)


    gt_poses = np.array(gt_poses)
    if vis_dict != None:
        vis_dict['rgbs'] = np.array(rgbs.cpu()).tolist()
        vis_dict['gt_poses'] = np.stack(gt_poses).tolist()
        vis_dict['matches_3ds'] = [tmp.tolist() for tmp in matches_3ds]
        if cfg.use_depth:
            vis_dict['keypoint_from_depths'] = [tmp.tolist() for tmp in keypoint_from_depths]
            if cfg.use_full_depth:
                vis_dict['fullpoint_from_depths'] = [tmp.tolist() for tmp in fullpoint_from_depths]

    depths[0][0][masks[0][0] == 0] = -1


    return matches_3ds, np.stack(rt_matrixs, axis=0), test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths,[depths[0][0]],rubbish_flag




def fps_optimize_views_from_test(poses, select_numbers=8,start_idx=0):
    # dist_mat = np.zeros((poses.shape[0],poses.shape[0]))
    time1 = time.time()
    # for i,pose1 in enumerate(poses):
    #     for j,pose2 in enumerate(poses):
    #         dist_mat[i,j],_ = compute_RT_errors(pose1,pose2)
    dist_mat = compute_R_errors_batch(poses,poses)

    time2 = time.time()
    select_views = np.zeros((select_numbers,), dtype=int)
    view_idx = start_idx
    dist_to_set = dist_mat[:,view_idx]
    for i in range(select_numbers):
        select_views[i] = view_idx
        dist_to_set = np.minimum(dist_to_set,dist_mat[:,view_idx])
        view_idx = np.argmax(dist_to_set)
    time3 = time.time()
    return select_views

def nearest_view_select(poses, select_numbers=8, start_idx=0):
    dist_mat = compute_R_errors_batch(poses, poses)

    select_views = np.zeros((select_numbers,), dtype=int)
    view_idx = start_idx
    for i in range(select_numbers):
        select_views[i] = view_idx
        # 将已选择的视角从距离矩阵中移除，避免重复选择
        dist_mat[:, view_idx] = np.inf
        # 更新到当前视角的距离数组，选择与当前视角相差最小的视角
        view_idx = np.argmin(dist_mat[view_idx])

    return select_views

def compute_results(q_preds,t_preds,gt_poses):
    r_preds = quaternion_to_matrix(q_preds) # n.3.3

    rt_preds = torch.zeros_like(gt_poses)   # n,4,4
    rt_preds[:,:3,:3] = r_preds
    rt_preds[:,:3,3] = t_preds
    rt_preds[:,3,3] = 1


    # pose_preds = torch.inverse(rt_preds)    # 注意这里搞错了，别对这个取逆，而要对pose取逆
    pose_preds = rt_preds
    gt_poses = torch.inverse(gt_poses)

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
        theta = np.arccos(np.clip((np.trace(R) - 1)/2,-1,1)) * 180/np.pi
        shift = np.linalg.norm(T1-T2) * 100
        # theta = min(theta, np.abs(360 - theta))
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
    return r_errors,t_errors

def compute_results_rt(pred_poses,gt_poses):
    r_errors = []
    t_errors = []
    #pred_poses = pred_poses.detach().cpu().numpy()
    #gt_poses = gt_poses.detach().cpu().numpy()

    for i in range(len(gt_poses)):
        R1 = gt_poses[i][:3, :3]/np.cbrt(np.linalg.det(gt_poses[i][:3, :3]))
        T1 = gt_poses[i][:3, 3]

        R2 = pred_poses[i][:3, :3]/np.cbrt(np.linalg.det(pred_poses[i][:3, :3]))
        T2 = pred_poses[i][:3, 3]

        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1)/2,-1,1)) * 180/np.pi
        shift = np.linalg.norm(T1-T2) * 100
        # theta = min(theta, np.abs(180 - theta))
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
    return r_errors,t_errors

def chamfer_distance(depth_point,pred_point):
    # input: torch(n,3) torch(m,3)
    dist_map = pairwise_distances_torch(depth_point,pred_point)    # n,m
    dist1,_ = torch.min(dist_map,dim=-1)                    # n
    dist2,_ = torch.min(dist_map,dim=0)                     # m
    dist = torch.cat((dist1,dist2),dim=0)
    return dist1

def main(cfg):
    # The idea of this file is to test DinoV2 matcher and multi frame optimization on Blender rendered data

    memory_pool = MemoryPool(cfg)

    # test_dataset = SimTrackDataset(dataset_location=test_dir, seqlen=S, features=feat_layer)
    test_dataset = TrackingDataset(dataset_location=cfg.test_dir, seqlen=cfg.S,features=cfg.feat_layer)
    ref_dataset = ReferenceDataset(dataset_location=cfg.ref_dir, num_views=840, features=cfg.feat_layer)
    sim_dataset = SimVideoDataset(features=19)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.B, shuffle=cfg.shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=cfg.shuffle)
    sim_dataloader = DataLoader(sim_dataset, batch_size=1, shuffle=False)
    iterloader = iter(test_dataloader)
    # Load ref images and init Dinov2 Matcher
    refs = next(iter(ref_dataloader))
    sim_loader = iter(sim_dataloader)
    # ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3)  # B, S, C, H, W
    # ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3)
    # ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3)
    # ref_images = torch.concat([ref_rgbs[0], ref_depths[0, :, 0:1], ref_masks[0, :, 0:1]], axis=1)
    # print(ref_images.shape)
    global_step = 0
    gripper_path = "/root/autodl-tmp/shiqian/code/gripper/franka_hand_obj/franka_hand.obj"
    pointcloud_path = "./pointclouds/gripper.txt"
    if not os.path.exists(pointcloud_path):
        gripper_pointcloud = sample_points_from_mesh(gripper_path, fps=True, n_pts=8192)
    else:
        gripper_pointcloud = read_pointcloud(pointcloud_path)

    matcher = Dinov2Matcher(ref_dir=cfg.ref_dir, refs=refs,
                            model_pointcloud=gripper_pointcloud,
                            feat_layer=cfg.feat_layer,
                            device=cfg.device)


    print(len(test_dataset))
    while global_step < cfg.max_iter:
        matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths,depths = [], [], [], [], [], [], []
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0

        read_start_time = time.time()

        sample = next(iterloader)
        # sim = next(sim_loader)
        read_time = time.time() - read_start_time
        iter_read_time += read_time

        if sample is not None:
            if cfg.record_vis: # & (global_step>cfg.key_number):
                vis_dict = {}
            else:
                vis_dict = None
            matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths,depths,rubbish_flag = run_model(sample,matcher,cfg,vis_dict)
        else:
            print('sampling failed')
        qt_pred_for_vis_seq = []
        print("Iteration {}".format(global_step))
        frame = {'matches_3d': matches_3ds[0],  # tensor
                 'gt_pose': gt_poses[0],        # np array
                 'rt_matrix': rt_matrixs[0],    # np array
                 'test_camera_K': test_camera_Ks[0],    # np array
                 'path':sample['path'],
                 'keypoint_from_depth':keypoint_from_depths[0],     # np array
                 'fullpoint_from_depth':fullpoint_from_depths[0],   # np array
                 'depth':depths[0],
                 'rubbish_flag':rubbish_flag}
        # frame: matches_3d,gt_pose,rt_matrix,camera_Ks)
        memory_time = time.time()
        memory_pool.refine_new_frame(frame,record_vis=cfg.record_vis,global_step=global_step,vis_dict=vis_dict)
        iter_end_time = time.time()
        print(f'memory_pool time cost {iter_end_time-memory_time}')     # 大约0.04s
        print("Iteration {}".format(global_step),' time',iter_end_time-iter_start_time) # 大约0.08s
    memory_pool.eliminate_all_frames_and_compute_result(global_step)

def main_sim(cfg):
    # The idea of this file is to test DinoV2 matcher and multi frame optimization on Blender rendered data


    ref_dir = "/root/autodl-tmp/shiqian/datasets/reference_views/%s" % cfg.gripper
    # test_dataset = SimTrackDataset(dataset_location=test_dir, seqlen=S, features=feat_layer)
    # test_dataset = TrackingDataset(dataset_location=cfg.test_dir, seqlen=cfg.S,features=cfg.feat_layer)
    ref_dataset = ReferenceDataset(dataset_location=ref_dir, num_views=840, features=cfg.feat_layer)
    sim_dataset = SimVideoDataset(gripper=cfg.gripper, features=19)
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.B, shuffle=cfg.shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=cfg.shuffle)
    sim_dataloader = DataLoader(sim_dataset, batch_size=1, shuffle=False)
    # iterloader = iter(test_dataloader)
    # Load ref images and init Dinov2 Matcher
    refs = next(iter(ref_dataloader))
    sim_loader = iter(sim_dataloader)
    # ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3)  # B, S, C, H, W
    # ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3)
    # ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3)
    # ref_images = torch.concat([ref_rgbs[0], ref_depths[0, :, 0:1], ref_masks[0, :, 0:1]], axis=1)
    # print(ref_images.shape)
    global_step = 0
    gripper_path = "/root/autodl-tmp/shiqian/datasets/final_20240419/%s/model/model.obj" % cfg.gripper
    pointcloud_path = "/root/autodl-tmp/shiqian/datasets/final_20240419/%s/model/sampled_4096.txt" % cfg.gripper
    if not os.path.exists(pointcloud_path):
        gripper_pointcloud = sample_points_from_mesh(gripper_path, fps=True, n_pts=4096)
    else:
        gripper_pointcloud = read_pointcloud(pointcloud_path)

    matcher = Dinov2Matcher(ref_dir=ref_dir, refs=refs,
                            model_pointcloud=gripper_pointcloud,
                            feat_layer=cfg.feat_layer,
                            device=cfg.device)


    print(len(sim_dataset))
    memory_pool = MemoryPool(cfg)
    all_preds = []
    all_gts = []
    while global_step < cfg.max_iter:
        global_step += 1
        sim = next(sim_loader)
        single_frame_rts = []
        for i in range(len(sim['rgb'][0])):
            print("Frame %d" % i)
            sim_frame = {
                'rgb': sim['rgb'][:,i:i+1,...],
                'depth':sim['depth'][:,i:i+1,...],
                'mask': sim['mask'][:, i:i + 1, ...],
                'c2w': sim['c2w'][:, i:i + 1, ...],
                'obj_pose': sim['obj_pose'][:, i:i + 1, ...],
                'feat': sim['feat'][:, i:i + 1, ...],
                'intrinsics': sim['intrinsics'],
            }
            matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths,depths = [], [], [], [], [], [], []

            iter_start_time = time.time()
            iter_read_time = 0.0

            read_start_time = time.time()

            # sample = next(iterloader)

            read_time = time.time() - read_start_time
            iter_read_time += read_time

            if sim is not None:
                if cfg.record_vis: # & (global_step>cfg.key_number):
                    vis_dict = {}
                else:
                    vis_dict = None
                matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths,depths,rubbish_flag = run_model(sim_frame,matcher,cfg,vis_dict)
            else:
                print('sampling failed')
            if cfg.pnp_only:
                #print(rt_matrixs[0].shape, gt_poses[0].shape)
                all_preds.append(rt_matrixs[0])
                all_gts.append(np.linalg.inv(gt_poses[0]))
                continue
            qt_pred_for_vis_seq = []
            print(f"seq{global_step}frame{i}")
            frame = {'matches_3d': matches_3ds[0],  # tensor
                     'gt_pose': gt_poses[0],        # np array
                     'rt_matrix': rt_matrixs[0],    # np array
                     'test_camera_K': test_camera_Ks[0],    # np array
                     'path':None,
                     'keypoint_from_depth':keypoint_from_depths[0],     # np array
                     'fullpoint_from_depth':fullpoint_from_depths[0],   # np array
                     'depth':depths[0],
                     "rubbish_flag":rubbish_flag}
            # frame: matches_3d,gt_pose,rt_matrix,camera_Ks
            memory_time = time.time()
            memory_pool.refine_new_frame(frame,record_vis=cfg.record_vis,global_step=global_step,vis_dict=vis_dict)
            iter_end_time = time.time()
            print(f'memory_pool time cost {iter_end_time-memory_time}')     # 大约0.04s
            print("Iteration {}".format(global_step),' time',iter_end_time-iter_start_time) # 大约0.08s
        if not cfg.pnp_only:
            memory_pool.eliminate_all_frames_and_compute_result(global_step)
        else:
            # compute_results_rt(all_preds,all_gts)
            result_name = f'{cfg.gripper}_{global_step}'
            memory_pool.eliminate_all_frames_and_compute_result(result_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1, help='batch size')
    parser.add_argument('--S', type=int, default=1, help='sequence length')
    parser.add_argument('--shuffle',default=False, action='store_true')
    parser.add_argument('--ref_dir', type=str, default='/root/autodl-tmp/shiqian/code/render/reference_views')
    parser.add_argument('--test_dir', type=str, default='/root/autodl-tmp/shiqian/datasets/Ty_data')
    parser.add_argument('--feat_layer', type=str, default=19)
    parser.add_argument('--max_iter',type=int, default=30)
    parser.add_argument('--device',type=str,default='cuda:3')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--refine_mode',type=str,default='a')
    parser.add_argument('--max_number',type=int, default=32)
    parser.add_argument('--key_number',type=int,default=8)
    parser.add_argument('--record_vis',default=False ,action='store_true')
    parser.add_argument('--view_number',type=int,default=10)
    parser.add_argument('--use_depth',default=True, action='store_true')
    parser.add_argument('--use_full_depth',default=False, action='store_true')  # 这个c++没写,一直关掉
    parser.add_argument('--noise',default='none', action='store_true')   # none random incremental
    parser.add_argument('--adjust',default=False, action='store_true')      # 低精度模式下需要打开，但是有问题
    parser.add_argument('--rela_mode',default='gt',action='store_true',help='gt or pr')     # 高精度模式：gt #低精度模式下：pr，但是有问题
    parser.add_argument('--use_cpp',default=True,action='store_true')
    parser.add_argument('--single_opt',default=False,action='store_true')   # 单帧优化
    parser.add_argument('--init',default='rela')        # 不要改
    
    parser.add_argument('--gripper', type=str, default="panda")
    parser.add_argument('--pnp_only',default=False,action='store_true')

    cfg = parser.parse_args()
    # main(cfg)
    main_sim(cfg)