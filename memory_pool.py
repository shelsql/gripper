import torch
from torch.optim import lr_scheduler

from utils.quaternion_utils import *
import numpy as np
from utils.spd import sample_points_from_mesh,compute_RT_errors
from datasets.ref_dataset import ReferenceDataset, SimTestDataset, SimTrackDataset
from datasets.ty_datasets import TrackingDataset
from torch.utils.data import DataLoader
import os
from matcher import Dinov2Matcher
import time
from utils.spd import read_pointcloud,depth_map_to_pointcloud,pairwise_distances_torch
import pickle
import random
from utils.geometric_vision import solve_pnp_ransac
import argparse
import scipy

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

class MemoryPool():
    def __init__(self,cfg):
        self.matches_3ds = []
        self.rt_matrixs = []
        self.test_camera_Ks = []
        self.gt_poses = []
        self.keypoint_from_depths = []
        self.fullpoint_from_depths = []
        self.paths = []
        self.initial_pose = None
        self.relative_poses = []

        self.q_for_compute_result = []
        self.t_for_compute_result = []
        self.gt_for_compute_result = []

        self.max_number = cfg.max_number
        self.key_number = cfg.key_number
        self.gripper_point = torch.tensor(read_pointcloud(cfg.gripper_point_path),device=cfg.device)
        self.cfg = cfg
    def refine_new_frame(self,frame,record_vis,global_step,vis_dict=None):
        # frame: matches_3d,gt_pose,rt_matrix,camera_Ks
        self.matches_3ds.insert(0,frame['matches_3d'])      # tensor(98,6) different shape
        self.rt_matrixs.insert(0,frame['rt_matrix'])        # np array
        self.test_camera_Ks.insert(0,frame['test_camera_K'])# np array
        self.gt_poses.insert(0,frame['gt_pose'])            # np array
        self.paths.insert(0,frame['path'])

        if self.initial_pose is None:
            self.initial_pose = frame['gt_pose']                    # np array

        if self.cfg.add_noise:
            noise_r = np.eye(3)
            random_rotation = scipy.spatial.transform.Rotation.from_rotvec(np.random.randn(3) * 0.1)
            noise_r = random_rotation.apply(noise_r)
            noise_t = np.random.randn(3)*0.01
            random_rt = np.eye(4)
            random_rt[:3,:3] = noise_r
            random_rt[:3,3] = noise_t
            self.relative_poses.insert(0, random_rt @ frame['gt_pose'] @ np.linalg.inv(self.initial_pose))
        else:
            self.relative_poses.insert(0,frame['gt_pose']@np.linalg.inv(self.initial_pose))

        # self.point_from_depth.insert(0,torch.tensor(frame['point_from_depth'],device=self.cfg.device))
        if self.cfg.use_depth:
            self.keypoint_from_depths.insert(0,frame['keypoint_from_depth'])    # np array
            if self.cfg.use_full_depth:
                self.fullpoint_from_depths.insert(0,frame['fullpoint_from_depth'])  # np array



        # 从内存池中fps挑选k帧出来辅助优化
        key_ids = fps_optimize_views_from_test(np.array(self.relative_poses),
                                               select_numbers=min(len(self.relative_poses), self.key_number),
                                               start_idx=0)
        if len(self.matches_3ds)<self.key_number:   # 当内存池不足key_number帧时，把其他关键帧也优化一下
            for frame_id in key_ids:
                views_idx_for_opt = fps_optimize_views_from_test(np.array(self.relative_poses),
                                                                 select_numbers=len(key_ids),
                                                                 start_idx=frame_id)
                q_pred, t_pred = self.optimize(views_idx_for_opt,self.cfg)
                r_pred = quaternion_to_matrix(q_pred)
                rt_pred = np.zeros((4,4))
                rt_pred[:3,:3] = r_pred.cpu().detach().numpy()
                rt_pred[:3, 3] = t_pred.cpu().detach().numpy()
                rt_pred[3, 3] = 1
                self.rt_matrixs[frame_id] = rt_pred

                if self.cfg.adjust:
                    q_preds, t_preds = self.bundle_adjust()
                    r_preds = quaternion_to_matrix(q_preds)
                    rt_preds = np.zeros_like(self.rt_matrixs)
                    rt_preds[:,:3,:3] = r_preds.cpu().detach().numpy()
                    rt_preds[:,:3, 3] = t_preds.cpu().detach().numpy()
                    rt_preds[:,3, 3] = 1
                    self.rt_matrixs = [tmp for tmp in rt_preds]


        else:
            qt_pred_for_vis_seq = [] if record_vis else None

            views_idx_for_opt = fps_optimize_views_from_test(np.array(self.relative_poses),
                                                             select_numbers=len(key_ids),
                                                             start_idx=0)
            q_pred, t_pred = self.optimize(views_idx_for_opt,
                                                self.cfg,
                                                qt_pred_for_vis_seq)
            r_pred = quaternion_to_matrix(q_pred)
            rt_pred = np.zeros((4, 4))
            rt_pred[:3, :3] = r_pred.cpu().detach().numpy()
            rt_pred[:3, 3] = t_pred.cpu().detach().numpy()
            rt_pred[3, 3] = 1
            self.rt_matrixs[0] = rt_pred

            if self.cfg.adjust:
                q_preds, t_preds = self.bundle_adjust()
                r_preds = quaternion_to_matrix(q_preds)
                rt_preds = np.zeros_like(self.rt_matrixs)
                rt_preds[:, :3, :3] = r_preds.cpu().detach().numpy()
                rt_preds[:, :3, 3] = t_preds.cpu().detach().numpy()
                rt_preds[:, 3, 3] = 1
                self.rt_matrixs = [tmp for tmp in rt_preds]

            if vis_dict != None:
                vis_dict['qt_preds'] = qt_pred_for_vis_seq
                if not os.path.exists(f'vis_results/memory_pool_tydata'):
                    os.makedirs(f'vis_results/memory_pool_tydata')
                with open(f'vis_results/memory_pool_tydata/step{global_step}.pkl', 'wb') as f:
                    pickle.dump(vis_dict, f)

        self.q_for_compute_result.append(matrix_to_quaternion(torch.tensor(self.rt_matrixs[0][:3,:3])).cpu().detach().numpy())
        self.t_for_compute_result.append(self.rt_matrixs[0][:3,3])
        self.gt_for_compute_result.append(self.gt_poses[0])
        # 如果内存池已满，则删除其中一帧
        # 以gt_pose中的旋转为依据
        if len(self.matches_3ds) > self.max_number:
            self.eliminate_one_frame()

    def eliminate_one_frame(self):
        dist_mat = np.zeros((len(self.relative_poses), len(self.relative_poses)))
        for i, pose1 in enumerate(self.relative_poses):  # TODO batch形式
            for j, pose2 in enumerate(self.relative_poses):
                dist_mat[i, j], _ = compute_RT_errors(pose1, pose2)
        distsum = np.sum(dist_mat,axis=1,keepdims=False)
        eliminate_id = np.argsort(distsum)[0]
        self.matches_3ds.pop(eliminate_id)
        self.rt_matrixs.pop(eliminate_id)
        # self.q_for_compute_result.append(matrix_to_quaternion(torch.tensor(final_rt[:3,:3])).cpu().detach().numpy())
        # self.t_for_compute_result.append(torch.tensor(final_rt[:3,3]).cpu().detach().numpy())
        self.test_camera_Ks.pop(eliminate_id)
        self.paths.pop(eliminate_id)
        if self.cfg.use_depth:
            self.keypoint_from_depths.pop(eliminate_id)
            if self.cfg.use_full_depth:
                self.fullpoint_from_depths.pop(eliminate_id)
        self.gt_poses.pop(eliminate_id)
        # self.gt_for_compute_result.append(torch.tensor(cor_gt,device=cfg.device,dtype=torch.float32).cpu().detach().numpy())
        self.relative_poses.pop(eliminate_id)

    def eliminate_all_frames_and_compute_result(self):
        while len(self.matches_3ds) > 0:
            self.eliminate_one_frame()
        compute_results(torch.tensor(np.stack(self.q_for_compute_result)),
                        torch.tensor(np.stack(self.t_for_compute_result)),
                        torch.tensor(np.stack(self.gt_for_compute_result)))

        results = np.concatenate([np.stack(self.q_for_compute_result),
                             np.stack(self.t_for_compute_result),
                             np.stack(self.gt_for_compute_result).reshape(-1, 16)],axis=1)  # 1024,4+3+16
        if not os.path.exists(f'results'):
            os.makedirs(f'results')
        depth_str = 'use_depth' if cfg.use_depth else 'no_depth'
        noise_str = 'add_noise' if cfg.add_noise else 'no_noise'
        adjust_str = 'adjust' if cfg.adjust else 'no_adjust'
        full_str = 'full' if cfg.use_full_depth else 'no_full'
        max_iter_str = cfg.max_iter

        np.savetxt(f'results/memory{self.max_number}_key{self.key_number}_iter{max_iter_str}_{depth_str}_{full_str}_{noise_str}_{adjust_str}.txt', results)

    def optimize(self,views_idx_for_opt,cfg,qt_pred_for_vis_seq=None):
        '''删除垃圾点的迭代方法，优化的是输入list的第一帧，后面的帧仅辅助'''
        # matches_3ds: list[tensor(342,6),...] different shape
        # other: list[np.array(4,4)or(3,3)], same shape

        matches_3ds = [self.matches_3ds[i] for i in views_idx_for_opt]   # list[tensor(342,6),...] different shape
        if cfg.use_depth:
            keypoint_from_depths = [torch.tensor(self.keypoint_from_depths[i],dtype=cfg.dtype,device=cfg.device) for i in views_idx_for_opt]        # list[tensor(98,3),...] different shape
            if self.cfg.use_full_depth:
                fullpoint_from_depths = [torch.tensor(self.fullpoint_from_depths[i],dtype=cfg.dtype,device=cfg.device) for i in views_idx_for_opt]      # list[tensor(2728,3)]

        rt_matrixs = torch.stack([torch.tensor(self.rt_matrixs[i],dtype=cfg.dtype,device=cfg.device) for i in views_idx_for_opt])            # b,4,4
        q_preds = matrix_to_quaternion(rt_matrixs[:,:3,:3])
        t_preds = rt_matrixs[:,:3,3]
        test_camera_Ks = torch.stack([torch.tensor(self.test_camera_Ks[i],dtype=cfg.dtype,device=cfg.device) for i in views_idx_for_opt])   # b,3,3
        relative_poses_tmp = torch.stack([torch.tensor(self.relative_poses[i],dtype=cfg.dtype,device=cfg.device) for i in views_idx_for_opt])                # b,4,4
        q_relatives = matrix_to_quaternion(relative_poses_tmp[:, :3, :3])  # b,4
        t_relatives = relative_poses_tmp[:, :3, 3]                  # b,3
        q_rela = q_relatives[0]
        t_rela = t_relatives[0]

        # 用所有初值相对于当前优化帧的姿势取平均(当达到关键帧数量后，不再使用当前帧通过pnp预测的pose)
        # q_relative_list, t_relative_list = [], []
        q_pred_list, t_pred_list = [], []
        for i in range(1, len(matches_3ds)) if (len(matches_3ds) > cfg.key_number and cfg.key_number!=1) else range(len(matches_3ds)):
            q_pred_tmp = quaternion_multiply(q_preds[i],quaternion_multiply(q_relatives[i],quaternion_invert(q_rela)))
            t_pred_tmp = quaternion_apply(q_pred_tmp,-t_rela) + quaternion_apply(q_preds[i],t_relatives[i]) + t_preds[i]
            q_pred_list.append(q_pred_tmp)
            t_pred_list.append(t_pred_tmp)
            # rt_relative = rt_matrixs[i] @ (relative_poses[i]) @ torch.inverse(relative_poses[0])
            # q_relative = matrix_to_quaternion(rt_relative[:3, :3])
            # q_relative_list.append(q_relative)
            # t_relative = rt_relative[:3, 3]
            # t_relative_list.append(t_relative)
        q_avg = torch.mean(torch.stack(q_pred_list), dim=0)
        t_avg = torch.mean(torch.stack(t_pred_list), dim=0)
        q_pred = torch.tensor(q_avg / torch.norm(q_avg), device=matches_3ds[0].device,requires_grad=True)
        t_pred = torch.tensor(t_avg, device=matches_3ds[0].device, requires_grad=True)
        # q_pred = torch.tensor(matrix_to_quaternion(rt_matrixs[:3, :3]), requires_grad=True)  # 4
        # t_pred = torch.tensor(rt_matrixs[:3, 3], requires_grad=True)  # 3
        optimizer = torch.optim.Adam([ q_pred, t_pred], lr=2e-3)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10],gamma=0.1)
        start_time = time.time()
        iteration = 0
        loss_change = 1
        loss_last = 0
        qt_pred_for_vis_frame = []

        # step1: refine with keypoints
        while iteration < 100 and abs(loss_change) > 1e-2:
            if cfg.record_vis:
                qt_pred_for_vis_frame.append((q_pred.tolist(), t_pred.tolist()))
            optimizer.zero_grad()
            reproj_dis_list = []
            key3d_dis_list = []
            for i in range(len(matches_3ds)):
                fact_2d = matches_3ds[i][:, 1:3].clone()  # 342,2
                pred_3dkey = quaternion_apply(quaternion_invert(q_relatives[i]),matches_3ds[i][:, 3:]) - quaternion_apply(quaternion_invert(q_relatives[i]),t_relatives[i])
                pred_3dkey = quaternion_apply(q_rela,pred_3dkey) + t_rela
                pred_3dkey = quaternion_apply(q_pred,pred_3dkey) + t_pred
                # use rotation matrix to change:
                # pred_3dkey = quaternion_apply(matrix_to_quaternion(torch.inverse(relative_poses[i])[:3, :3]),
                #                               matches_3ds[i][:, 3:]) + torch.inverse(relative_poses[i])[:3, 3]
                # pred_3dkey = quaternion_apply(matrix_to_quaternion(relative_poses[0][:3, :3]), pred_3dkey) + relative_poses[0][:3,
                #                                                                                        3]
                # pred_3dkey = quaternion_apply(q_pred, pred_3dkey) + t_pred


                if cfg.use_depth:
                    key3d_dis = torch.norm(keypoint_from_depths[i] - pred_3dkey, dim=-1)
                    key3d_dis_list.append(key3d_dis)

                proj_2d = (torch.matmul(test_camera_Ks[i], pred_3dkey.transpose(1, 0)) / pred_3dkey.transpose(1, 0)[2,
                                                                                         :]).transpose(1, 0)[:,
                          :2]  # 220,2
                reproj_dis = torch.norm(fact_2d - proj_2d, dim=1)  # 220,
                reproj_dis_list.append(reproj_dis)


            reproj_dis_list = torch.cat(reproj_dis_list, dim=0)
            mean = reproj_dis_list.mean()
            std = reproj_dis_list.std()
            within_3sigma = (reproj_dis_list >= mean - 1 * std) & (reproj_dis_list <= mean + 1 * std)
            reproj_dis_list = reproj_dis_list[within_3sigma]  # 删除垃圾点

            if cfg.use_depth:
                key3d_dis_list = torch.cat(key3d_dis_list, dim=0)
                mean = key3d_dis_list.mean()
                std = key3d_dis_list.std()
                within_3sigma = (key3d_dis_list >= mean - 1 * std) & (key3d_dis_list <= mean + 1 * std)
                key3d_dis_list = key3d_dis_list[within_3sigma]



            q_loss = 1e5 * (1 - torch.norm(q_pred)) ** 2
            reproj_loss = torch.mean(reproj_dis_list)
            # c_loss = 1000*torch.mean(chamfer_dis_list)
            if cfg.use_depth:
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

        # torch.cuda.empty_cache()

        # step2: refine with full depth image
        # 先剔去离群点再优化，而不是在优化过程中筛选

        if cfg.use_depth:
            if cfg.use_full_depth:
                fullpoint_from_depths_in3sigma_list = []
                for i in range(len(matches_3ds)):
                    pred_3dfull = quaternion_apply(quaternion_invert(q_relatives[i]),self.gripper_point) - quaternion_apply(quaternion_invert(q_relatives[i]),t_relatives[i])
                    pred_3dfull = quaternion_apply(q_rela, pred_3dfull) + t_rela
                    pred_3dfull = quaternion_apply(q_pred, pred_3dfull) + t_pred
                    chamfer_dis = chamfer_distance(fullpoint_from_depths[i],pred_3dfull)
                    mean = chamfer_dis.mean()
                    std = chamfer_dis.std()
                    within_3sigma = (chamfer_dis >= mean - 1 * std) & (chamfer_dis <= mean + 1 * std)
                    fullpoint_from_depths_in3sigma_list.append(fullpoint_from_depths[i][within_3sigma])

                iteration = 0
                loss_change = 1
                loss_last = 0

                # q_pred = q_pred.detach().cpu().numpy()
                # t_pred = t_pred.detach().cpu().numpy()
                # q_pred = torch.tensor(q_pred, device=cfg.device, dtype=cfg.dtype, requires_grad=True)
                # t_pred = torch.tensor(t_pred, device=cfg.device, dtype=cfg.dtype, requires_grad=True)
                while iteration < 100 and abs(loss_change) > 5e-3:
                    if cfg.record_vis:
                        qt_pred_for_vis_frame.append((q_pred.tolist(), t_pred.tolist()))
                    optimizer.zero_grad()
                    chamfer_dis_list = []
                    for i in range(len(matches_3ds)):
                        pred_3dfull = quaternion_apply(quaternion_invert(q_relatives[i]),
                                                       self.gripper_point) - quaternion_apply(
                            quaternion_invert(q_relatives[i]), t_relatives[i])
                        pred_3dfull = quaternion_apply(q_rela, pred_3dfull) + t_rela
                        pred_3dfull = quaternion_apply(q_pred, pred_3dfull) + t_pred

                        chamfer_dis = chamfer_distance(fullpoint_from_depths_in3sigma_list[i], pred_3dfull)
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
        matches_3ds = self.matches_3ds  # list[tensor(342,6),...] different shape
        if cfg.use_depth:
            keypoint_from_depths = [torch.tensor(tmp, dtype=cfg.dtype, device=cfg.device) for tmp in self.keypoint_from_depths]  # list[tensor(98,3),...] different shape
            if self.cfg.use_full_depth:
                fullpoint_from_depths = [torch.tensor(tmp, dtype=cfg.dtype, device=cfg.device) for tmp in self.fullpoint_from_depths]  # list[tensor(2728,3)]

        rt_matrixs = torch.stack(
            [torch.tensor(tmp, dtype=cfg.dtype, device=cfg.device) for tmp in self.rt_matrixs])  # b,4,4
        q_preds = matrix_to_quaternion(rt_matrixs[:, :3, :3]).requires_grad_()
        t_preds = rt_matrixs[:, :3, 3].requires_grad_()
        test_camera_Ks = torch.stack([torch.tensor(tmp, dtype=cfg.dtype, device=cfg.device) for tmp in
                                      self.test_camera_Ks])  # b,3,3
        optimizer = torch.optim.Adam([ q_preds, t_preds], lr=2e-3)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10],gamma=0.1)
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
                if cfg.use_depth:
                    key3d_dis = torch.norm(keypoint_from_depths[i] - pred_3dkey,dim=-1)
                    key3d_dis_list.append(key3d_dis)

            reproj_dis_list = torch.cat(reproj_dis_list, dim=0)
            mean = reproj_dis_list.mean()
            std = reproj_dis_list.std()
            within_3sigma = (reproj_dis_list >= mean - 1 * std) & (reproj_dis_list <= mean + 1 * std)
            reproj_dis_list = reproj_dis_list[within_3sigma]  # 删除垃圾点

            if cfg.use_depth:
                key3d_dis_list = torch.cat(key3d_dis_list, dim=0)
                mean = key3d_dis_list.mean()
                std = key3d_dis_list.std()
                within_3sigma = (key3d_dis_list >= mean - 1 * std) & (key3d_dis_list <= mean + 1 * std)
                key3d_dis_list = key3d_dis_list[within_3sigma]

            q_loss = 1e5 * (1 - torch.norm(q_preds,dim=-1).mean()) ** 2
            reproj_loss = torch.mean(reproj_dis_list)

            if cfg.use_depth:
                key3d_loss = 1000 * torch.mean(key3d_dis_list)
                loss = q_loss + reproj_loss + key3d_loss
                print(iteration, 'q_loss', q_loss.item(), 'reproj', reproj_loss.item(), 'key3d_loss', key3d_loss.item(),loss.item())
            else:
                loss = q_loss + reproj_loss
                # print(iteration, 'q_loss', q_loss.item(), 'reproj', reproj_loss.item(),  loss.item())

            loss.backward()
            optimizer.step()
            # scheduler.step()

            loss_change = loss - loss_last
            loss_last = loss
            iteration += 1

        # if cfg.use_depth:         # 显存放不下
        #     if cfg.use_full_depth:
        #         fullpoint_from_depths_in3sigma_list = []
        #         for i in range(len(matches_3ds)):
        #             pred_3dfull = quaternion_apply(q_preds[i],self.gripper_point) + t_preds[i]
        #             chamfer_dis = chamfer_distance(fullpoint_from_depths[i],pred_3dfull)
        #             mean = chamfer_dis.mean()
        #             std = chamfer_dis.std()
        #             within_3sigma = (chamfer_dis >= mean-1*std) & (chamfer_dis <= mean+1*std)
        #             fullpoint_from_depths_in3sigma_list.append(fullpoint_from_depths[i][within_3sigma])
        #
        #         iteration = 0
        #         loss_change = 1
        #         loss_last = 0
        #         while iteration<100 and abs(loss_change)> 5e-3:
        #             optimizer.zero_grad()
        #             chamfer_dis_list = []
        #             for i in range(len(matches_3ds)):
        #                 pred_3dfull = quaternion_apply(q_preds[i],self.gripper_point) + t_preds[i]
        #                 chamfer_dis = chamfer_distance(fullpoint_from_depths_in3sigma_list[i],pred_3dfull)
        #                 chamfer_dis_list.append(chamfer_dis)
        #             chamfer_dis_list = torch.cat(chamfer_dis_list, dim=0)
        #             # print(chamfer_dis_list.shape)
        #
        #             q_loss = 1e5 * (1 - torch.norm(q_preds,dim=-1).mean()) ** 2
        #             c_loss = 1000*torch.mean(chamfer_dis_list)
        #
        #             loss = q_loss + c_loss  # 问题出在：fullpoint_from_depths中的点数越来越多
        #             # print(iteration, 'q_loss', q_loss.item(), 'c_loss', c_loss.item(), loss.item())
        #             loss.backward()
        #             optimizer.step()
        #             # scheduler.step()
        #
        #             loss_change = loss.item() - loss_last
        #             loss_last = loss.item()
        #             iteration += 1

        return q_preds,t_preds

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
            matches_3ds.append(torch.cat(matches_3d_multi_view, dim=0))
            rt_matrixs.append(rt_matrix_multi_view[select_id])

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
            matches_3ds.append(matches_3d[inlier.reshape(-1)])
            rt_matrixs.append(rt_matrix)

        test_camera_Ks.append(test_camera_K)
        # depthss.append(depths)
        gt_cam_to_obj = np.dot(np.linalg.inv(o2ws[i]), c2ws[i])
        gt_obj_to_cam = np.linalg.inv(gt_cam_to_obj)
        gt_pose = gt_cam_to_obj
        gt_poses.append(gt_pose)

        if cfg.use_depth:
            keypoint_from_depth = torch.zeros_like(matches_3ds[i][:,3:6])
            keypoint_from_depth[:, 2] = depths[0][i][
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




    return matches_3ds, np.stack(rt_matrixs, axis=0), test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths




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
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.B, shuffle=cfg.shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=cfg.shuffle)
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

    matcher = Dinov2Matcher(ref_dir=cfg.ref_dir, refs=refs,
                            model_pointcloud=gripper_pointcloud,
                            feat_layer=cfg.feat_layer,
                            device=cfg.device)

    q_preds, t_preds, gt_poses_for_result = [], [], []
    print(len(test_dataset))
    while global_step < cfg.max_iter:
        matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths = [], [], [], [], [], []
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0

        read_start_time = time.time()

        sample = next(iterloader)
        read_time = time.time() - read_start_time
        iter_read_time += read_time

        if sample is not None:
            if cfg.record_vis & (global_step>cfg.key_number):
                vis_dict = {}
            else:
                vis_dict = None
            matches_3ds, rt_matrixs, test_camera_Ks, gt_poses,keypoint_from_depths,fullpoint_from_depths = run_model(sample,matcher,cfg,vis_dict)
        else:
            print('sampling failed')
        qt_pred_for_vis_seq = []
        print("Iteration {}".format(global_step))
        frame = {'matches_3d': matches_3ds[0],  # tensor
                 'gt_pose': gt_poses[0],        # np array
                 'rt_matrix': rt_matrixs[0],    # np array
                 'test_camera_K': test_camera_Ks[0],    # np array
                 'path':sample['path'],
                 # 'depth':depths[0][0][0],
                 'keypoint_from_depth':keypoint_from_depths[0],     # np array
                 'fullpoint_from_depth':fullpoint_from_depths[0]}   # np array
        # frame: matches_3d,gt_pose,rt_matrix,camera_Ks
        memory_pool.refine_new_frame(frame,record_vis=cfg.record_vis,global_step=global_step,vis_dict=vis_dict)
    memory_pool.eliminate_all_frames_and_compute_result()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1, help='batch size')
    parser.add_argument('--S', type=int, default=1, help='sequence length')
    parser.add_argument('--shuffle',default=False, action='store_true')
    parser.add_argument('--ref_dir', type=str, default='/root/autodl-tmp/shiqian/code/gripper/ref_views/franka_69.4_840')
    parser.add_argument('--test_dir', type=str, default='/root/autodl-tmp/shiqian/datasets/Ty_data')
    parser.add_argument('--feat_layer', type=str, default=19)
    parser.add_argument('--max_iter',type=int, default=500)
    parser.add_argument('--device',type=str,default='cuda:3')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--refine_mode',type=str,default='c')
    parser.add_argument('--max_number',type=int, default=32)
    parser.add_argument('--key_number',type=int,default=8)
    parser.add_argument('--record_vis',default=False,action='store_true')
    parser.add_argument('--view_number',type=int,default=10)
    parser.add_argument('--use_depth',default=True, action='store_true')
    parser.add_argument('--use_full_depth',default=True, action='store_true')
    parser.add_argument('--gripper_point_path', type=str, default="./pointclouds/gripper.txt")
    parser.add_argument('--add_noise',default=True, action='store_true')
    parser.add_argument('--adjust',default=False, action='store_true')

    cfg = parser.parse_args()
    main(cfg)