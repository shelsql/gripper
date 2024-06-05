from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import math
import os
import time

from utils.spd import get_2dbboxes, create_3dmeshgrid, transform_batch_pointcloud_torch
from utils.spd import save_pointcloud, transform_pointcloud_torch, project_points
from utils.spd import depth_map_to_pointcloud, read_pointcloud, transform_pointcloud,depth_map_to_pointcloud_tensor
# from utils.spd import calc_masked_batch_var, calc_coords_3d_var
from utils.quaternion_utils import *


import threading
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from models.uni3d import create_uni3d

def print_time(msg, last_time):
    now_time = time.time()
    print("%s .Time taken: %.2f" % (msg, now_time - last_time))
    return now_time

class Dinov2Matcher:
    def __init__(self, ref_dir, refs, model_pointcloud,
                 dino_layer,
                 repo_name="facebookresearch/dinov2",
                 model_name="dinov2_vitl14_reg",
                 size=448,
                 threshold=0.8,
                 upscale_ratio=1,
                 device="cuda:0",
                 uni3d_layer=19,
                 setting_name=None,
                 dino_name=None,
                 uni3d_name=None,
                 uni3d_color=False,
                 pca=None,
                 dino_pca=None,
                 uni3d_pca=None,
                 cfg=None):
        print("Initializing DinoV2 Matcher...")
        self.repo_name = repo_name
        self.model_name = model_name
        self.size = size
        self.threshold = threshold
        self.upscale_ratio = upscale_ratio
        self.pca=pca
        self.device = device
        self.patch_size = 14
        self.ref_dir = ref_dir
        self.setting_name = setting_name
        self.uni3d_color = uni3d_color
        self.cfg = cfg
        self.dino_pca = dino_pca
        self.uni3d_pca = uni3d_pca

        ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3).squeeze() # B, S, C, H, W
        num_refs, C, H, W = ref_rgbs.shape
        ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3).reshape(num_refs, 1, H, W)
        ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3).reshape(num_refs, 1, H, W)
        c2ws = torch.Tensor(refs['c2ws'][0]).float()
        o2ws = torch.Tensor(refs['obj_poses'][0]).float()
        
        #ref_rgbs = F.interpolate(ref_rgbs, scale_factor=0.15, mode="bilinear")
        #ref_depths = F.interpolate(ref_depths, scale_factor=0.15, mode="bilinear")
        #ref_masks = F.interpolate(ref_masks, scale_factor=0.15, mode="nearest")
        # ref_rgbs[ref_masks.repeat(1,3,1,1) == 0] = 0  注意，这个操作发生了更改
        
        ref_images = torch.concat([ref_rgbs, ref_depths, ref_masks], axis = 1).to(device)
    
        self.ref_c2ws = c2ws.to(device)
        self.ref_w2cs = torch.stack([torch.linalg.inv(self.ref_c2ws[i]) for i in range(num_refs)], axis = 0)
        self.ref_o2ws = o2ws.to(device)
        self.ref_w2os = torch.stack([torch.linalg.inv(self.ref_o2ws[i]) for i in range(num_refs)], axis = 0)
        self.ref_c2os = torch.bmm(self.ref_w2os, self.ref_c2ws)
        self.model_pc = torch.tensor(model_pointcloud, device=device)
        self.ref_images = ref_images
        self.ref_intrinsics = refs['intrinsics']
        '''
        save_pointcloud(self.model_pc.cpu().numpy(), "pointclouds/model.txt")
        
        for i in range(num_refs):
            ref_pointcloud = depth_map_to_pointcloud(ref_depths[i,0], ref_masks[i,0], refs['intrinsics'])
            ref_pointcloud = transform_pointcloud(ref_pointcloud, self.ref_c2os[i].cpu().numpy())
            save_pointcloud(ref_pointcloud, "pointclouds/ref_image_%.3d.txt" % i)
        '''
        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
        print("Preprocessing reference views...")
        cropped_ref_rgbs, cropped_ref_masks, ref_bboxes = self.prepare_images(ref_images)

        self.dino_layer = dino_layer
        if dino_layer>0:
            self.model = torch.hub.load(repo_or_dir="../dinov2",source="local", model=model_name, pretrained=False)
            self.model.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
            self.model.to(self.device)
            self.model.eval()

        # if refs['feats'] is None:
        #     self.model = torch.hub.load(repo_or_dir="../dinov2",source="local", model=model_name, pretrained=False)
        #     self.model.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
        #     self.model.to(self.device)
        #     self.model.eval()
        #     #print("Calculating reference view features...")
        #
        #     #print(cropped_ref_rgbs.shape)
        #     self.ref_features = self.extract_features(cropped_ref_rgbs,ref_bboxes)
        # else:
        self.ref_features = torch.tensor(refs['feats'])[0].float().to(device)   # ref的feature肯定要提前提好
        print("Reference view features obtained")

        N_refs, feat_C, feat_H, feat_W = self.ref_features.shape
        self.N_refs = N_refs
        assert(feat_H == feat_W)
        feat_size = feat_H
        self.feat_masks = F.interpolate(cropped_ref_masks, size = (feat_size, feat_size), mode = "nearest") # Nref, 1, 32, 32
        self.ref_bboxes = ref_bboxes

        print("DinoV2 Matcher done initialized")
        #self.vis_rgbs(cropped_ref_rgbs)
        #self.vis_features(cropped_ref_rgbs, self.feat_masks, self.ref_features)
        # TODO process ref images and calculate features


        if self.cfg.pca_type =='together':      # together就用setting name
            lowrank_str = '_lowrank'
        elif self.cfg.pca_type == 'nope':
            lowrank_str = ''
        elif self.cfg.pca_type =='respective':
            lowrank_str = '_lowrank_respective'
            setting_name = dino_name + uni3d_name

        if not os.path.exists(ref_dir + f'/{setting_name}_vision_word_list{lowrank_str}.npy'):
            print("Calculating BoW...")
            self.gen_and_save_refs_bags(setting_name,lowrank_str)
        #exit()
        self.vision_word_list = np.load(ref_dir + f'/{setting_name}_vision_word_list{lowrank_str}.npy') # 2048xc
        self.ref_bags = np.load(ref_dir + f'/{setting_name}_ref_bags{lowrank_str}.npy')

        self.uni3d_layer = uni3d_layer
        if self.uni3d_layer>0:
            self.uni3d = create_uni3d().to(self.device)
            checkpoint = torch.load("checkpoints/model_large.pt", map_location='cpu')
            sd = checkpoint['module']
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            self.uni3d.load_state_dict(sd)

        #print(self.vision_word_list.shape, self.ref_bags.shape)
    # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    def prepare_images(self, images):
        B, C, H, W = images.shape
        rgbs = images[:,0:3] / 255.0 # B, 3, H, W
        depths = images[:,3:4] # B, 1, H, W
        masks = images[:,4:5] # B, 1, H, W
        bboxes = get_2dbboxes(masks[:,0]) # B, 4
        cropped_rgbs = torch.zeros((B, 3, self.size, self.size), device = self.device)
        cropped_masks = torch.zeros((B, 1, self.size, self.size), device = self.device)
        for b in range(B):
            y_min, x_min, y_max, x_max = bboxes[b,0], bboxes[b,1], bboxes[b,2], bboxes[b,3]
            
            cropped_rgb = rgbs[b:b+1, :, y_min:y_max, x_min:x_max]
            cropped_mask = masks[b:b+1, :, y_min:y_max, x_min:x_max]
            cropped_rgb = F.interpolate(cropped_rgb, size=(self.size, self.size), mode="bilinear")
            cropped_mask = F.interpolate(cropped_mask, size=(self.size, self.size), mode="nearest")
            cropped_rgbs[b:b+1] = cropped_rgb
            cropped_masks[b:b+1] = cropped_mask
        cropped_rgbs = self.transform(cropped_rgbs)
        cropped_rgbs = cropped_rgbs.to(self.device)
        # cropped_rgbs[cropped_masks.repeat(1, 3, 1, 1) == 0] = 0  # Mask out background
        bboxes = torch.tensor(bboxes, device = self.device)
        #print(bboxes)
        return cropped_rgbs, cropped_masks, bboxes
    
    def extract_features(self, cropped_rgbs=None,rgbs=None,depths=None,masks=None,intrinsics=None, cropped_masks=None,bboxes=None,test_idx=None):
        # images: B, C, H, W  torch.tensor      B始终等于1
        # b,c,h,w = images.shape
        #print(images.max(), images.min(), images.mean())
        with torch.inference_mode():


            # uni3d_thread = threading.Thread(target=self.extract_dino_features, args=(cropped_rgbs))
            # dino_thread = threading.Thread(target=self.extract_uni3d_features, args=(depths, masks, intrinsics, rgbs,bboxes,test_idx))
            #
            # uni3d_thread.start()
            # dino_thread.start()
            #
            # uni3d_thread.join()
            # dino_thread.join()
            dino_feat = self.extract_dino_features(cropped_rgbs)
            uni3d_feat = self.extract_uni3d_features(depths, masks, intrinsics, rgbs,bboxes,test_idx)

            if self.dino_pca is not None:
                dino_feat = self.dino_pca.transform(dino_feat.reshape(-1, 32 * 32).transpose(1, 0)).transpose(1,0).reshape(-1,32,32)
                uni3d_feat = self.uni3d_pca.transform(uni3d_feat.reshape(-1, 32 * 32).transpose(1, 0)).transpose(1,0).reshape(-1,32,32)

            if (self.dino_layer>0) and (self.uni3d_layer>0):
                feat = torch.cat([dino_feat,uni3d_feat],dim=0)
            elif self.dino_layer <= 0:
                feat = uni3d_feat
            elif self.uni3d_layer<=0:
                feat = dino_feat




            if self.pca is not None:

                if self.cfg.pca_type=='together':
                    feat = self.pca.transform(feat.reshape(-1, 32 * 32).transpose(1, 0)).transpose(1,0).reshape(-1,32,32)    # 256,32,32
                elif self.cfg.pca_type=='sklearn':
                    feat = self.pca.transform(feat.cpu().numpy().reshape(-1, 32 * 32).transpose(1, 0)).transpose(1, 0).reshape(-1, 32,32)
                    feat = torch.tensor(feat,device=self.device,dtype=torch.float32)
            feat = feat.unsqueeze(0)

        return feat # B, C, H, W

    def extract_dino_features(self,cropped_rgbs):
        if self.dino_layer > 0:     # ??怎么偷偷给我把第一个维度删了呢？？
            if len(cropped_rgbs.shape) == 4:
                image_batch = cropped_rgbs[0:0 + 1]
            else:
                image_batch = cropped_rgbs.unsqueeze(0)
            dino_feat = self.model.get_intermediate_layers(image_batch, n=[self.dino_layer])[0]
            B, N_tokens, C = dino_feat.shape
            img_size = 448
            assert (N_tokens == img_size * img_size / 196)
            dino_feat = dino_feat.permute(0, 2, 1).reshape(B, C, img_size // 14, img_size // 14)
            dino_feat = dino_feat / dino_feat.norm(dim=1, keepdim=True)  # 注意这里是新加的，cat之前要都除以norm
            self.dino_feat = dino_feat[0]
            return dino_feat[0]
        else:
            return None

    def extract_uni3d_features(self,depths, masks, intrinsics, rgbs,bboxes,test_idx):
        intrinsic_matrix = torch.zeros((3, 3), device=self.device)
        intrinsic_matrix[0, 0] = intrinsics['fx']
        intrinsic_matrix[1, 1] = intrinsics['fy']
        intrinsic_matrix[0, 2] = intrinsics['cx']
        intrinsic_matrix[1, 2] = intrinsics['cy']
        intrinsic_matrix[2, 2] = 1

        if self.uni3d_layer > 0:
            # step1 先从深度图还原点云，获得512个feature及该group中心坐标
            # 点云要先中心化，但是投影回去的时候不要忘了再把中心位置加回去，不然patch找自己的geo feature时会找歪         重要！！
            print(depths[0][0].shape, masks[0][0].shape, intrinsics)
            point_cloud = depth_map_to_pointcloud_tensor(depths[0][0], masks[0][0], intrinsic_matrix)  # 从5ms优化到了0.5ms

            point_center = point_cloud.mean(0)
            point_cloud = point_cloud - point_center
            if self.uni3d_color:
                rgb = rgbs[0][:, masks[0][0].bool()].transpose(-1, -2).to(self.device) / 255.0  # 在这里选要不要用颜色
            else:
                rgb = torch.ones_like(point_cloud)

            #print(point_cloud.shape, rgb.shape)
            while point_cloud.shape[0] < 64:
                if point_cloud.shape[0] == 0:
                    point_cloud = torch.zeros((1, 3), device=self.device)
                    rgb = torch.ones_like(point_cloud)
                point_cloud = torch.cat([point_cloud, point_cloud], dim = 0)
                rgb = torch.cat([rgb, rgb], dim = 0)
            geo_feat, geo_center = self.uni3d.encode_pc(torch.cat([point_cloud.unsqueeze(0), rgb.unsqueeze(0)], dim=-1),
                                                        layer=self.uni3d_layer)

            geo_feat = geo_feat / geo_feat.norm(dim=-1, keepdim=True)  # cat之前都除以norm

            geo_center = geo_center + point_center


            projected_coordinates = intrinsic_matrix @ geo_center[0].transpose(-1, -2)
            projected_coordinates = (projected_coordinates[:2, :] / projected_coordinates[2, :]).transpose(-1, -2)
            proj_2d = torch.zeros_like(projected_coordinates)
            proj_2d[:, 0] = projected_coordinates[:, 1]
            proj_2d[:, 1] = projected_coordinates[:, 0]

            # step2 把512个点云投到2d，每个patch直接用最近的那个作为自己的geo feature

            test_2d_coords = self.idx_to_2d_coords(test_idx, bboxes[0:0 + 1])  # 这个是在360,640中的坐标

            dis = torch.norm(test_2d_coords[:, 1:].unsqueeze(1) - proj_2d.unsqueeze(0), dim=-1)  # 307?,512
            cor_id = dis.min(dim=1).indices  # 307？,范围为（0~512）

            uni3d_feat = torch.zeros((1024, 32, 32), device=self.device)
            uni3d_feat[:, test_idx[:, 1], test_idx[:, 2]] = geo_feat[0, 1:, :][cor_id].transpose(-1,-2)  # 1024,32,32 mask之外是全0

            self.uni3d_feat = uni3d_feat
            return uni3d_feat
        else:
            return None
    
    def idx_to_2d_coords_2(self, idxs, feat_size, test_bboxes):
        N, D = idxs.shape # batchno, featidx, refno, featidx
        assert(D == 4)
        coords = torch.zeros((N,6), device = self.device) # batchno, coords, refno, coords
        coords[:,0] = idxs[:,0] # batchno
        coords[:,3] = idxs[:,2] # refno
        test_bboxes = test_bboxes[idxs[:,0]] # N, 4
        ref_bboxes = self.ref_bboxes[idxs[:,2]]
        # Turn token idx into coord within 448*448 box
        coords[:,1] = (((idxs[:,1] // feat_size) * self.patch_size) + self.patch_size / 2) # Coord within 448*448 box
        coords[:,2] = (((idxs[:,1] % feat_size) * self.patch_size) + self.patch_size / 2) # Coord within 448*448 box
        coords[:,4] = (((idxs[:,3] // feat_size) * self.patch_size) + self.patch_size / 2) # Coord within 448*448 box
        coords[:,5] = (((idxs[:,3] % feat_size) * self.patch_size) + self.patch_size / 2) # Coord within 448*448 box
        # Turn coord within 448*448 box to coord on full image
        coords[:,1] = (coords[:,1] / 448.0 * (test_bboxes[:,2] - test_bboxes[:,0])) + test_bboxes[:,0]
        coords[:,2] = (coords[:,2] / 448.0 * (test_bboxes[:,3] - test_bboxes[:,1])) + test_bboxes[:,1]
        coords[:,4] = (coords[:,4] / 448.0 * (ref_bboxes[:,2] - ref_bboxes[:,0])) + ref_bboxes[:,0]
        coords[:,5] = (coords[:,5] / 448.0 * (ref_bboxes[:,3] - ref_bboxes[:,1])) + ref_bboxes[:,1]
        #TODO finish this
        return coords
    
    def idx_to_2d_coords(self, idxs, bboxes):
        N, D = idxs.shape # imgidx, feat x, feat y
        assert(D == 3)
        coords = torch.zeros_like(idxs)
        coords[:,0] = idxs[:,0]
        batch_bboxes = bboxes[idxs[:,0]]
        # Turn token idx into coord within 448*448 box
        coords[:,1:3] = idxs[:,1:3] * self.patch_size + self.patch_size / 2
        # Turn coord within 448*448 box to coord on full image
        coords[:,1:3] = (coords[:,1:3] / 448.0 * (batch_bboxes[:,2:4] - batch_bboxes[:,0:2])) + batch_bboxes[:,0:2]
        return coords
    
    def coords_2d_to_3d(self, coords, depth_maps, intrinsics, c2os):
        N, D = coords.shape
        assert(D==3)
        
        # Unpack intrinsic matrix
        fx = intrinsics['fx'].item()
        fy = intrinsics['fy'].item()
        cx = intrinsics['cx'].item()
        cy = intrinsics['cy'].item()
        
        #print("depth_maps", depth_maps.shape)
        depths = depth_maps[coords[:,0].int(), coords[:,1].int(), coords[:,2].int()]
        #print("depths", depths.shape)
        cam_space_x = (coords[:,2] - cx) * depths / fx
        cam_space_y = (coords[:,1] - cy) * depths / fy
        cam_space_z = depths
        cam_space_coords = torch.stack([cam_space_x, cam_space_y, cam_space_z], axis = 1) # N, 3
        
        c2os = c2os[coords[:, 0].int()]
        #print("c2os:", c2os.shape)
        obj_space_coords = transform_batch_pointcloud_torch(cam_space_coords, c2os)
        #print(obj_space_coords.shape)
        
        ref_ids = coords[:,0:1]
        ids_and_coords_3d = torch.concat([ref_ids, obj_space_coords], axis = 1)
        
        return ids_and_coords_3d
    
    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens
    
    def match_and_fuse(self, sample, step=0):

        if len(torch.Tensor(sample['rgb']).shape) == 3:  # 直接用dataset[i]取时
            rgbs = torch.Tensor(sample['rgb']).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # B, C, H, W
            depths = torch.Tensor(sample['depth']).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            masks = torch.Tensor(sample['mask']).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        else:
            rgbs = torch.Tensor(sample['rgb']).float().permute(0, 3, 1, 2).to(self.device)  # B, C, H, W
            depths = torch.Tensor(sample['depth']).float().permute(0, 3, 1, 2).to(self.device)
            masks = torch.Tensor(sample['mask']).float().permute(0, 3, 1, 2).to(self.device)
        
        rgbs[masks.repeat(1,3,1,1) == 0] = 0
        
        images = torch.concat([rgbs, depths[:,0:1], masks[:,0:1]], axis = 1)
        N_refs, feat_C, feat_H, feat_W = self.ref_features.shape
        B, C, H, W = rgbs.shape
        assert(feat_H == feat_W)
        feat_size = feat_H
        #print(images[:,4:5].sum())
        cropped_rgbs, cropped_masks, bboxes = self.prepare_images(images)
        if sample['feat'] is None:
            features = self.extract_features(cropped_rgbs) # B, 1024, 32, 32
        else:
            if len(torch.Tensor(sample['rgb']).shape) == 3:
                features = torch.tensor(sample['feat']).float().unsqueeze(0).to(self.device)
            else:
                features = torch.tensor(sample['feat']).float().to(self.device)
        N_tokens = feat_H * feat_W
        #print(features.shape)
        features = features.permute(0, 2, 3, 1).reshape(-1, feat_C) # B*N, C
        ref_features = self.ref_features.permute(0, 2, 3, 1).reshape(-1, feat_C) # 32*N, C
        cosine_sims = pairwise_cosine_similarity(features, ref_features) # B*N, 32*N
        cosine_sims = cosine_sims.reshape(B, N_tokens, N_refs, N_tokens) # B, 1024, 32, 1024
        cosine_sims = cosine_sims.reshape(B, feat_H, feat_W, N_refs, feat_H, feat_W) # B, 32, 32, Nref, 32, 32
        print("cosine sims max:", cosine_sims.max())
        batch_feat_masks = F.interpolate(cropped_masks, size=(feat_H, feat_W), mode = "nearest") # B, 1, 32, 32
        
        #self.vis_corr_map(cosine_sims, batch_feat_masks, cropped_rgbs)
        
        #batch_max_sims = torch.max(cosine_sims.reshape(B, feat_H, feat_W, N_refs, feat_H*feat_W), axis = 4) # B, 32, 32, Nref
        #good_refs = batch_max_sims > 0.9 # B, 32, 32, N_ref   bool
        cosine_sims = cosine_sims[batch_feat_masks[:,0] > 0]  # N_batch_pts, Nref, 32, 32
        #TODO calculate vars
        #cosine_sims_vars = calc_masked_batch_var(cosine_sims, self.feat_masks, self.device)
        #print(cosine_sims_vars.shape)

        test_idxs = create_3dmeshgrid(B, feat_H, feat_W, self.device)
        test_idxs = test_idxs[batch_feat_masks[:,0] > 0] # N_batch_pts, 3

        selected_refs = self.select_refs(features,batch_feat_masks,B,test_idxs,n=10).reshape(-1)    # b,10 -> 10  TODO 现在只能batch=1

        select_mask = torch.zeros_like(self.feat_masks,device=self.device)
        select_mask[selected_refs] = 1
        selected_feat_masks = self.feat_masks * select_mask

        cosine_sims = cosine_sims[:, selected_feat_masks[:,0] > 0]  # N_batch_pts, N_ref_pts
        ref_idxs = create_3dmeshgrid(N_refs, feat_H, feat_W, self.device)
        ref_idxs = ref_idxs[selected_feat_masks[:,0] > 0] # N_ref_pts, 3
        #good_refs = good_refs[batch_feat_masks[:,0] > 0] # N_batch_pts, N_ref



        test_2d_coords = self.idx_to_2d_coords(test_idxs, bboxes) # N_test_2d_pts, 3
        ref_2d_coords = self.idx_to_2d_coords(ref_idxs, self.ref_bboxes)
        ref_3d_coords = self.coords_2d_to_3d(ref_2d_coords, self.ref_images[:,3], self.ref_intrinsics, self.ref_c2os)
        
        ref_valid_idxs = torch.logical_and(ref_3d_coords[:,1]<10, ref_3d_coords[:,1]>-10)
        ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,2]<10)
        ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,2]>-10)
        ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,3]<10)
        ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,3]>-10)
        
        #ref_valid_idxs = torch.logical_and(ref_valid_idxs, cosine_sims)
        
        cosine_sims = cosine_sims[:, ref_valid_idxs] # N_test_2d_pts, N_3d_ref_pts
        ref_3d_coords = ref_3d_coords[ref_valid_idxs] # N_3d_ref_pts, 4
        
        num_good_sims = torch.sum(cosine_sims > self.threshold, dim = 1)
        good_pts = num_good_sims > 5 #Shape N_2d_pts
        cosine_sims = cosine_sims[good_pts]
        test_2d_coords = test_2d_coords[good_pts]
        print("cosine sims", cosine_sims.shape)
        
        ref_3d_coords_1 = ref_3d_coords.unsqueeze(0).repeat(cosine_sims.shape[0], 1, 1)
        test_2d_coords_1 = test_2d_coords.unsqueeze(1).repeat(1, cosine_sims.shape[1], 1)
        cosine_sim_coords = torch.concat([test_2d_coords_1, ref_3d_coords_1], axis=2)
        sims_and_coords = torch.concat([cosine_sims.unsqueeze(2), cosine_sim_coords], axis=2) # N_test_2d_pts, N_3d_ref_pts, 8 (sim, batchid, x, y, refid, x, y, z)
        
        #####  3D Fusion: Variance (Not good, trying clustering)
            
        #print(sims_and_coords.shape)
        #target_point_vars = calc_coords_3d_var(sims_and_coords, self.threshold)
        #var_threshold = 500
        
        #print(target_point_vars.shape)
        #print(target_point_vars.max(), target_point_vars.min(), target_point_vars.mean())
        #print(torch.max(target_point_vars, dim=1), torch.min(target_point_vars, dim=1), torch.mean(target_point_vars))
        #print(target_point_vars.max(), target_point_vars.min(), target_point_vars.mean())
        #print(target_point_vars[:10])
        #print(torch.tensor(target_point_vars>0.1)[:10])
        #exit()
        
        # self.save_sim_pts(cosine_sims, ref_3d_coords)
        
        ##### 3D Fusion: Gaussian Smoothing
        
        #dists_3d = pairwise_euclidean_distance(self.model_pc, ref_3d_coords[:,1:]).float() # N_pts, N_ref_pts
        # Generate similarity field. We want shape N_pts, N_test_2d_pts
        #gaussian_var = 0.002
        #gaussian_coeff = (1.0 / (gaussian_var*(math.sqrt(2*3.14159265)))) * torch.exp(-0.5 * torch.square(dists_3d / gaussian_var)) # N_pts, N_ref_pts
        #print(gaussian_coeff.dtype, cosine_sims.dtype, ref_3d_coords.dtype)
        #sim_field = torch.matmul(gaussian_coeff, cosine_sims.permute(1, 0)) / ref_3d_coords.shape[0]
        #self.vis_sim_field(sim_field, test_2d_coords, images)
        #self.save_sim_field(sim_field)
        #print(cosine_sims.shape, ref_3d_coords.shape, sim_field.shape)
        #print(cosine_sims.max(), cosine_sims.min(), sim_field.max(), sim_field.min())
        #threshold = sim_field.max() * 0.8
        
        ##### DBSCAN clustering
        matches = []
        noise_threshold = 0.2
        # Choose by threshold
        # good_idxs = cosine_sims > self.threshold # Bool N_2d_pts, N_3d_pts
        # Choose top N
        top_N = 50
        _, sorted_idxs = torch.sort(cosine_sims, descending=True, dim = 1)
        good_idxs = sorted_idxs[:, :top_N]
        dbscan = DBSCAN(eps = 0.008, min_samples=5)
        for idx_2d in range(cosine_sims.shape[0]):
            good_coords_3d = ref_3d_coords[good_idxs[idx_2d], 1:].cpu().numpy() # N,3 numpy
            labels = dbscan.fit(good_coords_3d).labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            n_total = labels.shape[0]
            noise_ratio = n_noise / n_total
            #print(noise_ratio)
            if noise_ratio > noise_threshold:
                continue
            for i in range(n_clusters):
                cluster_center = torch.tensor(np.mean(good_coords_3d[labels == i], axis = 0), device = self.device) # 3
                pt_2d = test_2d_coords[idx_2d]
                match = torch.concat([pt_2d, cluster_center], dim = 0)
                #print(match.shape)
                matches.append(match)
        
        if len(matches) == 0:
            return None
        matches = torch.stack(matches, dim = 0)
        print(matches.shape)
        
        #print(cosine_sims.shape)
        #max_sims, max_idxs = torch.max(cosine_sims, axis = 1) # max_idxs shape N_2d_pts
        #max_idxs = max_idxs[target_point_vars < var_threshold]
        #print(max_idxs.shape)
        #matches = torch.zeros((max_idxs.shape[0],6), device = self.device)
        #matches[:,3:] = ref_3d_coords[max_idxs, 1:]
        #print(test_2d_coords.shape, (max_sims>threshold).shape)
        #matches[:,:3] = test_2d_coords[target_point_vars < var_threshold]
        #save_pointcloud(ref_3d_coords.cpu().numpy(), "./pointclouds/ref_3d_coords.txt")
        #print(matches[:10])
        #self.vis_3d_matches(images, matches, selected_refs)
        return matches

    def gen_ref_centers(self,sample, step=0, N_select=30,gt_pose=None):
        # timestamp = time.time()
        rgbs = torch.Tensor(sample['rgb']).float().to(self.device)  # B, C, H, W
        depths = torch.Tensor(sample['depth']).float().to(self.device)
        masks = torch.Tensor(sample['mask']).float().to(self.device)
        if len(rgbs.shape) == 5:
            rgbs = rgbs[0]
            depths = depths[0]
            masks = masks[0]
        rgbs = rgbs.permute(0, 3, 1, 2)
        depths = depths.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        # print(masks.sum())
        rgbs[masks.repeat(1, 3, 1, 1) == 0] = 0

        images = torch.concat([rgbs, depths[:, 0:1], masks[:, 0:1]], axis=1)
        B, C, H, W = images.shape
        N_refs, feat_C, feat_H, feat_W = self.ref_features.shape
        assert (feat_H == feat_W)
        feat_size = feat_H
        # timestamp = print_time("Load data", timestamp)
        cropped_rgbs, cropped_masks, bboxes = self.prepare_images(images)
        if sample['feat'] is None:
            features = self.extract_features(cropped_rgbs)  # B, 1024, 32, 32
        else:
            features = torch.tensor(sample['feat']).float().to(self.device)
            # print(features.shape)
            if len(features.shape) == 5:
                features = features[0]
        N_tokens = feat_H * feat_W
        # print(features.shape)
        features = features.permute(0, 2, 3, 1).reshape(N_tokens, feat_C)  # B*N, C

        batch_feat_masks = F.interpolate(cropped_masks, size=(feat_H, feat_W), mode="nearest")  # B, 1, 32, 32
        test_idxs = create_3dmeshgrid(B, feat_H, feat_W, self.device)
        test_idxs = test_idxs[batch_feat_masks[:, 0] > 0]  # N_batch_pts, 3
        # timestamp = print_time("Prepare data", timestamp)

        selected_refs = self.select_refs(features, batch_feat_masks, B, test_idxs, n=N_select, gt_pose=None).reshape(-1)
        ref_poses = self.ref_c2ws[selected_refs][:,:3,:3]    # 30,3,3?
        R = torch.matmul(ref_poses.unsqueeze(1),ref_poses.transpose(-1,-2))
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
        theta_matrix = torch.arccos(torch.clip((trace-1)/2,-1,1)) *180/torch.pi
        clustering = DBSCAN(eps=40,min_samples=5,metric='precomputed').fit(theta_matrix.cpu().numpy())

        ref_centers = []
        matches_3d_list = []
        # 统计每个类别的元素个数
        unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
        for label in unique_labels:
            if label == -1:
                continue
            indices = np.where(clustering.labels_ == label)[0]

            # 求旋转的均值，加了权重
            weight = (torch.cos(torch.arange(len(indices)) / len(indices) * torch.pi / 2 + torch.pi / 2) + 1).to(self.device)
            # weight = torch.zeros(len(indices), device=self.device)  # 只用第一个会不会好一点？
            # weight[0] = 1
            q_avg = average_quaternion_torch(matrix_to_quaternion(ref_poses[torch.tensor(indices, device=self.device)]),weight)
            ref_center = quaternion_to_matrix(q_avg)
            ref_centers.append(ref_center)

            selected_ref = self.select_refs(features, batch_feat_masks, B, test_idxs, n=1, gt_pose=ref_center.cpu().numpy()).reshape(-1)
            # selected_ref = self.select_refs(features, batch_feat_masks, B, test_idxs, n=1,gt_pose=gt_pose).reshape(-1)

            # 仅debug，看一下选的ref和gt差了多少
            # diff_R = torch.tensor(gt_pose[:3,:3],device=self.device,dtype=torch.float32)@ref_center.transpose(-1,-2)
            # diff_trace = torch.diagonal(diff_R, dim1=-2, dim2=-1).sum(-1)
            # diff_theta_matrix = torch.arccos(torch.clip((diff_trace - 1) / 2, -1, 1)) * 180 / torch.pi


            seleced_refs_slice = selected_ref
            length = 1
            ref_features = self.ref_features.permute(0, 2, 3, 1)
            ref_features = ref_features[seleced_refs_slice].reshape(length * N_tokens, feat_C)  # 32*N, C
            cosine_sims = pairwise_cosine_similarity(features, ref_features)  # B*N, 32*N
            cosine_sims = cosine_sims.reshape(B, feat_H, feat_W, length, feat_H, feat_W)  # B, 32, 32, Nref, 32, 32
            cosine_sims = cosine_sims[batch_feat_masks[:, 0] > 0]  # N_batch_pts, Nref, 32, 32
            feat_masks = self.feat_masks[seleced_refs_slice]
            cosine_sims = cosine_sims[:, feat_masks[:, 0] > 0]  # N_batch_pts, N_ref_pts
            ref_idxs = create_3dmeshgrid(N_refs, feat_H, feat_W, self.device)
            ref_idxs = ref_idxs[seleced_refs_slice]
            ref_idxs = ref_idxs[feat_masks[:, 0] > 0]  # N_ref_pts, 3
            test_2d_coords = self.idx_to_2d_coords(test_idxs, bboxes)  # N_test_2d_pts, 3
            ref_2d_coords = self.idx_to_2d_coords(ref_idxs, self.ref_bboxes)
            ref_3d_coords = self.coords_2d_to_3d(ref_2d_coords, self.ref_images[:, 3], self.ref_intrinsics,
                                                 self.ref_c2os)
            ref_valid_idxs = torch.logical_and(ref_3d_coords[:, 1] < 10, ref_3d_coords[:, 1] > -10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:, 2] < 10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:, 2] > -10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:, 3] < 10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:, 3] > -10)
            cosine_sims = cosine_sims[:, ref_valid_idxs]  # N_test_2d_pts, N_3d_ref_pts
            ref_3d_coords = ref_3d_coords[ref_valid_idxs]  # N_3d_ref_pts, 4
            max_sims, max_inds = torch.max(cosine_sims, axis=1)
            good_matches = max_sims > self.threshold
            if good_matches.sum() < 4:
                print("Not enough good matches, lowering threshold")
                good_matches = max_sims > (self.threshold - 0.1)
            max_inds = max_inds[good_matches]
            match_2d_coords = test_2d_coords[good_matches]
            match_3d_coords = ref_3d_coords[max_inds, 1:]
            matches_3d = torch.concat([match_2d_coords, match_3d_coords], dim=1)
            matches_3d_list.append(matches_3d)  # list of different shape tensors

        return ref_centers,matches_3d_list

    def match_batch(self, sample, step=0, N_select=1,refine_mode='a',gt_pose=None,cluster=False,time_statistics_dict=None):
        timestamp = time.time()
        time0 = time.time()
        rgbs = torch.Tensor(sample['rgb']).float().to(self.device) # B, C, H, W
        depths = torch.Tensor(sample['depth']).float().to(self.device)
        masks = torch.Tensor(sample['mask']).float().to(self.device)
        intrinsics = sample['intrinsics']
        if len(rgbs.shape) == 5:
            rgbs = rgbs[0]
            depths = depths[0]
            masks = masks[0]
        rgbs = rgbs.permute(0, 3, 1, 2)
        depths = depths.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        #print(masks.sum())
        rgbs[masks.repeat(1,3,1,1) == 0] = 0
        
        images = torch.concat([rgbs, depths[:,0:1], masks[:,0:1]], axis = 1)
        B, C, H, W = images.shape
        N_refs, feat_C, feat_H, feat_W = self.ref_features.shape
        assert(feat_H == feat_W)
        feat_size = feat_H
        #timestamp = print_time("Load data", timestamp)
        cropped_rgbs, cropped_masks, bboxes = self.prepare_images(images)

        time0 = time.time()
        batch_feat_masks = F.interpolate(cropped_masks, size=(feat_H, feat_W), mode = "nearest") # B, 1, 32, 32
        time1 = time.time()
        test_idxs = create_3dmeshgrid(B, feat_H, feat_W, self.device)
        time2 = time.time()
        test_idxs = test_idxs[0][batch_feat_masks[0,0] > 0] # N_batch_pts(287?), 3     在32*32中的坐标
        time3 = time.time()
        time_statistics_dict['interpolate time'].append(time1-time0)
        time_statistics_dict['3dmesh time'].append(time2-time1)
        time_statistics_dict['mask time'].append(time3-time2)

        if sample['feat'] is None:

            features = self.extract_features(cropped_rgbs=cropped_rgbs,rgbs=rgbs,depths=depths,masks=masks,intrinsics=intrinsics,cropped_masks=cropped_masks,bboxes=bboxes,test_idx=test_idxs) # B, 1024, 32, 32
            extra_end = time.time()
            # print("Feature extraction took",extra_end-time3,"s")
            time_statistics_dict['extract feature time'].append(extra_end-time3)
        else:
            features = torch.tensor(sample['feat']).float().to(self.device)
            #print(features.shape)
            if len(features.shape) == 5:
                features = features[0]
        N_tokens = feat_H * feat_W
        #print(features.shape)
        features = features.permute(0, 2, 3, 1).reshape(N_tokens, feat_C) # B*N, C


        #timestamp = print_time("Prepare data", timestamp)

        matches_3d_list = []
        ref_pose_list = []

        time_ref_start = time.time()
        selected_refs = self.select_refs(features, batch_feat_masks, B, test_idxs, n=N_select,gt_pose=None).reshape(-1)  # TODO 现在只能batch=1
        time_ref_end = time.time()
        time_statistics_dict['select ref time'].append(time_ref_end-time_ref_start)
        if refine_mode == 'c':
            selected_refs = selected_refs.unsqueeze(0)
        for i,idx in enumerate(selected_refs):
            seleced_refs_slice = selected_refs[i:i + 1]
            length = 1
            if refine_mode == 'c':
                seleced_refs_slice = seleced_refs_slice[0]
                length = N_select

            ref_features = self.ref_features.permute(0, 2, 3, 1)
            ref_features = ref_features[seleced_refs_slice].reshape(length*N_tokens, feat_C) # 1*1024, C
            cosine_sims = pairwise_cosine_similarity(features, ref_features) # 1024,1024
            #cosine_sims = cosine_sims.reshape(B, N_tokens, N_refs, N_tokens) # B, 1024, 32, 1024
            #timestamp = print_time("Calculate corrs", timestamp)
            cosine_sims = cosine_sims.reshape(B, feat_H, feat_W, length, feat_H, feat_W) # B, 32, 32, Nref, 32, 32

            cosine_sims = cosine_sims[batch_feat_masks[:,0] > 0]  # N_batch_pts, Nref, 32, 32

            #print("cosine sims max:", cosine_sims.max())
            feat_masks = self.feat_masks[seleced_refs_slice]
            cosine_sims = cosine_sims[:, feat_masks[:,0] > 0]  # N_batch_pts, N_ref_pts

            ref_idxs = create_3dmeshgrid(N_refs, feat_H, feat_W, self.device)
            ref_idxs = ref_idxs[seleced_refs_slice]
            ref_idxs = ref_idxs[feat_masks[:,0] > 0] # N_ref_pts, 3
            #print(cosine_sims.shape)
            #timestamp = print_time("Some filtering", timestamp)
            test_2d_coords = self.idx_to_2d_coords(test_idxs, bboxes) # N_test_2d_pts, 3    3分别是第几张test iamge,在原尺寸图像中的坐标
            ref_2d_coords = self.idx_to_2d_coords(ref_idxs, self.ref_bboxes) # 321,3   # 3分别是ref image的序号,在原尺寸图像中的坐标
            ref_3d_coords = self.coords_2d_to_3d(ref_2d_coords, self.ref_images[:,3], self.ref_intrinsics, self.ref_c2os)   # 321,4     4分别是ref image的序号，点云坐标

            ref_valid_idxs = torch.logical_and(ref_3d_coords[:,1]<10, ref_3d_coords[:,1]>-10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,2]<10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,2]>-10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,3]<10)
            ref_valid_idxs = torch.logical_and(ref_valid_idxs, ref_3d_coords[:,3]>-10)
            #timestamp = print_time("Calculate coords", timestamp)
            #ref_valid_idxs = torch.logical_and(ref_valid_idxs, cosine_sims)

            cosine_sims = cosine_sims[:, ref_valid_idxs] # N_test_2d_pts, N_3d_ref_pts
            ref_3d_coords = ref_3d_coords[ref_valid_idxs] # N_3d_ref_pts, 4

            #print(test_2d_coords.shape, ref_3d_coords.shape)

            max_sims, max_inds = torch.max(cosine_sims, axis = 1)
            good_matches = max_sims > self.threshold
            tmp_threhold = self.threshold
            while good_matches.sum() < 4:
                print("Not enough good matches, lowering threshold")
                tmp_threhold = tmp_threhold-0.1
                good_matches = max_sims > tmp_threhold

            max_inds = max_inds[good_matches]
            match_2d_coords = test_2d_coords[good_matches]
            match_3d_coords = ref_3d_coords[max_inds,1:]
            matches_3d = torch.concat([match_2d_coords, match_3d_coords], dim = 1)
            #self.vis_3d_matches(images, matches_3d, seleced_refs_slice, step)
            matches_3d_list.append(matches_3d)      # list of different shape tensors
            ref_pose_list.append(self.ref_c2ws[idx])     # list of tensors
            #timestamp = print_time("Choose matches", timestamp)
        #print(matches_3d.shape)
        time_match_end = time.time()
        time_statistics_dict['match time'].append(time_match_end-time_ref_end)
        return matches_3d_list, ref_pose_list,selected_refs

    def match_2d_to_3d(self, matches_2d):
        #TODO: Project 2D coords to gripper space using ref intrinsics
        # matches_2d: N, 6  batchno, coords, refno, coords
        # want to get N, 6  batchno, coords, 3dcoords
        
        # Unpack intrinsic matrix
        fx = self.ref_intrinsics['fx'].item()
        fy = self.ref_intrinsics['fy'].item()
        cx = self.ref_intrinsics['cx'].item()
        cy = self.ref_intrinsics['cy'].item()
        
        print("matches_2d:", matches_2d.shape)
        ref_depths = self.ref_images[:,3] # N_ref, H, W
        depths = ref_depths[matches_2d[:,3].int(), matches_2d[:,4].int(), matches_2d[:,5].int()]
        cam_space_x = (matches_2d[:,5] - cx) * depths / fx
        cam_space_y = (matches_2d[:,4] - cy) * depths / fy
        cam_space_z = depths
        cam_space_coords = torch.stack([cam_space_x, cam_space_y, cam_space_z], axis = 1) # N, 3
        print("cam_space_coords:", cam_space_coords.shape)
        c2os = self.ref_c2os[matches_2d[:, 3].int()]
        print("c2os:", c2os.shape)
        world_space_coords = transform_batch_pointcloud_torch(cam_space_coords, c2os)
        matches_3d = torch.zeros_like(matches_2d)
        matches_3d[:,:3] = matches_2d[:,:3]
        matches_3d[:,3:] = world_space_coords
        return matches_3d

    def save_sim_pts(self, cosine_sims, ref_3d_coords):
        pt_id = 0
        sims = cosine_sims[pt_id]
        sims = (sims - sims.min()) / (sims.max() - sims.min())
        rgbs = torch.zeros_like(ref_3d_coords[:,1:])
        rgbs[:,0] = sims
        rgbs[:,2] = 1 - sims
        rgbs *= 255
        rgb_pts = torch.concat([ref_3d_coords[:,1:], rgbs], axis=1)
        save_pointcloud(rgb_pts, "./pointclouds/sim_pts.txt")
        
    def save_sim_field(self, sim_field):
        pt_id = 45
        sims = sim_field[:,pt_id]
        sims = (sims - sims.min()) / (sims.max() - sims.min())
        rgbs = torch.zeros_like(self.model_pc)
        rgbs[:,0] = sims
        rgbs[:,2] = 1 - sims
        rgbs *= 255
        rgb_pts = torch.concat([self.model_pc, rgbs], axis=1)
        save_pointcloud(rgb_pts, "./pointclouds/sim_field.txt")
    
    def vis_2d_matches(self, images, matches_2d, size = 360):
        #TODO: Resize images, transform coords, draw lines
        B, C, H, W = images.shape
        assert(C == 5)
        rgb_0 = images[0, :3].permute(1, 2, 0).cpu().numpy() # H, W, 3
        N_ref, C, H_ref, W_ref = self.ref_images.shape
        N, D = matches_2d.shape
        matches_2d = matches_2d[matches_2d[:,0] == 0]
        assert(D == 6)
        for i_ref in tqdm(range(self.N_refs)):
            ref_rgb = self.ref_images[i_ref, :3].permute(1, 2, 0).cpu().numpy() # H, W, 3
            matches = matches_2d[matches_2d[:,3] == i_ref] # N_match, 6
            full_img = np.zeros((max(H_ref, H), W_ref + W, 3))
            full_img[:H,:W] = rgb_0
            full_img[:H_ref, W:] = ref_rgb
            for i_match in range(matches.shape[0]):
                y_1, x_1 = matches[i_match,1].int().item(), matches[i_match,2].int().item()
                y_2, x_2 = matches[i_match,4].int().item(), matches[i_match,5].int().item()
                x_2 += W
                #print(x_1, y_1, x_2, y_2)
                full_img = cv2.line(full_img, (x_1,y_1), (x_2,y_2), (0,0,255), 1)
            cv2.imwrite("./match_vis/match2d_%.2d.png" % i_ref, full_img)
            
    def vis_3d_matches(self, images, matches_3d, selected_refs, step=0, size = (640,360)):
        #TODO: Resize images, transform coords, draw lines
        B, C, H, W = images.shape
        print(images.shape)
        assert(C == 5)
        rgb_0 = images[0, :3].permute(1, 2, 0).cpu().numpy() # H, W, 3
        rgb_0 = cv2.resize(rgb_0, dsize=size,interpolation=cv2.INTER_LINEAR)
        refs = self.ref_images[selected_refs]
        N_ref, C, H_ref, W_ref = refs.shape
        N, D = matches_3d.shape
        matches_3d = matches_3d[matches_3d[:,0] == 0]
        assert(D == 6)
        camera_K = torch.zeros((3,3), device = self.device)
        camera_K[0,0] = self.ref_intrinsics['fx']
        camera_K[1,1] = self.ref_intrinsics['fy']
        camera_K[0,2] = self.ref_intrinsics['cx']
        camera_K[1,2] = self.ref_intrinsics['cy']
        camera_K[2,2] = 1
        
        for i in tqdm(range(N_ref)):
            i_ref = selected_refs[i]
            ref_rgb = self.ref_images[i_ref, :3].permute(1, 2, 0).cpu().numpy() # H, W, 3
            ref_rgb = cv2.resize(ref_rgb, dsize=size,interpolation=cv2.INTER_LINEAR)
            #matches = matches_3d[matches_3d[:,3] == i_ref] # N_match, 6
            coords_3d = matches_3d[:, 3:]
            if coords_3d.shape[0] == 0:
                print("Ref %.2d skipped"% i_ref)
                #continue
            c2o = self.ref_c2os[i_ref]
            coords_3d_homo = torch.concat([coords_3d, torch.ones((coords_3d.shape[0],1), device = coords_3d.device)], axis=1)
            coords_3d_cam = torch.matmul(torch.linalg.inv(c2o), coords_3d_homo.permute(1,0)).permute(1,0)
            coords_3d_cam = coords_3d_cam[:,:3] / coords_3d_cam[:,3:4]
            coords_2d = torch.matmul(camera_K, coords_3d_cam.permute(1,0)).permute(1,0)
            coords_2d = coords_2d[:,:2] / coords_2d[:,2:3]
            full_img = np.zeros((size[1], size[0]*2, 3))
            full_img[:,:size[0]] = rgb_0
            full_img[:, size[0]:] = ref_rgb
            for i_match in range(matches_3d.shape[0]):
                y_1, x_1 = matches_3d[i_match,1].item(), matches_3d[i_match,2].item()
                y_2, x_2 = coords_2d[i_match,1].item(), coords_2d[i_match,0].item()
                y_1 = int(y_1 * (size[1] / H))
                x_1 = int(x_1 * (size[0] / W))
                y_2 = int(y_2 * (size[1] / H_ref))
                x_2 = int(x_2 * (size[0] / W_ref))
                x_2 += size[0]
                #print(x_1, y_1, x_2, y_2)
                full_img = cv2.line(full_img, (x_1,y_1), (x_2,y_2), (0,0,255), 1)
            cv2.imwrite("./match_vis/match3d_%.2d_%.3d.png" % (step, i), full_img[:,:,::-1])
            
    def vis_features(self, images, feat_masks, feats):
        B, C, H, W = images.shape
        B, C_feat, H_feat, W_feat = feats.shape
        images = images.permute(0, 2, 3, 1).cpu().numpy() # B, H, W, 3
        feats = feats.permute(0, 2, 3, 1).reshape(B, H_feat*W_feat, C_feat).cpu().numpy() # B, H*W, 1024
        feat_masks = feat_masks.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        reshaped_masks = feat_masks.reshape(B, H_feat*W_feat)
        #print(feat_masks.shape)
        for i in range(B):
            pca = PCA(n_components=3)
            feat_map = np.zeros((H_feat, W_feat, 3))
            feats_i = feats[i, reshaped_masks[i] > 0].astype(np.float32)
            for j in range(10):
                #print(feats_i[j, :10])
                pass
            #print(np.max(feats_i), np.min(feats_i), np.mean(feats_i))
            pca_feats = pca.fit_transform(feats_i) # H*W, 3
            #print(np.max(pca_feats), np.min(pca_feats), pca_feats.shape)
            pca_feats = (pca_feats - np.min(pca_feats)) / (np.max(pca_feats) - np.min(pca_feats))
            #print(np.max(pca_feats), np.min(pca_feats), pca_feats.shape)
            feat_map[feat_masks[i] > 0] = pca_feats
            feat_map = feat_map*255
            cv2.imwrite("./match_vis/ref_feat_%.2d.png" % i, feat_map)
    
    def vis_corr_map(self, cosine_sims, batch_feat_mask, cropped_rgbs):
        # cosine sims shape B, feat_H, feat_W, N_ref, feat_H, feat_W
        coords = [14, 5]
        for i in range(32):
            corr_map = cosine_sims[0, coords[0], coords[1], i]
            corr_map[self.feat_masks[i,0]==0] = 0
            corr_map = corr_map.cpu().numpy()
            corr_map = (corr_map - np.min(corr_map)) / (np.max(corr_map) - np.min(corr_map))
            corr_img = np.zeros((32, 32, 3))
            corr_img[:,:,2] = corr_map
            full_img = np.zeros((32, 64, 3))
            full_img[:,32:,:] = corr_img
            rgb = F.interpolate(cropped_rgbs, size = (32, 32))[0].permute(1,2,0).cpu().numpy()
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
            full_img[:,:32,:] = rgb
            full_img[coords[0], coords[1],:] = np.array([0,0,1])
            full_img*=255.0
            cv2.imwrite("./match_vis/corr_map_%.2d.png"%i, full_img)
    
    def vis_sim_pts(self, cosine_sims, test_2d_coords, ref_3d_coords, images):
        # cosine_sims N_2d_pts, N_3d_pts
        # test_2d_coords N_2d_pts, 3
        # images B, 5, H, W
        pt_id = 45
        coord = test_2d_coords[pt_id]
        print("2D coords:", coord)
        sims = cosine_sims[pt_id] # shape N_3d_pts
        w2cs = torch.stack([torch.linalg.inv(self.ref_c2ws[i]) for i in range(32)], axis = 0)
        #print(w2cs.shape)
        sim_cam_space_coords = transform_pointcloud_torch(ref_3d_coords[:,1:], w2cs) # 32, N, 3
        print(sim_cam_space_coords.shape)
        sim_img_coords = project_points(sim_cam_space_coords, self.ref_intrinsics) # 32, N, 3
        
        for i in range(32):
            full_img = np.zeros((360,1280,3))
            test_img = images[0, :3].permute(1,2,0).cpu().numpy()
            H, W, C = test_img.shape
            full_img[:H, :W] = test_img
            #print(coord)
            full_img[coord[1]-1:coord[1]+1, coord[2]-1:coord[2]+1] = np.array([0,0,255])
            heat_map = np.zeros((180,320,3))
            sim_coords = sim_img_coords[i].astype(int) # N, 2
            for j in range(sim_coords.shape[0]):
                heat_map[sim_coords[j,1], sim_coords[j,0],2] += sims[j]
            heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
            heat_map *= 255.0
            heat_map = cv2.resize(heat_map, (640,360), interpolation=cv2.INTER_LINEAR)
            ref_img = self.ref_images[i,4:5].repeat(3,1,1).permute(1,2,0).cpu().numpy()
            ref_img = cv2.resize(ref_img, (640,360), interpolation=cv2.INTER_LINEAR)
            ref_img = 1 - ((ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img)))
            ref_img *= 255.0
            full_img[:, W:] = np.clip(heat_map + ref_img, 0, 255)
            cv2.imwrite("./match_vis/sim_pts_%.2d.png"%i, full_img)
            
    
    def vis_sim_field(self, sim_field, test_2d_coords, images):
        # sim_field N_model_pts, N_2d_pts
        # test_2d_coords N_2d_pts, 3
        # images B, 5, H, W
        pt_id = 45
        sims = sim_field[:,pt_id] # shape N_model_pts
        w2cs = torch.stack([torch.linalg.inv(self.ref_c2ws[i]) for i in range(32)], axis = 0)
        #print(w2cs.shape)
        sim_cam_space_coords = transform_pointcloud_torch(self.model_pc, w2cs) # 32, N, 3
        print(sim_cam_space_coords.shape)
        sim_img_coords = project_points(sim_cam_space_coords, self.ref_intrinsics)
        #print(sim_img_coords.shape)
        #print(sim_img_coords[:10])
        coord = test_2d_coords[pt_id]
        for i in range(32):
            full_img = np.zeros((360,1280,3))
            test_img = images[0, :3].permute(1,2,0).cpu().numpy()
            test_img[coord[1]-1:coord[1]+1, coord[2]-1:coord[2]+1] = np.array([0,0,255])
            test_img = cv2.resize(test_img, (640,360), interpolation=cv2.INTER_LINEAR)
            H, W, C = test_img.shape
            full_img[:H, :W] = test_img
            #print(coord)
            #full_img[coord[1]-1:coord[1]+1, coord[2]-1:coord[2]+1] = np.array([0,0,255])
            heat_map = np.zeros((180,320,3))
            sim_coords = sim_img_coords[i].astype(int) # N, 2
            for j in range(sim_coords.shape[0]):
                heat_map[sim_coords[j,1], sim_coords[j,0],2] += sims[j]
            heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
            heat_map *= 255.0
            heat_map = cv2.resize(heat_map, (640,360), interpolation=cv2.INTER_LINEAR)
            ref_img = self.ref_images[i,4:5].repeat(3,1,1).permute(1,2,0).cpu().numpy()
            ref_img = cv2.resize(ref_img, (640,360), interpolation=cv2.INTER_LINEAR)
            ref_img = 1 - ((ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img)))
            ref_img *= 255.0
            full_img[:, W:] = np.clip(heat_map + ref_img, 0, 255)
            cv2.imwrite("./match_vis/sim_field_%.2d.png"%i, full_img)
            
              
    def vis_rgbs(self, rgbs):
        rgbs = rgbs.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(rgbs.shape[0]):
            rgb = rgbs[i]
            print("Visualizing image", str(i), rgb.shape, np.max(rgb), np.min(rgb), np.mean(rgb))
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
            rgb *= 255
            cv2.imwrite("./match_vis/cropped_rgb_%.2d.png" % i, rgb)

    def gen_and_save_refs_bags(self,setting_name=None,lowrank_str=None):
        N_refs,feat_C,feat_H,feat_W = self.ref_features.shape
        ref_features = self.ref_features.permute(0,2,3,1)       # nref,32,32,1024
        ref_features = ref_features[self.feat_masks[:,0]>0]     # n_ref_pts(26326),1024

        ref_idxs = create_3dmeshgrid(N_refs, feat_H, feat_W,self.device)
        ref_idxs = ref_idxs[self.feat_masks[:,0]>0]     # 26326,3
        ref_img_ids = ref_idxs[:,0]                     # 26326
        n_ref_pts = ref_img_ids.shape[0]

        kmeans = KMeans(n_clusters=2048).fit(ref_features.cpu())

        centers = kmeans.cluster_centers_   # 2048,1024

        ref_bags = torch.zeros(N_refs,2048).to(self.device)     # 64,2048
        
        centers_tensor = torch.tensor(centers,device=self.device)

        for i in tqdm(range(n_ref_pts)):  # TODO how to not use forloop
            descriptors_centers_dis = pairwise_euclidean_distance(ref_features[i:i+1], centers_tensor)# 26326,2048
            sorted_dis,sorted_indices = torch.sort(descriptors_centers_dis, dim=1)# both 26326,2048
            for j in range(3):
                indice = sorted_indices[0,j]
                dis = torch.exp(-(sorted_dis[0,j]**2)/200)
                ref_bags[ref_img_ids[i],indice] += dis

        ref_bags = np.array(ref_bags.cpu())

        np.save(self.ref_dir + f'/{setting_name}_ref_bags{lowrank_str}.npy',ref_bags)
        np.save(self.ref_dir + f'/{setting_name}_vision_word_list{lowrank_str}.npy',centers)

        return

    def select_refs(self,features,batch_feat_mask,b,test_idxs, n,gt_pose=None):
        # B*N,C
        features = features.reshape(32,32,-1)[batch_feat_mask[0,0]>0] # n_test_2d_pts(381), C
        descriptors_centers_dis = pairwise_euclidean_distance(features, torch.tensor(self.vision_word_list, device=self.device)) # 381，2048
        sorted_dis, sorted_indices = torch.sort(descriptors_centers_dis, dim=1)
        test_bags = torch.zeros(2048, device = self.device)
        test_bags_2 = torch.zeros(2048, device = self.device)
        n_test_pts = features.shape[0]

        test_img_ids = test_idxs[:,0]

        # Method 1
        '''
        for i in range(n_test_pts):
            for j in range(3):
                indice = sorted_indices[i,j]
                dis = torch.exp(-(sorted_dis[i,j]**2)/200)
                test_bags[indice] += dis
        '''
        for j in range(3):
            indices = sorted_indices[:,j]
            dis = torch.exp(-(sorted_dis[:,j]**2)/200)
            #for i in range(n_test_pts):
                #test_bags_2[indices[i]] += dis[i]
            test_bags.index_add_(0, indices, dis)
        test_bags = test_bags.unsqueeze(0)
        '''
        # Method 2    
        for j in range(3):
            indices = sorted_indices[:,j]
            #print(indices)
            dis = torch.exp(-(sorted_dis[:,j]**2)/200)
            #test_bags_2[indices] += dis
            test_bags_2.scatter_add(0, indices, dis)
        '''
        #print(test_bags[:20])
        #print(test_bags_2[:20])

        bag_cos_sim = pairwise_cosine_similarity(test_bags,torch.tensor(self.ref_bags,device=self.device))   # b,v
        _,sorted_view_indices = torch.sort(bag_cos_sim,dim=1,descending=True)

        # if gt_pose is not None:
        #     r_errors = []
        #     t_errors = []
        #     gt_pose = gt_pose
        #     ref_poses = self.ref_c2ws.detach().cpu().numpy()
        #
        #     for i in range(ref_poses.shape[0]):
        #         R1 = gt_pose[:3, :3] / np.cbrt(np.linalg.det(gt_pose[:3, :3]))
        #         R2 = ref_poses[i, :3, :3] / np.cbrt(np.linalg.det(ref_poses[i, :3, :3]))
        #         R = R1 @ R2.transpose()
        #         theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        #         r_errors.append(theta)
        #     r_errors = np.stack(r_errors, axis=0)
        #     return np.argsort(r_errors)[0]

        return sorted_view_indices[:,:n]

