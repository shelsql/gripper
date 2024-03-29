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

from utils.spd import get_2dbboxes, create_3dmeshgrid, transform_batch_pointcloud_torch
from utils.spd import save_pointcloud, transform_pointcloud_torch, project_points
# from utils.spd import calc_masked_batch_var, calc_coords_3d_var

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

class Visualizer:
    def __init__(
        self,
        logdir
    ):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
    
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
            cv2.imwrite("./match_vis/match3d_%.2d.png" % step, full_img)
            
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
