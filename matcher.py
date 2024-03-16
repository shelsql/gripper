from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from utils.spd import get_2dbboxes, create_3dmeshgrid, transform_batch_pointcloud_torch

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class Dinov2Matcher:
    def __init__(self, refs, model_pointcloud,
                 repo_name="facebookresearch/dinov2",
                 model_name="dinov2_vitl14_reg",
                 size=448,
                 half_precision=False,
                 threshold=0.75,
                 upscale_ratio=1,
                 device="cuda:3"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.size = size
        self.half_precision = half_precision
        self.threshold = threshold
        self.upscale_ratio = upscale_ratio
        self.device = device
        
        ref_rgbs = torch.Tensor(refs['rgbs']).float().permute(0, 1, 4, 2, 3).squeeze() # B, S, C, H, W
        ref_depths = torch.Tensor(refs['depths']).float().permute(0, 1, 4, 2, 3).squeeze()
        ref_masks = torch.Tensor(refs['masks']).float().permute(0, 1, 4, 2, 3).squeeze()
        c2ws = torch.Tensor(refs['c2ws'][0]).float()
        
        ref_rgbs = torch.flip(ref_rgbs, dims = [2])
        ref_depths = torch.flip(ref_depths, dims = [2])
        ref_masks = torch.flip(ref_masks, dims = [2])
        
        flip_x = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
        flip_x = flip_x.unsqueeze(0).repeat(32, 1, 1).float()
        c2ws = torch.bmm(c2ws, flip_x)
        
        ref_rgbs = F.interpolate(ref_rgbs, scale_factor=0.15, mode="bilinear")
        ref_depths = F.interpolate(ref_depths, scale_factor=0.15, mode="bilinear")
        ref_masks = F.interpolate(ref_masks, scale_factor=0.15, mode="nearest")
        
        ref_images = torch.concat([ref_rgbs, ref_depths[:,0:1], ref_masks[:,0:1]], axis = 1).to(device)
    
        self.ref_c2ws = c2ws.to(device)
        self.model_pc = model_pointcloud
        self.ref_images = ref_images
        self.ref_intrinsics = refs['intrinsics']

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir="../dinov2",source="local", model=model_name, pretrained=False)
            self.model.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
        else:
            self.model = torch.hub.load(repo_or_dir="../dinov2",source="local", model=model_name, pretrained=False)
            self.model.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
        
        print("Calculating reference view features...")
        
        cropped_ref_rgbs, cropped_ref_masks, ref_bboxes = self.prepare_images(ref_images)
        #print(cropped_ref_rgbs.shape)
        self.ref_features = self.extract_features(cropped_ref_rgbs)
        N_refs, feat_C, feat_H, feat_W = self.ref_features.shape
        self.N_refs = N_refs
        assert(feat_H == feat_W)
        feat_size = feat_H
        self.feat_masks = F.interpolate(cropped_ref_masks, size = (feat_size, feat_size), mode = "nearest") # Nref, 1, 32, 32
        self.ref_bboxes = ref_bboxes
        #self.vis_rgbs(cropped_ref_rgbs)
        #self.vis_features(cropped_ref_rgbs, self.feat_masks, self.ref_features)
        # TODO process ref images and calculate features

    # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    def prepare_images(self, images):
        B, C, H, W = images.shape
        rgbs = images[:,0:3] / 255.0 # B, 3, H, W
        depths = images[:,3:4] # B, 1, H, W
        masks = images[:,4:5] # B, 1, H, W
        bboxes = get_2dbboxes(masks[:,0]) # B, 4
        cropped_rgbs = torch.zeros((B, 3, self.size, self.size))
        cropped_masks = torch.zeros((B, 1, self.size, self.size))
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
        bboxes = torch.tensor(bboxes, device = self.device)
        return cropped_rgbs, cropped_masks, bboxes
    
    def extract_features(self, images):
        # images: B, C, H, W  torch.tensor
        #print(images.max(), images.min(), images.mean())
        with torch.inference_mode():
            if self.half_precision:
                image_batch = images.half().to(self.device)
            else:
                image_batch = images.to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0]
            B, N_tokens, C = tokens.shape
            assert(N_tokens == self.size*self.size / 196)
            tokens = tokens.permute(0,2,1).reshape(B, C, self.size//14, self.size//14)
            if self.upscale_ratio != 1:
                tokens = F.interpolate(tokens, scale_factor = self.upscale_ratio)
            #print(tokens.shape)
            #print(tokens.max(), tokens.min(), tokens.mean())
        return tokens # B, C, H, W
    
    def idx_to_2d_coords(self, idxs, feat_size, test_bboxes):
        N, D = idxs.shape # batchno, featidx, refno, featidx
        assert(D == 4)
        coords = torch.zeros((N,6), device = self.device) # batchno, coords, refno, coords
        coords[:,0] = idxs[:,0] # batchno
        coords[:,3] = idxs[:,2] # refno
        test_bboxes = test_bboxes[idxs[:,0]] # N, 4
        ref_bboxes = self.ref_bboxes[idxs[:,2]]
        # Turn token idx into coord within 448*448 box
        coords[:,1] = (((idxs[:,1] // feat_size) * self.model.patch_size) + self.model.patch_size / 2) # Coord within 448*448 box
        coords[:,2] = (((idxs[:,1] % feat_size) * self.model.patch_size) + self.model.patch_size / 2) # Coord within 448*448 box
        coords[:,4] = (((idxs[:,3] // feat_size) * self.model.patch_size) + self.model.patch_size / 2) # Coord within 448*448 box
        coords[:,5] = (((idxs[:,3] % feat_size) * self.model.patch_size) + self.model.patch_size / 2) # Coord within 448*448 box
        # Turn coord within 448*448 box to coord on full image
        coords[:,1] = (coords[:,1] / 448.0 * (test_bboxes[:,2] - test_bboxes[:,0])) + test_bboxes[:,0]
        coords[:,2] = (coords[:,2] / 448.0 * (test_bboxes[:,3] - test_bboxes[:,1])) + test_bboxes[:,1]
        coords[:,4] = (coords[:,4] / 448.0 * (ref_bboxes[:,2] - ref_bboxes[:,0])) + ref_bboxes[:,0]
        coords[:,5] = (coords[:,5] / 448.0 * (ref_bboxes[:,3] - ref_bboxes[:,1])) + ref_bboxes[:,1]
        #TODO finish this
        return coords
    
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
    
    def match_batch(self, images):
        B, C, H, W = images.shape
        N_refs, feat_C, feat_H, feat_W = self.ref_features.shape
        assert(feat_H == feat_W)
        feat_size = feat_H
        #print(images[:,:3].max(),images[:,:3].min(),images[:,:3].mean())
        cropped_rgbs, cropped_masks, bboxes = self.prepare_images(images)
        #self.vis_rgbs(cropped_rgbs)
        #print(cropped_rgbs.max(), cropped_rgbs.min(), cropped_rgbs.mean())
        features = self.extract_features(cropped_rgbs) # B, 1024, 32, 32
        N_tokens = feat_H * feat_W
        #print(features.shape)
        features = features.permute(0, 2, 3, 1).reshape(-1, feat_C) # B*N, C
        ref_features = self.ref_features.permute(0, 2, 3, 1).reshape(-1, feat_C) # 32*N, C
        cosine_sims = pairwise_cosine_similarity(features, ref_features) # B*N, 32*N
        cosine_sims = cosine_sims.reshape(B, N_tokens, N_refs, N_tokens) # B, 1024, 32, 1024
        cosine_sims = cosine_sims.reshape(B, feat_H, feat_W, N_refs, feat_H, feat_W) # B, 32, 32, Nref, 32, 32
        
        batch_feat_masks = F.interpolate(cropped_masks, size=(feat_H, feat_W), mode = "nearest") # B, 1, 32, 32
        #TODO: make this faster
        cosine_sims[batch_feat_masks[:,0] == 0] = 0 # Set 0 for all points not on gripper in test images
        cosine_sims[:,:,:,self.feat_masks[:,0] == 0] # Set 0 for all points not on ref gripper image
        cosine_sims = cosine_sims.reshape(B, N_tokens, N_refs, N_tokens) # B, 1024, 32, 1024
        #print(cosine_sims.shape)
        max_sims, max_inds = torch.max(cosine_sims, axis = 3) # B, 1024, 32
        # Here we want to get rid of all the sims that are below the threshold
        #print(max_sims.shape, max_inds.shape)
        max_coords = create_3dmeshgrid(B, N_tokens, N_refs, device = self.device) # B, 1024, 32, 3
        filtered_inds = max_sims > self.threshold
        max_sims = max_sims[filtered_inds]
        max_inds = max_inds[filtered_inds]
        max_coords = max_coords[filtered_inds]
        #print(max_sims.shape, max_inds.shape, max_coords.shape) # shape is N_matches
        matches_2d_inds = torch.concat([max_coords, max_inds.unsqueeze(1)], axis = 1) # N, 4. Batchnumber test_featid refnumber ref_featid
        matches_2d_coords = self.idx_to_2d_coords(matches_2d_inds, feat_size, bboxes)
        # N, 6   batchno, coords, refno, coords
        #print(matches_2d_coords.shape)
        self.vis_2d_matches(images, matches_2d_coords)
        matches_3d = self.match_2d_to_3d(matches_2d_coords)
        #self.vis_3d_matches(images, matches_3d)
        return matches_3d
    
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
        c2ws = self.ref_c2ws[matches_2d[:, 3].int()]
        print("c2ws:", c2ws.shape)
        world_space_coords = transform_batch_pointcloud_torch(cam_space_coords, c2ws)
        matches_3d = torch.zeros_like(matches_2d)
        matches_3d[:,:3] = matches_2d[:,:3]
        matches_3d[:,3:] = world_space_coords
        return matches_3d
    
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
            
    def vis_3d_matches(self, images, matches_3d, down=2):
        #TODO: Resize images, transform coords, draw lines
        B, C, H, W = images.shape
        assert(C == 5)
        rgb_0 = images[0, :3].permute(1, 2, 0).cpu().numpy() # H, W, 3
        N_ref, C, H_ref, W_ref = self.ref_images.shape
        N, D = matches_3d.shape
        matches_3d = matches_3d[matches_3d[:,0] == 0]
        assert(D == 6)
        camera_K = torch.zeros((3,3), device = self.device)
        camera_K[0,0] = self.ref_intrinsics['fx']
        camera_K[1,1] = self.ref_intrinsics['fy']
        camera_K[0,2] = self.ref_intrinsics['cx']
        camera_K[1,2] = self.ref_intrinsics['cy']
        camera_K[2,2] = 1
        
        for i_ref in tqdm(range(self.N_refs)):
            ref_rgb = self.ref_images[i_ref, :3].permute(1, 2, 0).cpu().numpy() # H, W, 3
            ref_rgb = cv2.resize(ref_rgb, dsize=(640,360),interpolation=cv2.INTER_LINEAR)
            H_ref = 360
            W_ref = 640
            #matches = matches_3d[matches_3d[:,3] == i_ref] # N_match, 6
            coords_3d = matches_3d[:, 3:]
            if coords_3d.shape[0] == 0:
                print("Ref %.2d skipped"% i_ref)
                continue
            c2w = self.ref_c2ws[i_ref]
            coords_3d_homo = torch.concat([coords_3d, torch.ones((coords_3d.shape[0],1), device = coords_3d.device)], axis=1)
            coords_3d_cam = torch.matmul(torch.linalg.inv(c2w), coords_3d_homo.permute(1,0)).permute(1,0)
            coords_3d_cam = coords_3d_cam[:,:3] / coords_3d_cam[:,3:4]
            coords_2d = torch.matmul(camera_K, coords_3d_cam.permute(1,0)).permute(1,0)
            coords_2d = coords_2d[:,:2] / coords_2d[:,2:3]
            full_img = np.zeros((max(H_ref, H), W_ref + W, 3))
            full_img[:H,:W] = rgb_0
            full_img[:H_ref, W:] = ref_rgb
            for i_match in range(matches_3d.shape[0]):
                y_1, x_1 = matches_3d[i_match,1].int().item(), matches_3d[i_match,2].int().item()
                y_2, x_2 = coords_2d[i_match,1].int().item(), coords_2d[i_match,0].int().item()
                y_2 //= 3
                x_2 //= 3
                x_2 += W
                #print(x_1, y_1, x_2, y_2)
                full_img = cv2.line(full_img, (x_1,y_1), (x_2,y_2), (0,0,255), 1)
            cv2.imwrite("./match_vis/match3d_%.2d.png" % i_ref, full_img)
            
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
            
    def vis_rgbs(self, rgbs):
        rgbs = rgbs.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(rgbs.shape[0]):
            rgb = rgbs[i]
            print("Visualizing image", str(i), rgb.shape, np.max(rgb), np.min(rgb), np.mean(rgb))
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
            rgb *= 255
            cv2.imwrite("./match_vis/cropped_rgb_%.2d.png" % i, rgb)