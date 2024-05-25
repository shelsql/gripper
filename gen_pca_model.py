import sys
import os
import json
import glob
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import joblib
from utils.spd import get_2dbboxes, create_3dmeshgrid
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse

class PCALowrank():
    def __init__(self):
        self.V = None
        self.mean = None
        # self.std = None

    def fit(self,X,q=256):
        X_mean = torch.mean(X,dim=0)
        # X_std = torch.std(X,dim=0)
        X_centered = (X - X_mean)# / X_std
        U,S,V = torch.pca_lowrank(X_centered,q=q)
        self.V = V
        self.mean = X_mean
        # self.std = X_std

    def transform(self,X):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        else:
            assert len(X.shape) == 2
        X_centered = (X - self.mean)# / self.std
        return torch.matmul(X_centered,self.V)

    # def normalize_only(self,X):
    #     return (X - self.mean) / self.std


class ReferenceDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/code/gripper/rendered_franka",
                 dino_name=None,
                 uni3d_name=None
                 ):
        super().__init__()
        print("Loading reference view dataset...")
        # self.N = num_views
        self.dataset_location = dataset_location
        self.rgb_paths = glob.glob(dataset_location + "/*png")
        self.camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
        print("Found %d views in %s" % (len(self.rgb_paths), self.dataset_location))
        self.size = 448
        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
        self.device = 'cuda:0'
        if cfg.pca_type == 'torch':
            self.pca = PCALowrank()
        elif cfg.pca_type == 'sklearn':
            self.pca = PCA(n_components=256)
        self.dino_name = dino_name
        self.uni3d_name = uni3d_name

    def __getitem__(self, index):

        rgbs = []
        depths = []
        masks = []
        c2ws = []
        obj_poses = []
        feats = []
            # feats = None
        camera_intrinsic = json.loads(open(self.camera_intrinsic_path).read())
        ref_features_list = []
        print('loading data...')
        for idx,glob_rgb_path in tqdm(enumerate(self.rgb_paths)):
            path = glob_rgb_path[:-8]

            rgb_path = path + "_rgb.png"
            depth_path = path + "_depth1.exr"
            mask_path = path + "_id1.exr"
            c2w_path = path + "_c2w.npy"
            obj_pose_path = path + "_objpose.npy"

            # print(rgb_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:1]
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:1]
            c2w = np.load(c2w_path)
            obj_pose = np.load(obj_pose_path)

            if self.dino_name is not None:
                dino_path = path + f"_feats_{self.dino_name}.npy"
                dino_feat = np.load(dino_path)
            if self.uni3d_name is not None:
                uni3d_path = path + f"_feats_{self.uni3d_name}.npy"
                uni3d_feat = np.load(uni3d_path)
            if (self.dino_name is not None) and (self.uni3d_name is not None):
                feat = np.concatenate([dino_feat,uni3d_feat],axis=0)
            elif self.dino_name is None:
                feat = uni3d_feat
            elif self.uni3d_name is None:
                feat = dino_feat

            feats.append(feat)

            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
            c2ws.append(c2w)
            obj_poses.append(obj_pose)

            if (idx+1)%650 == 0 or idx==len(self.rgb_paths)-1:

                rgbs = np.stack(rgbs, axis=0)
                depths = np.stack(depths, axis=0)
                masks = np.stack(masks, axis=0)
                c2ws = np.stack(c2ws, axis=0)
                obj_poses = np.stack(obj_poses, axis=0)

                feats = np.stack(feats, axis=0)  # 840,1024,32,32

                # print(depths.shape)

                ref_rgbs = torch.Tensor(rgbs).unsqueeze(0).float().permute(0, 1, 4, 2, 3).squeeze() # B, S, C, H, W
                num_refs, C, H, W = ref_rgbs.shape
                ref_depths = torch.Tensor(depths).unsqueeze(0).float().permute(0, 1, 4, 2, 3).reshape(num_refs, 1, H, W)
                ref_masks = torch.Tensor(masks).unsqueeze(0).float().permute(0, 1, 4, 2, 3).reshape(num_refs, 1, H, W)
                ref_images = torch.concat([ref_rgbs, ref_depths, ref_masks], dim = 1).to(self.device)
                ref_features = torch.tensor(feats).float().to(self.device)

                cropped_ref_masks = self.prepare_images(ref_images)
                feat_masks = F.interpolate(cropped_ref_masks, size=(32, 32), mode="nearest")

                N_refs, feat_C, feat_H, feat_W = ref_features.shape
                ref_features = ref_features.permute(0, 2, 3, 1)  # nref,32,32,1024
                ref_features = ref_features[feat_masks[:, 0] > 0]  # n_ref_pts(26326),1024
                ref_idxs = create_3dmeshgrid(N_refs, feat_H, feat_W, self.device)
                ref_idxs = ref_idxs[feat_masks[:, 0] > 0]  # 26326,3
                ref_img_ids = ref_idxs[:, 0]  # 26326
                ref_features_list.append(ref_features)

                rgbs = []
                depths = []
                masks = []
                c2ws = []
                obj_poses = []
                feats = []

                # self.pca.partial_fit(ref_features.cpu().numpy())

        if cfg.pca_type == 'torch':
            ref_features_list = torch.cat(ref_features_list)#.numpy()
        else:
            ref_features_list = torch.cat(ref_features_list).cpu().numpy()
        print('computing pca...')
        self.pca.fit(ref_features_list)
        pca_name = ''
        if self.dino_name is not None:
            pca_name = pca_name + self.dino_name
        if self.uni3d_name is not None:
            pca_name = pca_name + self.uni3d_name


        if cfg.pca_type == 'sklearn':
            joblib.dump(self.pca,f'{self.dataset_location}/{pca_name}_pca_model.joblib')
        elif cfg.pca_type == 'torch':
            np.save(f'{self.dataset_location}/{pca_name}_pca_V',self.pca.V.cpu().numpy())
            np.save(f'{self.dataset_location}/{pca_name}_pca_mean',self.pca.mean.cpu().numpy())
            # np.save(f'{self.dataset_location}/{pca_name}_pca_std',self.pca.std.cpu().numpy())

        for idx,glob_rgb_path in enumerate(self.rgb_paths):
            path = glob_rgb_path[:-8]

            if self.dino_name is not None:
                dino_path = path + f"_feats_{self.dino_name}.npy"
                dino_feat = np.load(dino_path)
            if self.uni3d_name is not None:
                uni3d_path = path + f"_feats_{self.uni3d_name}.npy"
                uni3d_feat = np.load(uni3d_path)
            if (self.dino_name is not None) and (self.uni3d_name is not None):
                feat = np.concatenate([dino_feat,uni3d_feat],axis=0)
            elif self.dino_name is None:
                feat = uni3d_feat
            elif self.uni3d_name is None:
                feat = dino_feat

            if cfg.pca_type == 'sklearn':
                pca_feat = self.pca.transform(feat.reshape(-1,32*32).transpose(1,0)).transpose(1,0).reshape(-1,32,32)
                np.save(path + f"_feats_{pca_name}_pca.npy",pca_feat)
            elif cfg.pca_type == 'torch':
                pca_feat = self.pca.transform(torch.tensor(feat,device=self.device).reshape(-1,32*32).transpose(1,0)).transpose(1,0).reshape(-1,32,32)
                np.save(path + f"_feats_{pca_name}_pca_lowrank.npy",pca_feat.cpu().numpy())



        sample = {
                "rgbs": rgbs
            }

        return sample

    def __len__(self):
        return 1

    def prepare_images(self, images):
        B, C, H, W = images.shape
        rgbs = images[:, 0:3] / 255.0  # B, 3, H, W
        depths = images[:, 3:4]  # B, 1, H, W
        masks = images[:, 4:5]  # B, 1, H, W
        bboxes = get_2dbboxes(masks[:, 0])  # B, 4
        cropped_rgbs = torch.zeros((B, 3, self.size, self.size), device=self.device)
        cropped_masks = torch.zeros((B, 1, self.size, self.size), device=self.device)
        for b in range(B):
            y_min, x_min, y_max, x_max = bboxes[b, 0], bboxes[b, 1], bboxes[b, 2], bboxes[b, 3]

            cropped_rgb = rgbs[b:b + 1, :, y_min:y_max, x_min:x_max]
            cropped_mask = masks[b:b + 1, :, y_min:y_max, x_min:x_max]
            cropped_rgb = F.interpolate(cropped_rgb, size=(self.size, self.size), mode="bilinear")
            cropped_mask = F.interpolate(cropped_mask, size=(self.size, self.size), mode="nearest")
            cropped_rgbs[b:b + 1] = cropped_rgb
            cropped_masks[b:b + 1] = cropped_mask
        cropped_rgbs = self.transform(cropped_rgbs)
        cropped_rgbs = cropped_rgbs.to(self.device)
        bboxes = torch.tensor(bboxes, device=self.device)
        # print(bboxes)
        return  cropped_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gripper_name',default='panda',  help='single gripper_name')
    parser.add_argument('--pca_type',default='torch',help='sklearn or torch')
    parser.add_argument('--dino_layer',default=19,type=int)
    parser.add_argument('--uni3d_layer', default=-1, type=int)
    parser.add_argument('--ref_dir',default=f"/home/data/tianshuwu/data/ref_840")
    cfg = parser.parse_args()
    # gripper_name = ['robotiq2f140','robotiq2f85','robotiq3f','shadowhand','kinova','panda',]

    gripper = cfg.gripper_name

    dino_name = f'dino{cfg.dino_layer}' if cfg.dino_layer>0 else None
    uni3d_name = f'uni3d{cfg.uni3d_layer}_nocolor' if cfg.uni3d_layer>0 else None
    ref_dir = f'{cfg.ref_dir}/{cfg.gripper_name}'
    # test_dataset = SimTrackDataset(dataset_location=test_dir, seqlen=S, features=feat_layer)
    # test_dataset = TrackingDataset(dataset_location=cfg.test_dir, seqlen=cfg.S,features=cfg.feat_layer)
    ref_dataset = ReferenceDataset(dataset_location=ref_dir, dino_name=dino_name,uni3d_name=uni3d_name)

    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.B, shuffle=cfg.shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=False)

    # iterloader = iter(test_dataloader)
    # Load ref images and init Dinov2 Matcher
    refs = next(iter(ref_dataloader))