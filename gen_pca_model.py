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


class ReferenceDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/code/gripper/rendered_franka",
                 num_views=64,
                 features=23,
                 pca=None,
                 ):
        super().__init__()
        print("Loading reference view dataset...")
        self.N = num_views
        self.features = features
        self.dataset_location = dataset_location
        self.rgb_paths = glob.glob(dataset_location + "/*png")
        self.camera_intrinsic_path = dataset_location + "/camera_intrinsics.json"
        self.pca = pca
        print("Found %d views in %s" % (len(self.rgb_paths), self.dataset_location))
        self.size = 448
        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
        self.device = 'cuda:0'
        self.pca = PCA(n_components=256)

    def __getitem__(self, index):

        rgbs = []
        depths = []
        masks = []
        c2ws = []
        obj_poses = []
        if self.features > 0:
            feats = []
            # feats = None
        camera_intrinsic = json.loads(open(self.camera_intrinsic_path).read())
        ref_features_list = []
        for idx,glob_rgb_path in enumerate(self.rgb_paths):
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
            if self.features > 0:
                feat_path = path + "_feats_%.2d.npy" % self.features
                feat = np.load(feat_path)
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
                if self.features > 0:
                    feats = np.stack(feats, axis=0)  # 840,1024,32,32
                else:
                    feats = None
                # print(depths.shape)

                ref_rgbs = torch.Tensor(rgbs).unsqueeze(0).float().permute(0, 1, 4, 2, 3).squeeze() # B, S, C, H, W
                num_refs, C, H, W = ref_rgbs.shape
                ref_depths = torch.Tensor(depths).unsqueeze(0).float().permute(0, 1, 4, 2, 3).reshape(num_refs, 1, H, W)
                ref_masks = torch.Tensor(masks).unsqueeze(0).float().permute(0, 1, 4, 2, 3).reshape(num_refs, 1, H, W)
                ref_images = torch.concat([ref_rgbs, ref_depths, ref_masks], dim = 1).to(self.device)
                ref_features = torch.tensor(feats).float().to(self.device)

                cropped_ref_rgbs, cropped_ref_masks, ref_bboxes = self.prepare_images(ref_images)
                feat_masks = F.interpolate(cropped_ref_masks, size=(32, 32), mode="nearest")

                N_refs, feat_C, feat_H, feat_W = ref_features.shape
                ref_features = ref_features.permute(0, 2, 3, 1)  # nref,32,32,1024
                ref_features = ref_features[feat_masks[:, 0] > 0]  # n_ref_pts(26326),1024
                ref_idxs = create_3dmeshgrid(N_refs, feat_H, feat_W, self.device)
                ref_idxs = ref_idxs[feat_masks[:, 0] > 0]  # 26326,3
                ref_img_ids = ref_idxs[:, 0]  # 26326
                ref_features_list.append(ref_features)

                # self.pca.partial_fit(ref_features.cpu().numpy())

                rgbs = []
                depths = []
                masks = []
                c2ws = []
                obj_poses = []
                if self.features > 0:
                    feats = []
        ref_features_list = torch.cat(ref_features_list).cpu().numpy()
        self.pca.fit(ref_features_list)
        joblib.dump(self.pca,f'{self.dataset_location}/pca_model.joblib')


        sample = {
                "rgbs": rgbs,
                "depths": depths,
                "masks": masks,
                "c2ws": c2ws,
                "obj_poses": obj_poses,
                "feats": feats,
                "intrinsics": camera_intrinsic
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
        return cropped_rgbs, cropped_masks, bboxes

if __name__ == '__main__':

    ref_dir = f"/root/autodl-tmp/shiqian/code/render/reference_more/panda"
    # test_dataset = SimTrackDataset(dataset_location=test_dir, seqlen=S, features=feat_layer)
    # test_dataset = TrackingDataset(dataset_location=cfg.test_dir, seqlen=cfg.S,features=cfg.feat_layer)
    ref_dataset = ReferenceDataset(dataset_location=ref_dir, num_views=840, features=19,pca=None)

    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.B, shuffle=cfg.shuffle)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=False)

    # iterloader = iter(test_dataloader)
    # Load ref images and init Dinov2 Matcher
    refs = next(iter(ref_dataloader))