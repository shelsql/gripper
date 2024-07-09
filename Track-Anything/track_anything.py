import json
import sys

import PIL
import imageio.plugins.opencv
from tqdm import tqdm
import torchvision.transforms as transforms
from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse
from PIL import Image
sys.path.append('/home/data/tianshuwu/code/gripper')
from memory_pool import init_for_trackany,inference_pose_for_trackany
import cv2
import torch
from utils.spd import read_pointcloud,transform_pointcloud,project_points_float
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from typing import Any, Dict, Optional, Tuple, Union

class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)
        self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)

    # def inference_step(self, first_flag: bool, interact_flag: bool, image: np.ndarray, 
    #                    same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     if first_flag:
    #         mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
    #         return mask, logit, painted_image
        
    #     if interact_flag:
    #         mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #         return mask, logit, painted_image
        
    #     mask, logit, painted_image = self.xmem.track(image, logit)
    #     return mask, logit, painted_image
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image
    
    # def interact(self, image: np.ndarray, same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #     return mask, logit, painted_image

    def generator(self, images: list, template_mask:np.ndarray):
        
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i ==0:           
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                
            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images

    def generator_pose_prediction(self,template_mask:np.ndarray):
        parser = argparse.ArgumentParser()
        # 不用动的参数
        parser.add_argument('--B', type=int, default=1, help='batch size')
        parser.add_argument('--S', type=int, default=1, help='sequence length')
        parser.add_argument('--shuffle', default=False, action='store_true')
        parser.add_argument('--test_dir', type=str, default='/home/data/tianshuwu/data/Ty_data')
        parser.add_argument('--dtype', default=torch.float32)
        parser.add_argument('--record_vis', default=False, action='store_true')
        parser.add_argument('--use_full_depth', default=False, action='store_true')  # 这个c++没写,一直关掉
        parser.add_argument('--noise', default='none', action='store_true')  # none random incremental
        parser.add_argument('--adjust', default=False, action='store_true')  # 低精度模式下需要打开，但是有问题
        parser.add_argument('--rela_mode', default='gt', action='store_true',
                            help='gt or pr')  # 高精度模式：gt #低精度模式下：pr，但是有问题
        parser.add_argument('--use_cpp', default=True, action='store_true')
        parser.add_argument('--single_opt', default=False, action='store_true')  # 不用这个
        parser.add_argument('--pnp_only', default=False, action='store_true')
        parser.add_argument('--device', type=str, default='cuda:0')
        parser.add_argument('--uni3d_color', default=False, action='store_true')  # 没效果，不用了
        # 可能会动的参数
        parser.add_argument('--use_depth', default=True, action='store_true')
        parser.add_argument('--ref_dir', type=str, default='/home/data/tianshuwu/data/ref_960')
        parser.add_argument('--max_iter', type=int, default=30)
        parser.add_argument('--refine_mode', type=str, default='d')
        parser.add_argument('--max_number', type=int, default=32)
        parser.add_argument('--key_number', type=int, default=8)
        parser.add_argument('--view_number', type=int, default=5)
        parser.add_argument('--gripper', type=str, default="heph_new")
        parser.add_argument('--init', default='rela')  # rela 或者 pnp，多帧优化时rela，单帧优化时pnp，同时要把refine mode改为a
        parser.add_argument('--result_fold_name', default='tmp')
        parser.add_argument('--test_data_type', default='sim', help='sim or real')

        # 只用dino时，要用together
        # 目前阶段，不动就可以
        parser.add_argument('--pca_type', default='together', help='together or respective or nope')
        parser.add_argument('--dino_layer', type=int, default=19)
        parser.add_argument('--uni3d_layer', type=int, default=-1)
        cfg = parser.parse_args()



        memory_pool,matcher = init_for_trackany(cfg)
        masks = []
        logits = []
        painted_images = []
        camera_intrinsic_path = '/home/data/tianshuwu/data/data_sample_2/camera_intrinsics.json'
        camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
        camera_intrinsic = camera_intrinsic['intrinsics']
        # camera_intrinsic_path = '/home/data/tianshuwu/data/Ty_data/12_D415_right_1/_camera_settings.json'
        # camera_intrinsic = json.loads(open(camera_intrinsic_path).read())
        # camera_intrinsic = camera_intrinsic['camera_settings'][0]['intrinsic_settings']

        for i in tqdm(range(251)):
            rgb, depth, pose = self.get_data_sample(i)
            if i == 0:
                mask, logit, painted_image = self.xmem.track(rgb, template_mask)
                masks.append(mask)
                logits.append(logit)
                # painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(rgb)
                masks.append(mask)
                logits.append(logit)
                # painted_images.append(painted_image)
            # cv2.imwrite(f'/home/data/tianshuwu/data/data_sample_2/data_{i}/mask.png',mask)
            frame = {
                'rgb': torch.tensor(rgb).unsqueeze(0).unsqueeze(0),
                'depth': torch.tensor(depth).unsqueeze(0).unsqueeze(0),
                'mask': torch.tensor(mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1),
                'c2w':  torch.tensor(np.linalg.inv(pose)).unsqueeze(0).unsqueeze(0),
                'obj_pose': torch.tensor(np.eye(4)).unsqueeze(0).unsqueeze(0),
                'feat':None,
                'intrinsics': camera_intrinsic
            }
            pred_pose = inference_pose_for_trackany(memory_pool,matcher,frame,cfg)

            # 这个可视化
            vis_pred_image = self.vis_pose(rgb, pred_pose)                      # zhushi
            vis = np.concatenate([painted_image,vis_pred_image],axis=1)     # zhushi


            painted_images.append(vis)

        result_name = f'{cfg.gripper}'
        memory_pool.eliminate_all_frames_and_compute_result(result_name)

        return masks, logits, painted_images

    def vis_pose(self,rgb,pred_pose):
        point_cloud = read_pointcloud('/home/data/tianshuwu/code/gripper/heph_2048.txt')
        np_image = rgb

        pred_cloud = transform_pointcloud(point_cloud, pred_pose)

        # sim dataset时，相机内参
        camera_intrinsic = {'cx': 327.649, 'cy': 236.966, 'fx': 607.164, 'fy': 606.177}
        h, w = 480, 640  # 360,640

        pred_proj_coords = project_points_float(pred_cloud, camera_intrinsic).astype(np.int16)
        render_image = np.zeros((h, w), dtype=np.uint8)
        render_image[np.clip(pred_proj_coords[:, 1], 0, h - 1), np.clip(pred_proj_coords[:, 0], 0, w - 1)] = 1
        render_image = render_image > 0
        # 膨胀操作
        dilated_image = binary_dilation(render_image, iterations=5)
        # 填充孔洞
        filled_image = binary_fill_holes(dilated_image)
        # 如果需要，可以再进行侵蚀操作以恢复原来的形状
        render_image = binary_erosion(filled_image, iterations=5)
        ndimage.binary_fill_holes(render_image)
        img_contour = make_contour_overlay(np_image, render_image[:, :, np.newaxis], (255, 0, 0), 0)['img']
        return img_contour

    def get_data_sample(self,data_idx):
        # data_idx = str(data_idx).rjust(3,'0')
        rgb_path = f'/home/data/tianshuwu/data/sample_data_3/data_{data_idx}/rgb.png'
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth_path = f'/home/data/tianshuwu/data/sample_data_3/data_{data_idx}/depth_align.png'
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, np.newaxis]
        depth = depth / 1000.0
        pose = np.load(f'/home/data/tianshuwu/data/sample_data_3/data_{data_idx}/pose.npy')


        return rgb, depth, pose

    def get_data_panda(self,data_idx,seq_path='/home/data/tianshuwu/data/Ty_data/6_D415_left_1'):
        data_idx = str(data_idx).rjust(6,'0')
        rgb_path = f'{seq_path}/{data_idx}.png'
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth_path = f'{seq_path}/{data_idx}.exr'
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, np.newaxis]
        depth = depth / 1000.0

        kpts_path = f'{seq_path}/{data_idx}.json'
        kpts = json.loads(open(kpts_path).read())
        gripper_info = kpts[0]['keypoints'][8]
        gripper_t = torch.tensor(gripper_info["location_wrt_cam"]).numpy()
        gripper_r = torch.tensor(gripper_info["R2C_mat"]).numpy()
        gripper_rt = np.zeros((4, 4))
        gripper_rt[:3, :3] = gripper_r
        gripper_rt[:3, 3] = gripper_t
        gripper_rt[3, 3] = 1  # obj2cam
        c2w = np.linalg.inv(gripper_rt)


        return rgb, depth, np.linalg.inv(c2w)




def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=6080, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 

def make_contour_overlay(
    img: np.ndarray,
    render: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None,
    dilate_iterations: int = 1,
) -> Dict[str, Any]:

    if color is None:
        color = (0, 255, 0)

    mask_bool = get_mask_from_rgb(render)
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)[:, :, None]
    mask_rgb = np.concatenate((mask_uint8, mask_uint8, mask_uint8), axis=-1)

    # maybe dilate this a bit to make edges thicker
    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)

    # dilate
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)

    img_contour = np.copy(img)
    img_contour[canny > 0] = color

    return {
        "img": img_contour,
        "mask": mask_bool,
        "canny": canny,
    }

def get_mask_from_rgb(img: np.ndarray) -> np.ndarray:
    img_t = torch.as_tensor(img)
    mask = torch.zeros_like(img_t)
    mask[img_t > 0] = 255
    mask = torch.max(mask, dim=-1)[0]
    mask_np = mask.numpy().astype(np.bool_)
    return mask_np

if __name__ == "__main__":
    masks = None
    logits = None
    painted_images = None
    images = []
    image  = np.array(PIL.Image.open('/hhd3/gaoshang/truck.jpg'))
    args = parse_augment()
    # images.append(np.ones((20,20,3)).astype('uint8'))
    # images.append(np.ones((20,20,3)).astype('uint8'))
    images.append(image)
    images.append(image)

    mask = np.zeros_like(image)[:,:,0]
    mask[0,0]= 1
    trackany = TrackingAnything('/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth','/ssd1/gaomingqi/checkpoints/XMem-s012.pth', args)
    masks, logits ,painted_images= trackany.generator(images, mask)
        
        
    
    
    