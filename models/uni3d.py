import torch
import timm
import numpy as np
from torch import nn
# import losses

from models.point_encoder import PointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc,layer):
        xyz = pc[:,:,:3].contiguous()
        color = pc[:,:,3:].contiguous()
        pc_feat = self.point_encoder(xyz, color,layer=layer)
        return pc_feat

    def forward(self, pc, text, image):
        text_embed_all = text
        image_embed = image   
        pc_embed = self.encode_pc(pc)
        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}

# def get_filter_loss(args):
#     return losses.Uni3d_Text_Image_Loss()

def get_metric_names(model):
    return ['loss', 'uni3d_loss', 'pc_image_acc', 'pc_text_acc']

def create_uni3d():
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model("eva02_large_patch14_448", checkpoint_path='', drop_path_rate=0.0)

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder,)
    return model


if __name__ == '__main__':
    model = create_uni3d().to('cuda:0')
    random_point = torch.randn((4,4096,6),device='cuda:0')
    feat = model.encode_pc(random_point)
    feat = feat/feat.norm(dim=-1, keepdim=True)

    print(6)