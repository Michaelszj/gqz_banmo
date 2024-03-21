#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View, getWorld2View2, getWorld2View3, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R # .detach().cpu().numpy()
        self.T = T # .detach().cpu().numpy()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.FoVx = np.deg2rad(49.1)
        self.FoVy = np.deg2rad(49.1)
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} philed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = width
        self.image_height = height

        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # import pdb;pdb.set_trace()
        # from pdb import set_trace; set_trace()
        # self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda()
        self.world_view_transform = getWorld2View3(self.R, self.T).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(torch.float32)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        
def rtk_from_angles(delta, phi, resolution=256, radius=1.):
    rtk_novel = torch.zeros([4,4],dtype=torch.float32).cuda()
    Rmat = torch.tensor([[-np.sin(delta),0.,-np.cos(delta)],
                            [-np.cos(delta)*np.sin(phi),-np.cos(phi),np.sin(delta)*np.sin(phi)],
                            [-np.cos(delta)*np.cos(phi),np.sin(phi),np.sin(delta)*np.cos(phi)]])
    Tmat = torch.tensor([0.,0.,1.])*radius
    r = float(resolution)
    rtk_novel[3] = torch.tensor([r,r,r/2,r/2]).cuda()
    rtk_novel[:3,:3] = Rmat
    rtk_novel[:3,3]  = Tmat
    return rtk_novel
    
def angles_from_rtk(rtk):
    Rmat = rtk[:3,:3]
    z = rtk[2]
    phi = torch.arcsin(z[1])
    cos_phi = torch.cos(phi)
    theta = torch.arccos(-z[0]/cos_phi)
    if z[2]<0:
        theta = -theta
    return theta.detach().cpu().numpy(), phi.detach().cpu().numpy()

