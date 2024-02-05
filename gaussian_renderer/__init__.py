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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import Camera
import torch.nn.functional as F
import pytorch3d
import pytorch3d.transforms as T
def render(cam:Camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    
    # import pdb;pdb.set_trace()
    viewpoint_camera_ = cam

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera_.image_height),
        image_width=int(viewpoint_camera_.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera_.world_view_transform,
        projmatrix=viewpoint_camera_.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera_.camera_center,
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_d = means2D.detach()
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    shs = pc.get_features

    # import pdb;pdb.set_trace()
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_mask, _ = rasterizer(
        means3D = means3D,
        means2D = means2D_d,
        shs = None,
        colors_precomp = torch.ones_like(means3D,device='cuda'),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_mask = rendered_mask[0]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_mask,  
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_auto(cam : Camera, xyz, scales, rotations, features, opacity, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    
    


    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=torch.eye(4,dtype=cam.world_view_transform.dtype).cuda(),
        projmatrix=cam.projection_matrix.to(torch.float32),
        sh_degree=pc.active_sh_degree,
        campos=torch.zeros(3,dtype=cam.world_view_transform.dtype).cuda(),
        prefiltered=False,
        debug=False
    )
    # import pdb;pdb.set_trace()
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    viewmatrix=cam.world_view_transform
    quat = T.matrix_to_quaternion(viewmatrix[:3,:3])
    quat = quat/torch.norm(quat)
    rotations_cam = T.quaternion_multiply(quat,rotations)
    means3D = xyz@viewmatrix[:3,:3]+viewmatrix[3,:3]
    means2D = screenspace_points
    means2D_d = means2D.detach()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    
    cov3D_precomp = None
    shs = features
    colors_precomp = None
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations_cam,
        cov3D_precomp = cov3D_precomp)

    
    rendered_mask, _ = rasterizer(
        means3D = means3D,
        means2D = means2D_d,
        shs = None,
        colors_precomp = torch.ones_like(means3D,device='cuda'),
        opacities = opacity,
        scales = scales,
        rotations = rotations_cam,
        cov3D_precomp = cov3D_precomp)
    rendered_mask = rendered_mask[0]
    # import pdb;pdb.set_trace()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, # (3,h,w)
            "mask": rendered_mask,                   # (h,w)
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
    
