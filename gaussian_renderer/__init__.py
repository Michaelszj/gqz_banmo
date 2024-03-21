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
from utils.general_utils import matrix2quaternion, build_covariance_auto, build_covariance_lbs
from scene.cameras import Camera
import torch.nn.functional as F
import pytorch3d
import pytorch3d.transforms as T

def render_points(cam:Camera, points, color, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(points[:,:3], dtype=points.dtype, requires_grad=True, device="cuda") + 0
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
        sh_degree=0,
        campos=viewpoint_camera_.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    

    means3D = points[:,:3]
    means2D = screenspace_points
    opacity = torch.ones_like(points[:,0])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = points[:,7:]
    rotations = points[:,3:7]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = color


    # import pdb;pdb.set_trace()
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    return {"render": rendered_image,
            "mask": rendered_alpha,  
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

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
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    

    means3D = pc.get_xyz
    means2D = screenspace_points
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
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    return {"render": rendered_image,
            "mask": rendered_alpha,  
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_auto(cam : Camera, xyz, features, 
                pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0,color_replace=None,
                rot_delta=None,append_points=None,append_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    shape = xyz.shape
    if append_points is not None:
        xyz = torch.cat([xyz,append_points[:,:3]],dim=0)
        # features = torch.cat([features,append_color],dim=0)
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
    viewmatrix=(cam.world_view_transform)
    means3D = xyz@viewmatrix[:3,:3]+viewmatrix[3,:3]
    means2D = screenspace_points
    scales = pc.get_scaling
    rotations = pc.get_rotation
    opacity = pc.get_opacity
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    if rot_delta is None:
        cov3D_precomp = build_covariance_auto(scales,rotations,viewmatrix[:3,:3].T)
    else:
        cov3D_precomp = build_covariance_lbs(scales,rotations,viewmatrix[:3,:3].T,rot_delta)
    
    if append_points is not None:
        # scales = torch.cat([scales,append_points[:,7:]],dim=0)
        # rotations = torch.cat([rotations,append_points[:,3:7]],dim=0)
        opacity = opacity/10.
        opacity = torch.cat([opacity,torch.ones_like(append_points[:,:1],dtype=opacity.dtype).cuda()],dim=0) 
        cov3D_append = build_covariance_auto(append_points[:,7:],append_points[:,3:7],viewmatrix[:3,:3].T)
        cov3D_precomp = torch.cat([cov3D_precomp,cov3D_append],dim=0)
    scales = None
    rotations = None
    shs = pc.get_features
    colors_precomp = None
    # import pdb;pdb.set_trace()
    if color_replace is not None:
        shs = None
        colors_precomp =color_replace
    if append_points is not None:
        shs = None
        colors_precomp = torch.cat([colors_precomp,append_color],dim=0)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,  
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
    
