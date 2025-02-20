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
from einops import repeat

import math
from scene.gaussian_model import GaussianModel

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center # (anchor - viewpoint_camera.camera_center)@viewpoint_camera.world_view_transform[:3,:3] 同下
    # ob_view = anchor @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3] # lidar坐标系的dir

    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)
        appearance_rd = pc.get_appearance_rd(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            raydrop = pc.get_raydrop_mlp(torch.cat([cat_local_view, appearance_rd], dim=1)) # 
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
            raydrop = pc.get_raydrop_mlp(torch.cat([cat_local_view_wodist, appearance_rd], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
            raydrop = pc.get_raydrop_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
            raydrop = pc.get_raydrop_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, pc.color_channel-1])
    raydrop = raydrop.reshape([anchor.shape[0]*pc.n_offsets, 1])
    color = torch.cat([color,raydrop],dim=1) # color and raydrop will be concatenated together , as the rasterization' input

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, pc.color_channel, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

from diff_lidargs_rasterization import GaussianRasterizationSettings, GaussianRasterizer
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((xyz.shape[0], 4), dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    depth_max = 80
    depth_min = 0

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_camera.beam_inclinations,  # TODO 输入一个beam
        debug=pipe.debug,
        lidar_far = int(depth_max),
        lidar_near = int(depth_min)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, occ, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "depth":depth,
                "occ":occ,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_image,
                "depth":depth,
                "occ":occ,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    depth_max = 80.0
    depth_min = 0.0

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_camera.beam_inclinations,
        debug=pipe.debug,
        lidar_far = int(depth_max),
        lidar_near = int(depth_min)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0


def renderComposite(viewpoint_cam, background, pipe, valid_model_info, retain_grad=True):
    total_opacity_dict = {}
    total_mask_dict = {}
    total_visable_dict = {}
    total_xyz = None
    total_color = None
    total_opacity = None
    total_scaling = None
    total_rot = None
    total_points_with_model_id = None
    total_screenspace_points = None    

    init = False
    for model_info in valid_model_info:
        data_type = model_info.model_gaussians.get_anchor.dtype
        model_visible_mask = prefilter_voxel(
            model_info.model_view,
            model_info.model_gaussians,
            pipe,
            background
        )
        model_xyz, model_color, model_opacity, model_scaling, model_rot, neural_opacity, mask = (
            generate_neural_gaussians(
                model_info.model_view,
                model_info.model_gaussians,
                model_visible_mask,
                True
            )
        )
        total_opacity_dict[model_info.model_id] = neural_opacity
        total_mask_dict[model_info.model_id] = mask
        total_visable_dict[model_info.model_id] = model_visible_mask

        curr_screenspace_points = torch.zeros((model_xyz.shape[0], 4), dtype=data_type, requires_grad=model_info.need_train, device="cuda")
        curr_points_with_model_id = torch.full((model_xyz.shape[0],), model_info.model_id)

        model_to_world_r = (
            torch.from_numpy(model_info.model_pose[:3, :3]).cuda().to(torch.float32)
        )
        model_to_world_t = (
            torch.from_numpy(model_info.model_pose[:3, 3]).cuda().to(torch.float32)
        )
        model_to_world_q = Rotation.from_matrix(
            torch.from_numpy(model_info.model_pose[:3, :3])
        ).as_quat()  # x y z w
        w1 = model_to_world_q[3]
        x1 = model_to_world_q[0]
        y1 = model_to_world_q[1]
        z1 = model_to_world_q[2]

        w2 = model_rot[:, 0]
        x2 = model_rot[:, 1]
        y2 = model_rot[:, 2]
        z2 = model_rot[:, 3]
        model_rot[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        model_rot[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        model_rot[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        model_rot[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        model_xyz = model_to_world_r @ model_xyz.t() + model_to_world_t.reshape(3, 1)

        if not init:
            total_xyz = model_xyz.t()
            total_color = model_color
            total_opacity = model_opacity
            total_scaling = model_scaling
            total_rot = model_rot
            total_screenspace_points = curr_screenspace_points
            total_points_with_model_id = curr_points_with_model_id
            init = True
        else:
            total_xyz = torch.cat((total_xyz, model_xyz.t()), dim=0)
            total_color = torch.cat((total_color, model_color), dim=0)
            total_opacity = torch.cat((total_opacity, model_opacity), dim=0)
            total_scaling = torch.cat((total_scaling, model_scaling), dim=0)
            total_rot = torch.cat((total_rot, model_rot), dim=0)
            total_screenspace_points = torch.cat((total_screenspace_points, curr_screenspace_points), dim=0)
            total_points_with_model_id = torch.cat((total_points_with_model_id, curr_points_with_model_id), dim = 0)

    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
    if retain_grad:
        try:
            total_screenspace_points.retain_grad()
        except:
            pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_cam.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_cam.beam_inclinations,  # TODO 输入一个beam
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, occ, radii = rasterizer(
        means3D = total_xyz,
        means2D = total_screenspace_points,
        shs = None,
        colors_precomp = total_color,
        opacities = total_opacity,
        scales = total_scaling,
        rotations = total_rot,
        cov3D_precomp = None)

    return {"render": rendered_image,
            "depth":depth,
            "occ":occ,
            "viewspace_points": total_screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "points_with_model_id": total_points_with_model_id,
            "selection_mask": total_mask_dict,
            "visable_mask": total_visable_dict,
            "neural_opacity": total_opacity_dict,
            "scaling": total_scaling,
            }