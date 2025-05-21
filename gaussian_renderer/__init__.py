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
from scipy.spatial.transform import Rotation

import math
from scene.gaussian_model import GaussianModel
from diff_lidargs_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import time
from utils.general_utils import rotation_matrix_to_quaternion, quaternionRawMultiply
import pdb

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, is_trick=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    if is_trick:
        ob_view = anchor - viewpoint_camera.lidar_center#viewpoint_camera.original_lidar_center if viewpoint_camera.original_lidar_center is not None else anchor - viewpoint_camera.lidar_center
    else:
        ob_view = anchor - viewpoint_camera.camera_center# 对于obj来说 camera_center 与lidar center不同 这是个bug 暂时将错就错

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
            raydrop = pc.get_raydrop_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
            raydrop = pc.get_raydrop_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
            raydrop = pc.get_raydrop_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
            raydrop = pc.get_raydrop_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, pc.color_channel-1])# TODO if using new raydrop mlp, should pc.color_channel-1
    raydrop = raydrop.reshape([anchor.shape[0]*pc.n_offsets, 1])
    color = torch.cat([color,raydrop],dim=1) # color and raydrop will be concatenated together , as the rasterization' input

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 6]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([5, 3, pc.color_channel, 6, 3], dim=-1)
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:2]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,2:6])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def renderComposite(viewpoint_cam, background, pipe, valid_model_info, max_depth, insert_objs=None, retain_grad=True):
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

    init = False  # TODO 是否直接把这部分数据预处理放到loader里
    for model_info in valid_model_info: 
        data_type = model_info.model_gaussians.get_anchor.dtype
        model_visible_mask = prefilter_voxel(
            model_info.model_view,
            model_info.model_gaussians,
            pipe,
            background,
            max_depth
        )
        model_xyz, model_color, model_opacity, model_scaling, model_rot, neural_opacity, mask = (
            generate_neural_gaussians(
                model_info.model_view,
                model_info.model_gaussians,
                model_visible_mask,
                True,
                is_trick = not init
            )
        )
        total_opacity_dict[model_info.model_id] = neural_opacity
        total_mask_dict[model_info.model_id] = mask
        total_visable_dict[model_info.model_id] = model_visible_mask

        curr_screenspace_points = torch.zeros((model_xyz.shape[0], 4), dtype=data_type, requires_grad=retain_grad, device="cuda")
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

    if retain_grad:
        try:
            total_screenspace_points.retain_grad()
        except:
            pass
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        bg=background,
        scale_modifier=1.0,
        depth_threshold=0.001,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_cam.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_cam.beam_inclinations,  # TODO 输入一个beam
        debug=pipe.debug,
        lidar_far = int(max_depth),
        lidar_near = int(0)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap, pixels = rasterizer(
        means3D = total_xyz,
        means2D = total_screenspace_points,
        shs = None,
        colors_precomp = total_color,
        opacities = total_opacity,
        scales = total_scaling,
        rotations = total_rot,
        cov3D_precomp = None)

    depth = allmap[0:1]

    if isinstance(insert_objs, list) and len(insert_objs) > 0:
        for each_obj in insert_objs:
            if "sim_pose" in each_obj:
                latest_xyz = (each_obj["xyz"] @ each_obj["sim_pose"].T)[:,:3]
                total_xyz = torch.cat((total_xyz, latest_xyz), dim=0)
            else:
                total_xyz = torch.cat((total_xyz, each_obj["xyz"][:,:3]), dim=0)
            total_color = torch.cat((total_color, each_obj["color"]), dim=0)
            total_opacity = torch.cat((total_opacity, each_obj["opacity"]), dim=0)
            total_scaling = torch.cat((total_scaling, each_obj["scaling"][:,:2]), dim=0)
            rot_obj2world = rotation_matrix_to_quaternion(each_obj["sim_pose"][:3,:3])
            real_rot = quaternionRawMultiply(each_obj["rot"], rot_obj2world)
            total_rot = torch.cat((total_rot, real_rot), dim=0)
            # total_rot = torch.cat((total_rot, each_obj["rot"]), dim=0)

            obj_screenspace_points = torch.zeros((each_obj["xyz"].shape[0], 4), dtype=data_type, requires_grad=False, device="cuda")
            total_screenspace_points = torch.cat((total_screenspace_points, obj_screenspace_points), dim=0)
        rendered_image_2, radii_2, allmap_2, _ = rasterizer(
            means3D = total_xyz,
            means2D = total_screenspace_points,
            shs = None,
            colors_precomp = total_color,
            opacities = total_opacity,
            scales = total_scaling,
            rotations = total_rot,
            cov3D_precomp = None)
        depth = allmap_2[0:1]
        rendered_image[1:2] = rendered_image_2[1:2]
    
    # rendered_image = rendered_image_2
    # radii = radii_2
    occ = allmap[1:2]
    mid_depth_diff = torch.abs(allmap[5:6] - allmap[0:1])

    return {"render": rendered_image,
            "depth":depth,
            "occ":occ,
            "mid_depth_diff": mid_depth_diff,
            "viewspace_points": total_screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "points_with_model_id": total_points_with_model_id,
            "selection_mask": total_mask_dict,
            "visable_mask": total_visable_dict,
            "neural_opacity": total_opacity_dict,
            "scaling": total_scaling,
            "xyz": total_xyz,
            "opa": total_opacity
            }


def render(viewpoint_cam, background, pipe, gs, max_depth, insert_objs=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = gs.get_color_mlp.training
    visible_mask = prefilter_voxel(
        viewpoint_cam,
        gs,
        pipe,
        background,
        max_depth
    )
        
    xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(
        viewpoint_cam, 
        gs, 
        visible_mask, 
        is_training = True,
        is_trick = True
    )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((xyz.shape[0], 4), dtype=xyz.dtype, requires_grad=is_training, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        bg=background,
        scale_modifier=1.0,
        depth_threshold=0.001,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_cam.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_cam.beam_inclinations,  # TODO 输入一个beam
        debug=pipe.debug,
        lidar_far = int(max_depth),
        lidar_near = int(0)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap, pixels  = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    depth = allmap[0:1] 
    
    if isinstance(insert_objs, list) and len(insert_objs) > 0:
        for each_obj in insert_objs:
            if "sim_pose" in each_obj:
                latest_xyz = (each_obj["xyz"] @ each_obj["sim_pose"].T)[:,:3]
                xyz = torch.cat((xyz, latest_xyz), dim=0)
            else:
                xyz = torch.cat((xyz, each_obj["xyz"][:,:3]), dim=0)
            color = torch.cat((color, each_obj["color"]), dim=0)
            opacity = torch.cat((opacity, each_obj["opacity"]), dim=0)
            scaling = torch.cat((scaling, each_obj["scaling"][:,:2]), dim=0)
            rot_obj2world = rotation_matrix_to_quaternion(each_obj["sim_pose"][:3,:3]) # 如果植入的物体不去转它，这个分量其实就是单位阵，但是需要考虑植入时去转它的情况，这时候世界坐标系下的rot会变
            real_rot = quaternionRawMultiply(each_obj["rot"], rot_obj2world)
            rot = torch.cat((rot, real_rot), dim=0)
            
            obj_screenspace_points = torch.zeros((each_obj["xyz"].shape[0], 4), dtype=xyz.dtype, requires_grad=is_training, device="cuda")
            screenspace_points = torch.cat((screenspace_points, obj_screenspace_points), dim=0)
        rendered_image_2, radii_2, allmap_2, _  = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None) 
        # rendered_image[0:1,...] = rendered_image_2[0:1,...]
        depth = allmap_2[0:1] # depth都覆盖 保留raydrop和intensity
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return {"render": rendered_image,
            "depth":depth,
            "occ":allmap[1:2],
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, max_depth, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=0.001,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_camera.beam_inclinations,
        debug=pipe.debug,
        lidar_far = int(max_depth),
        lidar_near = int(1)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:2],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
