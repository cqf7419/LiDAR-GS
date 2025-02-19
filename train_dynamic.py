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

import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from skimage.metrics import structural_similarity
import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim,raydrop_lossf,l2_loss
from gaussian_renderer import prefilter_voxel, render, network_gui, renderComposite
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from scene.waymoDynamic import waymo_dynamic

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.lidar_utils import pano_to_lidar_with_intensities,pano_to_lidar
from utils.lidar_utils import PointsMeter

from typing import NamedTuple

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera
    valid_mask: np.array
    gt_mask: torch.Tensor


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool


def training(waymo_dynamic_model, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    model_id_list = [0]
    model_id_list.extend(waymo_dynamic_model.get_dynamic_obj_id_list())
    print("model_id_list ", model_id_list)

    static_views = None
    model_id_scene_info = {}
    model_id = 0
    for model_name in model_id_list:
        model_gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.color_channel)
        model_scene = Scene(dataset, model_name, waymo_dynamic_model, model_gaussians, ply_path=ply_path, shuffle=False)
        
        if model_scene.init_status:
            model_gaussians.training_setup(opt)
            if checkpoint:
                (model_params, first_iter) = torch.load(checkpoint)
                model_gaussians.restore(model_params, opt)

            time_poses = {}
            total_views = model_scene.getTotalCameras()
            print("total_views ", len(total_views))
            
            if model_name == 0:
                static_views = total_views

            for view in total_views:
                timestamp = view.image_name
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = np.transpose(view.R)
                camera_pose[:3, 3] = view.T
                time_poses[timestamp] = TimePose(camera_pose=camera_pose, view=view, valid_mask=view.img_mask, gt_mask=view.original_image)

            model_id_scene_info[model_id] = GaussianView(gaussians=model_gaussians, scene=model_scene, time_poses=time_poses)
            model_id += 1

    test_timestamp = [10,20,31,41]
    train_views = []
    test_views = []
    for idx, scene_view in enumerate(static_views):
        render_timestamp = scene_view.image_name
        if render_timestamp in test_timestamp:
            test_views.append(scene_view)
        else:
            train_views.append(scene_view)

    valid_timestamp_model = {}
    for view in static_views:
        timestamp = view.image_name
        valid_timestamp_model[timestamp] = []
        for model_id in model_id_scene_info.keys():
            if timestamp in model_id_scene_info[model_id].time_poses:
                valid_timestamp_model[timestamp].append(model_id)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = []
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_intensity_loss_for_log = 0.0
    ema_scale_for_log = 0.0
    ema_dp_for_log = 0.0
    ema_dgx_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if len(viewpoint_stack) == 0:
            viewpoint_stack = train_views.copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_timestamp = viewpoint_cam.image_name

        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(viewpoint_cam.R)
        world_to_camera_pose[:3, 3] = viewpoint_cam.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.update_learning_rate(iteration)
            model_to_world = camera_to_world_pose @ model_to_camera_pose

            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=True
                )
            )

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = renderComposite(viewpoint_cam, background, pipe, valid_model_info, retain_grad=True)
        image, depth, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
            render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
            render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
        points_with_model_id = render_pkg["points_with_model_id"]
        visable_mask = render_pkg["visable_mask"]

        # --------------- intensity & depth loss -----------------------#
        gt_image = viewpoint_cam.original_image.cuda()
        ray_drop = gt_image[0:1,...]
        gt_intensity = gt_image[1:2,...] * ray_drop
        gt_depth = gt_image[2:3,...] * ray_drop
        render_intensity = image[0:1,...]

        if True:
            render_raydrop = image[1:2,...]
            render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
            render_intensity = render_intensity*ray_drop 
            depth = depth*ray_drop
            mse_loss = torch.nn.MSELoss()
            raydrop_loss = 10 * mse_loss(render_raydrop,ray_drop)
        else:
            render_intensity = render_intensity*ray_drop
            depth = depth*ray_drop
            raydrop_loss = torch.tensor([0.0],device="cuda")

        Ll1 = l1_loss(render_intensity, gt_intensity) 
        depth_loss = l1_loss(depth, gt_depth)
        ssim_loss = (1.0 - ssim(render_intensity, gt_intensity))

        # ---------------------- scale reg ----------------------------#
        scaling_reg = 0.01 * scaling.prod(dim=1).mean()
        if visibility_filter.sum() > 0:
            scale = scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            plane_reg = 10.0*min_scale_loss.mean()
            scaling_reg = scaling_reg + plane_reg

        # ---------------------- occ reg ------------------------------#
        alpha = render_pkg['occ']
        o = alpha.clamp(1e-6, 1-1e-6)
        loss_opa = -(o*torch.log(o)).mean() - ((1-o) * torch.log(1 - o)).mean() 
        loss_opa = 0.05 * loss_opa

        intensity_loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss)
        # loss = intensity_loss+ depth_loss + scaling_reg
        loss = depth_loss + scaling_reg + loss_opa + intensity_loss + raydrop_loss #+ grad_loss
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_intensity_loss_for_log = 0.4 * intensity_loss.item() + 0.6 * ema_intensity_loss_for_log
            ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            ema_scale_for_log = 0.4 * scaling_reg.item() + 0.6 * ema_scale_for_log
            ema_dp_for_log = 0.4*raydrop_loss.item() + 0.6*ema_dp_for_log
            # ema_dgx_for_log = 0.4*grad_loss.item() + 0.6*ema_dgx_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "i_L": f"{ema_intensity_loss_for_log:.{7}f}",
                    "d_L": f"{ema_depth_loss_for_log:.{7}f}",
                    # "s_l": f"{ema_scale_for_log:.{7}f}",
                    # "dgx": f"{ema_dgx_for_log:.{7}f}",
                    "dp_L": f"{ema_dp_for_log:.{7}f}",
                    # "r>0": f"{len(torch.nonzero(voxel_visible_mask))}",
                    # "anchor": f"{len(gaussians.get_anchor)}"
                    })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                print("---------------------train----------------------")
                train_composite_report(tb_writer, dataset_name, iteration, train_views, test_timestamp,\
                    model_id_scene_info, valid_timestamp_model, pipe, background, wandb, logger)

                print("---------------------test----------------------")
                train_composite_report(tb_writer, dataset_name, iteration, test_views, test_timestamp,\
                    model_id_scene_info, valid_timestamp_model, pipe, background, wandb, logger)

            for model_id, model_info in model_id_scene_info.items():
                if iteration in saving_iterations:
                    model_scene = model_info.scene
                    model_scene.save(iteration, model_id)
                
                model_gaussian = model_info.gaussians

                # densification
                if iteration < opt.update_until and iteration > opt.start_stat and model_gaussian.get_anchor.shape[0] < opt.densify_until_num_points:
                    if model_id in valid_timestamp_model[render_timestamp]:
                        curr_viewspace_point_tensor_grad = viewspace_point_tensor.grad[points_with_model_id == model_id]
                        curr_visibility_filter = visibility_filter[points_with_model_id == model_id]
                        model_gaussian.training_statis(curr_viewspace_point_tensor_grad, opacity[model_id], curr_visibility_filter, offset_selection_mask[model_id], visable_mask[model_id])  

                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0: # TODO 如果影响震荡 就把update_interval放大
                        model_gaussian.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                elif iteration == opt.update_until:
                    del model_gaussian.opacity_accum
                    del model_gaussian.offset_gradient_accum
                    del model_gaussian.offset_denom
                    torch.cuda.empty_cache()

                # Optimizer step
                if iteration < opt.iterations:
                    model_gaussian.optimizer.step()
                    model_gaussian.optimizer.zero_grad(set_to_none = True)
                if (iteration in checkpoint_iterations):
                    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((model_gaussian.capture(), iteration), model_info.scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def train_composite_report(tb_writer, dataset_name, iteration, static_views, test_timestamp,\
                    model_id_scene_info, valid_timestamp_model, pipe, background, wandb, logger):
    l1_test = 0.0
    psnr_test = 0.0
    in_mae = 0.0
    in_rmse = 0.0
    in_medae = 0.0
    in_ssim = 0.0
    # point 
    cd_test= 0.0
    fscore_test = 0.0
    rmse = 0.0
    mae = 0.0
    medae = 0.0
    
    Ll1 = 0
    depth_loss = 0
    ssim_loss = 0
    
    total_number = len(static_views)

    for idx, scene_view in enumerate(static_views):
        render_timestamp = scene_view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(scene_view.R)
        world_to_camera_pose[:3, 3] = scene_view.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.eval()
            model_to_world = camera_to_world_pose @ model_to_camera_pose
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=False
                )
            )
        
        render_pkg = renderComposite(scene_view, background, pipe, valid_model_info, retain_grad=False)
        image, depth = render_pkg["render"], render_pkg["depth"]

        gt_image = scene_view.original_image.cuda()
        ray_drop = gt_image[0:1,...]
        gt_intensity = gt_image[1:2,...] * ray_drop
        gt_depth = gt_image[2:3,...] * ray_drop
        render_intensity = image[0:1,...]

        if True:
            render_raydrop = image[1:2,...]
            render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
            render_intensity = render_intensity*ray_drop # 直接使用gt的raydrop mask
            depth = depth*ray_drop
            mse_loss = torch.nn.MSELoss()
            raydrop_loss = 10 * mse_loss(render_raydrop,ray_drop)

        if iteration >= 3000:
            depth_numpy = depth.detach().cpu().numpy()
            point_with_intensity = pano_to_lidar(depth_numpy[0, :, :], lidar_K=None, beam_inclinations=scene_view.beam_inclinations.detach().cpu().numpy())

            gt_depth_numpy = gt_depth.detach().cpu().numpy()
            gt_point_with_intensity = pano_to_lidar(gt_depth_numpy[0, :, :], lidar_K=None, beam_inclinations=scene_view.beam_inclinations.detach().cpu().numpy())            

            np.savetxt(model_id_scene_info[0].scene.model_path +"/"+ str(iteration) + "_" + str(render_timestamp) + "_render_points.txt", point_with_intensity, delimiter=",", fmt="%.2f")
            np.savetxt(model_id_scene_info[0].scene.model_path +"/"+ str(iteration) + "_" + str(render_timestamp) + "_gt__points.txt", gt_point_with_intensity, delimiter=",", fmt="%.2f")

        l1_test += l1_loss(render_intensity, gt_intensity)
        psnr_test += psnr(render_intensity, gt_intensity).mean().double()
        curr_depth_loss = l1_loss(depth, gt_depth)
        depth_loss += curr_depth_loss
        ssim_loss += (1.0 - ssim(render_intensity, gt_intensity))
        points_meter = PointsMeter(scale=1, intrinsics=None, beam_inclinations=scene_view.beam_inclinations.detach().cpu().numpy())
        
        points_meter.update(depth, gt_depth)
        cd_fs = points_meter.measure()
        cd_test += cd_fs[0]

        error_in_abs = torch.abs(render_intensity - gt_intensity)
        in_mae += error_in_abs.mean()
        in_rmse += torch.sqrt((error_in_abs*error_in_abs).mean())
        in_medae += error_in_abs.median()

        in_ssim += structural_similarity(render_intensity[0].detach().cpu().numpy(),
                    gt_intensity[0].detach().cpu().numpy(),
                    data_range=1.0)
        
        fscore_test += cd_fs[1]
        error_depth_abs = torch.abs(depth - gt_depth)
        mae += error_depth_abs.mean()
        rmse += torch.sqrt((error_depth_abs*error_depth_abs).mean())
        medae += error_depth_abs.median()

    psnr_test /= total_number
    l1_test /= total_number
    in_ssim /= total_number
    in_mae /= total_number
    in_rmse /= total_number
    in_medae /= total_number

    cd_test /= total_number
    fscore_test /= total_number
    mae /= total_number   
    rmse /= total_number 
    medae /= total_number

    logger.info("\n[ITER {}] Evaluating: intensity: L1 {} PSNR {} SSIM {} MAE {} RMSE {} MadAE {} // depth: CD {} Fscore {} MAE {} MedAE {} RMSE {}".format(\
                        iteration, l1_test, psnr_test,in_ssim,in_mae,in_rmse,in_medae, cd_test, fscore_test,mae,medae,rmse))

    torch.cuda.empty_cache()
    for model_id in valid_timestamp_model[render_timestamp]:
        model_gaussian = model_id_scene_info[model_id].gaussians
        model_gaussian.train()


def training_report(tb_writer, opt, dataset_name, iteration, Ll1, depth_loss, loss, l1_loss, elapsed, testing_iterations, \
                    scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/intensity_l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # intensity
                l1_test = 0.0
                psnr_test = 0.0
                in_mae = 0.0
                in_rmse = 0.0
                in_medae = 0.0
                in_ssim = 0.0
                # point 
                cd_test= 0.0
                fscore_test = 0.0
                rmse = 0.0
                mae = 0.0
                medae = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    gt_image = viewpoint.original_image.cuda()
                    ray_drop = gt_image[0:1,...]
                    gt_intensity = gt_image[1:2,...] * ray_drop
                    gt_depth = gt_image[2:3,...] * ray_drop
                    render_raydrop = render_pkg["render"][1:2,...]
                    render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
                    image = torch.clamp(render_pkg["render"][0:1,...], 0.0, 1.0) *render_raydrop_mask
                    if tb_writer and (idx < 10):
                        from utils.general_utils import colormap
                        depth = render_pkg["depth"] * ray_drop
                        depth = depth / opt.depth_max 
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_intensity[None], global_step=iteration)
                            gt_depth = gt_depth/opt.depth_max
                            gt_depth = colormap(gt_depth.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth_depth".format(viewpoint.image_name), gt_depth[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_intensity).mean().double()
                    psnr_test += psnr(image, gt_intensity).mean().double()
                    error_in_abs = torch.abs(image - gt_intensity)
                    in_mae += error_in_abs.mean()
                    in_rmse += torch.sqrt((error_in_abs*error_in_abs).mean())
                    in_medae += error_in_abs.median()
                    in_ssim += structural_similarity(image[0].detach().cpu().numpy(),
                                gt_intensity[0].detach().cpu().numpy(),
                                data_range=1.0)
                    
                    depth_max = opt.depth_max
                    depth_min = opt.depth_min # kitti = 1  waymo == 5
                    depth_render = torch.clamp(render_pkg["depth"][0:1,...], depth_min, depth_max) *render_raydrop_mask 
                    points_meter = PointsMeter(scale=1, intrinsics=None, beam_inclinations=viewpoint.beam_inclinations.detach().cpu().numpy())
                    points_meter.update(depth_render, gt_image[2:3,...] * ray_drop)
                    cd_fs = points_meter.measure()
                    cd_test += cd_fs[0]
                    fscore_test += cd_fs[1]
                    error_depth_abs = torch.abs(depth_render - gt_image[2:3,...] * ray_drop)
                    mae += error_depth_abs.mean()
                    rmse += torch.sqrt((error_depth_abs*error_depth_abs).mean())
                    medae += error_depth_abs.median()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                in_ssim /= len(config['cameras']) 
                in_mae /= len(config['cameras'])  
                in_rmse /= len(config['cameras']) 
                in_medae /= len(config['cameras']) 

                cd_test /= len(config['cameras'])   
                fscore_test /= len(config['cameras'])  
                mae /= len(config['cameras'])   
                rmse /= len(config['cameras'])   
                medae /= len(config['cameras'])   
                logger.info("\n[ITER {}] Evaluating {}: intensity: L1 {} PSNR {} SSIM {} MAE {} RMSE {} MadAE {} // depth: CD {} Fscore {} MAE {} MedAE {} RMSE {}".format(iteration, config['name'], l1_test, psnr_test,in_ssim,in_mae,in_rmse,in_medae, cd_test, fscore_test,mae,medae,rmse))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"][0:1,...], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    print("model_path ", model_path)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    print("args.source_path ", args.source_path)
    print("dataset ", dataset)
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training

    waymo_dynamic_model = waymo_dynamic("/mnt_gx/usr/lansheng/processed_data_dynamic/13271285919570645382_5320_000_5340_000")
    training(waymo_dynamic_model, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(waymo_dynamic_model, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")
