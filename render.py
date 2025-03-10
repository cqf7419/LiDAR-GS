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
import torch

import numpy as np
import sys
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel, renderComposite
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from scene import Scene

import cv2
from utils.lidar_utils import PointsMeter, pano_to_lidar_with_intensities, filter_pcd
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
from scene.cameras import Camera
import open3d as o3d
from utils.obj_utils import get_obj_type, loadStaticObj
from utils.data_partition_utils import getBlockInfo
import math

cpu_count = os.cpu_count()
torch.set_num_threads(cpu_count)

from typing import NamedTuple
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

def render_set(gt_dynamic_model, dataset, name, iteration, valid_timestamp_model, model_id_scene_info, views, pipeline, background, insert_objs, insert_dynamic_obj=False):
    path_name = dataset.model_path.split("/")# /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS/outputs/scene_3/track_0/TOP
    # pesudo_path = os.path.join("/mnt_gx/cqf/LiDAR_Pesudo_Data/ParaLane", path_name[-3], path_name[-2]+"_to_"+dataset.para_lane_track_list[0] ,path_name[-1]) 
    # render_path = os.path.join(pesudo_path, "renders")
    # gt_path = os.path.join(pesudo_path, "gt")
    # os.makedirs(render_path, exist_ok=True)
    # os.makedirs(gt_path, exist_ok=True)
    render_path = os.path.join(dataset.model_path, "renders")
    gt_path = os.path.join(dataset.model_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_timestamp = view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(view.R)
        world_to_camera_pose[:3, 3] = view.T
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

        torch.cuda.synchronize(); t0 = time.time()
        render_pkg = renderComposite(view, background, pipeline, valid_model_info, dataset.max_depth, insert_objs = insert_objs, retain_grad=False)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]#.detach().cpu().numpy()
        render_intensity = rendering[0:1,...]
        depth = render_pkg["depth"]

        gt = view.original_image.cuda()
        ray_drop = gt[0:1,...]
        gt_intensity = (gt[1:2,...] * ray_drop).detach().cpu().numpy()
        gt_depth = (gt[2:3,...] * ray_drop).detach().cpu().numpy()
        render_raydrop = rendering[1:2,...]
        render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
        if dataset.newcar_render == "GT2V1":
            render_intensity = render_intensity*render_raydrop_mask*ray_drop # 直接使用gt的raydrop mask
            depth = depth*render_raydrop_mask*ray_drop
            occ = render_pkg["occ"]*render_raydrop_mask*ray_drop
        else:
            render_intensity = render_intensity*render_raydrop_mask*ray_drop
            depth = depth*render_raydrop_mask*ray_drop
            occ = render_pkg["occ"]*render_raydrop_mask*ray_drop

        depth_numpy = depth.detach().cpu().numpy()
        intensity_numpy = render_intensity.detach().cpu().numpy()

        point_with_intensity = pano_to_lidar_with_intensities(depth_numpy[0, :, :],intensity_numpy[0], lidar_K=None, beam_inclinations=view.beam_inclinations.detach().cpu().numpy())
        gt_point_with_intensity = pano_to_lidar_with_intensities(gt_depth[0, :, :],gt_intensity[0], lidar_K=None, beam_inclinations=view.beam_inclinations.detach().cpu().numpy())
        
        if False: # 密度滤波
            make_raydrop = filter_pcd(point_with_intensity[:,:3])
            point_with_intensity = point_with_intensity[make_raydrop]

        if True:# 转到baselidar系
            if name == "simulation_newcar":
                sensor2baselidar = gt_dynamic_model.get_sensor2baselidar_newcar(dataset.sensorid)
            else:
                sensor2baselidar = gt_dynamic_model.get_sensor2baselidar(dataset.sensorid) 
            points = point_with_intensity[:,:3]
            points = (np.pad(points[...,:3], ((0,0),(0, 1)), constant_values=1) @ sensor2baselidar.T)[:,:3]  
            point_with_intensity[:,:3] = points
            gt_points = gt_point_with_intensity[:,:3]
            gt_points = (np.pad(gt_points[...,:3], ((0,0),(0, 1)), constant_values=1) @ sensor2baselidar.T)[:,:3] 
            gt_point_with_intensity[:,:3] = gt_points

        # header = "X Y Z Intensity\n"  # 保存点云
        np.savetxt(os.path.join(render_path, "{}.txt".format(render_timestamp)), point_with_intensity, fmt='%.4f', comments='') # header=header,
        np.savetxt(os.path.join(gt_path, "{}.txt".format(render_timestamp)), gt_point_with_intensity, fmt='%.4f', comments='') # header=header,


    
     
def render_sets(gt_dynamic_model, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, insert_static_obj : bool, insert_dynamic_obj:bool, insert_objs = None, obj_type=None):
    model_id_list = [0]
    if gt_dynamic_model.get_dynamic_obj_id_list() is not None:
        model_id_list.extend(gt_dynamic_model.get_dynamic_obj_id_list())
    print("model_id_list ", model_id_list) 

    with torch.no_grad():
        static_views = None
        model_id_scene_info = {}

        for model_id in model_id_list:
            model_gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                    dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.color_channel)
            model_scene = Scene(dataset, model_id, gt_dynamic_model, model_gaussians,load_iteration=iteration, shuffle=False)
            
            if model_scene.init_status:

                time_poses = {}
                total_views = model_scene.getTotalCameras()
                print("total_views ", len(total_views))
                
                if model_id == 0:
                    static_views = total_views

                for view in total_views:
                    timestamp = view.image_name
                    camera_pose = np.eye(4)
                    camera_pose[:3, :3] = np.transpose(view.R)
                    camera_pose[:3, 3] = view.T
                    time_poses[timestamp] = TimePose(camera_pose=camera_pose, view=view, valid_mask=view.img_mask, gt_mask=view.original_image)

                model_id_scene_info[model_id] = GaussianView(gaussians=model_gaussians, scene=model_scene, time_poses=time_poses) 
                
            model_gaussians.eval()
        ######### TODO 
        test_timestamp = []#[10,20,31,41]
        train_views = []
        test_views = []
        for idx, scene_view in enumerate(static_views):
            render_timestamp = scene_view.image_name
            if render_timestamp in test_timestamp:
                test_views.append(scene_view)
            else:
                train_views.append(scene_view)
        ######## so hard
        valid_timestamp_model = {}
        for view in static_views:
            timestamp = view.image_name
            valid_timestamp_model[timestamp] = []
            for model_id in model_id_scene_info.keys():
                if timestamp in model_id_scene_info[model_id].time_poses:
                    valid_timestamp_model[timestamp].append(model_id)  # 记录每一帧timestep（0-50）下出现的obj的id


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            if insert_static_obj:
                if obj_type is not None:
                    render_set(gt_dynamic_model, dataset, "simulation/"+obj_type, iteration, valid_timestamp_model, model_id_scene_info, train_views, pipeline, background, insert_objs)
                else:
                    render_set(gt_dynamic_model, dataset, "simulation_add_manhole", iteration, valid_timestamp_model, model_id_scene_info, train_views, pipeline, background, insert_objs)
            elif insert_dynamic_obj:
                render_set(gt_dynamic_model, dataset, "simulation_dynamic", iteration, valid_timestamp_model, model_id_scene_info, train_views, pipeline, background, insert_objs, insert_dynamic_obj=True)
            elif dataset.newcar_render == "GT2V1":
                render_set(gt_dynamic_model, dataset, "simulation_newcar", iteration, valid_timestamp_model, model_id_scene_info, train_views, pipeline, background, insert_objs)
            else:
                render_set(gt_dynamic_model, dataset, "train", iteration, valid_timestamp_model, model_id_scene_info, train_views, pipeline, background, insert_objs)

        # if not skip_test:
        #      render_set(dataset, "test", iteration, valid_timestamp_model, model_id_scene_info, test_views, pipeline, background, insert_objs)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser) # , sentinel=True
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--insert_static_obj", action="store_true")
    parser.add_argument("--insert_dynamic_obj", action="store_true")
    parser.add_argument("--blockinfo", type=str, default = None)
    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    model_args = model.extract(args)    
    model_args.para_lane_track_list = args.para_lane_track_list.split(",")
    print("para_lane_track_list", model_args.para_lane_track_list )
    if args.caseid == "None":
        from scene.GT_ParaLane_dataloader import GT_Dataloader
        # block_info_with_extend, block_info_without_extend, _ , __= getBlockInfo("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/ParaLane")
    elif args.caseid == 'pesudo':
        from scene.GT_ParaLane_Common_dataloader import GT_Dataloader
    else:
        if "segment" in args.caseid:
             from scene.Waymo_Dynamic_dataloader import Waymo_Dataloader as GT_Dataloader
        else:
            from scene.GT_Dynamic_dataloader import GT_Dataloader
    block_info_with_extend, block_info_without_extend, _, __ = getBlockInfo(args.blockinfo)
    for block_id, train_frame_times in block_info_without_extend.items():
        gt_dynamic_model = GT_Dataloader(model_args, train=False, train_frame_times=train_frame_times)
        model_args.block_id = block_id # update block id
        objs = None
        render_sets(gt_dynamic_model, model_args, 4000, pipeline.extract(args), args.skip_train, args.skip_test, args.insert_static_obj, args.insert_dynamic_obj, objs)
        