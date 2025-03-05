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
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import imageio
# from utils.lidar_utils import lidar_to_pano_with_intensities

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    beam_inclinations: np.array
    lidar_center: np.array
    img_mask: np.array
    original_lidar_center: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    total_cameras: list
    nerf_normalization: dict
    ply_path: str

def GT_readCamerasFromTransforms(model_id, waymo_dynamic_model, path, transformsfile, white_background, newcar_render=None, extension=".png"):
    ply_path = os.path.join(path, "points3d.ply")
    cam_infos = []
    
    W_lidar, H_lidar = waymo_dynamic_model.get_lidar_res()
    beam_inclinations = waymo_dynamic_model.get_beam_inclination()
    all_frame_num = waymo_dynamic_model.get_frames_nums()

    if model_id == 0:
        occured_frames = [i for i in range(all_frame_num)]
    else:
        occured_frames = waymo_dynamic_model.get_obj_frames(model_id)
        
    if len(occured_frames) < 1: 
        return None, None, None
        
    start_frame_id = occured_frames[0]
    
    # print("model_id ", model_id)
    # print("occured_frames ", occured_frames)
    # print("start_frame_id ", start_frame_id)

    all_l2w = []
    original_l2w = []
    if newcar_render == "GT2V1":
        all_l2w = waymo_dynamic_model.getlidar2world_newcar()
        original_l2w = waymo_dynamic_model.getlidar2world()
        print("[ New Car Info ]")
    else:
        all_l2w = waymo_dynamic_model.getlidar2world()
        original_l2w = all_l2w

    for idx in range(all_frame_num):
        if idx not in occured_frames:
            continue

        FovX = 2
        FovY = 2
        l2w = all_l2w[idx]
        w2l = np.linalg.inv(l2w)

        if model_id == 0:            
            R = np.transpose(w2l[:3,:3])
            T = w2l[:3, 3]
            img_mask = waymo_dynamic_model.get_mask(idx)
        else:
            object2lidar = waymo_dynamic_model.get_obj2lidar(idx, model_id, newcar_render=False)
            lidar2object = np.linalg.inv(object2lidar)

            R = np.transpose(object2lidar[:3,:3])
            T = object2lidar[:3, 3]
            img_mask = np.ones((H_lidar, W_lidar))

        lidar_center_zero = np.zeros([1,3],dtype=np.float32)
        lidar_center = (np.pad(lidar_center_zero, ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
        original_lidar_center = (np.pad(lidar_center_zero, ((0,0),(0, 1)), constant_values=1) @ original_l2w[idx].T)[:,:3]
        image_lidar = waymo_dynamic_model.get_rangeview(idx)
        image_name = waymo_dynamic_model.frameid_2_timestep[idx]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_lidar,
                        image_path=None, image_name=image_name, width=W_lidar, height=H_lidar,beam_inclinations = beam_inclinations,lidar_center=lidar_center,
                        img_mask=img_mask, original_lidar_center=original_lidar_center))
    
    if waymo_dynamic_model.train:
        if model_id == 0:
            pointcloud = waymo_dynamic_model.get_static_pcd()
            sample_number = 500000
        else:
            pointcloud = waymo_dynamic_model.get_obj_pcd(model_id)
            sample_number = 10000 if pointcloud.shape[0]>10000 else pointcloud.shape[0]

        if pointcloud.shape[0] > sample_number:
            indices = np.random.choice(pointcloud.shape[0], sample_number, replace=True)
            pointcloud = pointcloud[indices]

        print("pointcloud number ", pointcloud.shape[0])
    
        if pointcloud.shape[0] < 1:
            return None, None, None
    else:
        if model_id == 0:
            pointcloud = None
        else:
            pointcloud = waymo_dynamic_model.get_obj_pcd(model_id)
            if pointcloud.shape[0] < 1:
                return None, None, None

    return cam_infos,pointcloud,ply_path


def readGTInfo(model_id, waymo_dynamic_model, path, white_background, eval, newcar_render=None , extension=".png", ply_path=None): 
    cam_infos,pointcloud,ply_path = GT_readCamerasFromTransforms(model_id, waymo_dynamic_model, path, "transforms_train.json", white_background, newcar_render, extension)
    
    if cam_infos is None:
        return None

    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = {"translate": 0.0, "radius": 1.0}

    num_pts = pointcloud.shape[0] if pointcloud is not None else 1
    shs = np.zeros((num_pts,3)) # np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=pointcloud, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           total_cameras=cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
###########################################################################################

sceneLoadTypeCallbacks = {
    "GT": readGTInfo
}