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
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
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

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    total_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    print("scene radius: ", radius)

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# waymo alignmif 版本的json读取
def waymo_readCamerasFromTransforms(model_id, waymo_dynamic_model, path, transformsfile, white_background, extension=".png"):
    ply_path = os.path.join(path, "points3d.ply")
    cam_infos = []
    
    W_lidar = 2650
    H_lidar = 64
    beam_inclinations = waymo_dynamic_model.get_beam_inclination()
    pointcloud = []

    if model_id == 0:
        occured_frames = [ℹ for i in range(50)]
    else:
        occured_frames = waymo_dynamic_model.get_obj_frames(model_id)
        
    if len(occured_frames) < 5:
        return None, None, None
        
    start_frame_id = occured_frames[0]
    
    print("model_id ", model_id)
    print("occured_frames ", occured_frames)
    print("start_frame_id ", start_frame_id)

    all_frame_num = 50
    all_l2w = waymo_dynamic_model.getlidar2world()
    obj_occur_id = 0
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
            img_mask = waymo_dynamic_model.get_static_mask(idx)
            curr_point = waymo_dynamic_model.get_static_pcd_each_frame(idx, l2w)
        else:
            object2world = waymo_dynamic_model.get_obj2world(obj_occur_id, model_id)
            object2lidar = w2l @ object2world
            lidar2object = np.linalg.inv(object2lidar)
            obj_occur_id += 1

            R = np.transpose(object2lidar[:3,:3])
            T = object2lidar[:3, 3]
            _, img_mask = waymo_dynamic_model.get_mask(idx, model_id)
            curr_point = waymo_dynamic_model.get_object_pcd_each_frame(model_id, idx, lidar2object)   # object to world

        pointcloud.append(curr_point)

        lidar_center = np.zeros([1,3],dtype=np.float32)
        
        lidar_center = (np.pad(lidar_center, ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
        image_lidar = waymo_dynamic_model.get_rangeview(idx,H_lidar,W_lidar)

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_lidar,
                        image_path=None, image_name=idx, width=W_lidar, height=H_lidar,beam_inclinations = beam_inclinations,lidar_center=lidar_center,
                        img_mask=img_mask))

    pointcloud = np.concatenate(pointcloud, axis=0)

    sample_number = 500000
    if model_id != 0:
        sample_number = 10000

    if pointcloud.shape[0] > sample_number:
        indices = np.random.choice(pointcloud.shape[0], sample_number, replace=True)
        pointcloud = pointcloud[indices]
    

    print("pointcloud number ", pointcloud.shape[0])
    
    if pointcloud.shape[0] < 100:
        return None, None, None

    return cam_infos,pointcloud,ply_path


def readDynamicWaymoInfo(model_id, waymo_dynamic_model, path, white_background, eval, extension=".png", ply_path=None): 
    print("Reading Training Transforms")
    cam_infos,pointcloud,ply_path = waymo_readCamerasFromTransforms(model_id, waymo_dynamic_model, path, "transforms_train.json", white_background, extension)
    
    if cam_infos is None:
        return None

    train_cam_infos = []
    test_cam_infos = []
    for idx, c in enumerate(cam_infos):
        if idx == 10 or idx == 20 or idx == 31 or idx == 41:
            test_cam_infos.append(c)
        else:
            train_cam_infos.append(c)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if os.path.exists(ply_path):  
        os.remove(ply_path)#删除文件
    if not os.path.exists(ply_path):
                # 初始化颜色和法向量
        num_pts = pointcloud.shape[0]
        shs = np.zeros((num_pts,3)) # np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=pointcloud, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           total_cameras=cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
