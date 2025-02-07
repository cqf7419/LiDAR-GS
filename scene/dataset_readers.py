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
from utils.lidar_utils import lidar_to_pano_with_intensities, pano_to_lidar, get_beam_inclinations

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
    ray_dir: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # print(f'FovX: {FovX}, FovY: {FovY}')

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

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

def readColmapSceneInfo(path, images, eval, lod, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if lod>0:
            print(f'using lod, using eval')
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f'test_cam_infos: {len(test_cam_infos)}')
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False, undistorted=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# waymo alignmif 版本的json读取
def waymo_readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",data_label=None):
    ply_path = os.path.join(path, "points3d.ply")
    print(ply_path)
    cam_infos = []
    
    frames_train = None
    frames_test = None
    W_lidar = 2650
    H_lidar = 64
    beam_inclinations = None
    data_name = "waymo"

    # mask = np.ones((1,H_lidar, W_lidar))

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        W_lidar = contents["w_lidar"]
        H_lidar = contents["h_lidar"]
        frames_train = contents["frames"]
        if "beam_inclinations" in contents:
            beam_inclinations = contents["beam_inclinations"] # waymo
            data_name = "waymo"
        else:
            beam_inclinations = get_beam_inclinations(2.0,26.9,H_lidar).copy() # kitti  # 
            print(beam_inclinations)
            data_name = "kitti"

    if data_label == "waymo":
        test_transformsfile = "transforms_test.json"
    else:
        test_transformsfile = "transforms_"+data_label+"_test.json"
        print(test_transformsfile)
    with open(os.path.join(path, test_transformsfile)) as json_file:
        contents_test = json.load(json_file)
        frames_test = contents_test["frames"]

    pcds = []
    first_frame_ind = 0
    # for idx, frame in enumerate(frames_train):
    all_frame_num = 50
    print("[exp]: only train {} frame!".format(all_frame_num))
    for idx in range(all_frame_num):
        if data_name == "waymo":
            if idx == 10 or idx == 20 or idx == 31 or idx == 41: # test frame
                frame = frames_test[idx//10-1]
                print("test pcd file: ",idx,frame["file_path"])
            else:
                if idx==30 or idx==40:
                    frame = frames_train[idx-idx//10+1]
                else:
                    frame = frames_train[idx-idx//10]
                print("pcd file: ",idx,frame["file_path"])
        elif data_name == "kitti":
            if idx == 13 or idx == 26 or idx == 39: # test frame
                frame = frames_test[idx//13-1]
                print("test pcd file: ",idx,frame["file_path"])
            else:
                frame = frames_train[idx-idx//13]
                print("pcd file: ",idx,frame["file_path"])
        else:
            raise ValueError("错误的数据集加载")

        image_path = path+"/"+frame["file_path"]
        image_name = image_path.split('/')[-1].split('.')[0] # 

        fx = contents["fl_x"]
        fy = contents["fl_y"]
        cx = contents["cx"]
        cy = contents["cy"]
        w = contents["w"]
        h = contents["h"]
        FovX = focal2fov(focal=fx,pixels=w)
        FovY = focal2fov(focal=fy,pixels=h) # maybe unuseful
        intrinsics = [2.0,26.9]



        l2w = np.array(frame["lidar2world"]) # [4,4]
        w2l = np.linalg.inv(l2w)
        R = np.transpose(w2l[:3,:3]) 
        T = w2l[:3, 3]

        lidar_center = np.zeros([1,3],dtype=np.float32)
        lidar_center = (np.pad(lidar_center, ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]

        rangeview_image_lidar = np.load(os.path.join(path, frame["lidar_file_path"].replace(" ", ""))) 
        range_view = np.zeros((H_lidar, W_lidar, 3))
        range_view[:, :, 1] = rangeview_image_lidar[:,:,1] # intensities
        range_view[:, :, 2] = rangeview_image_lidar[:,:,2] #pano
        ray_drop = np.where(range_view.reshape(-1, 3)[:, 2] <= 0.0, 0.0,
                            1.0).reshape(H_lidar, W_lidar, 1)


        lidar_point = pano_to_lidar(rangeview_image_lidar[:, :, 2],beam_inclinations=beam_inclinations)
        pcd_world = (np.pad(lidar_point[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
        pcds.append(pcd_world)

        image_lidar = np.concatenate(
            [
                ray_drop,
                np.clip(range_view[:, :, 1, None], 0, 1),
                range_view[:, :, 2, None]
            ],
            axis=-1,
        )

        # render dir  # 
        i, j = np.meshgrid(np.arange(W_lidar, dtype=np.float32),
                        np.arange(H_lidar, dtype=np.float32),
                        indexing='xy')
        beta = -(i - W_lidar / 2.0) / W_lidar * 2.0 * np.pi
        alpha = np.expand_dims(beam_inclinations[::-1], 1).repeat(W_lidar, 1)
        dirs = np.stack([
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ], -1)


            # 存取每帧相机信息
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_lidar,
                        image_path=image_path, image_name=image_name, width=W_lidar, height=H_lidar,beam_inclinations = beam_inclinations,lidar_center=lidar_center,ray_dir=dirs))
        # break   # 只取一帧
        
    
    # 用于初始化gs的点云
    pointcloud = np.concatenate(pcds, axis=0)
    indices = np.random.choice(pointcloud.shape[0], 500000, replace=True) 
    pointcloud = pointcloud[indices]

    return cam_infos,pointcloud,ply_path
    
def readwaymoInfo(path, white_background, eval, data_label, extension=".png", ply_path=None): 
    print("Reading Training Transforms")
    if data_label == "waymo":
        cam_infos,pointcloud,ply_path = waymo_readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,data_label)
    else:
        cam_infos,pointcloud,ply_path = waymo_readCamerasFromTransforms(path, "transforms_"+data_label+"_train.json", white_background, extension,data_label)

    train_cam_infos = []
    test_cam_infos = []
    for idx, c in enumerate(cam_infos):
        if data_label == "waymo":
            if idx == 10 or idx == 20 or idx == 31 or idx == 41:
                test_cam_infos.append(c)
            else:
                train_cam_infos.append(c)
        else:
            if idx == 13 or idx == 26 or idx == 39:
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
        
        storePly(ply_path, pointcloud, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

#############################################################################################################
from scene.dataset_readers_dynmaic import readDynamicWaymoInfo
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "waymo": readwaymoInfo,
    "waymo_dynamic":readDynamicWaymoInfo,

}