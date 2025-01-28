# waymo alignmif 版本的json读取
import os
import json
import numpy as np
from scene.dataset_readers import CameraInfo,SceneInfo,getNerfppNorm,storePly,fetchPly,BasicPointCloud
from utils.lidar_utils import lidar_to_pano_with_intensities,cal_beam_inclinations
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
import pickle
from typing import NamedTuple

class LidarInfo(NamedTuple):
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


def gt2_readCamerasFromTransforms(path, white_background, extension=".png"):
    ply_path = os.path.join(path, "points3d.ply")
    cam_infos = []
    
    pickle_file_path = path + "/"+os.path.basename(path)+".pkl"
    with open(pickle_file_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        frames = data["frames"]
        beam_inclinations = cal_beam_inclinations()
        W_lidar = int(360/0.2)
        H_lidar = int(32)
        pcds = []
        list_c2ws = []
        # beam_inclinations = contents['beam_inclinations']
        for idx, frame in enumerate(frames):
            if idx<1: continue # 跳过第一帧 点云有问题
            if idx>=50: break

            if "optimized_pose" in frame:
                l2w = np.array(frame["optimized_pose"])
            else:
                l2w = np.array(frame["lidar2world"]) # [4,4]

            # angle_z = np.arctan2(l2w[1, 0], l2w[0, 0])
            # Rz = np.array([[ np.cos(angle_z), -np.sin(angle_z), 0],
            #             [ np.sin(angle_z),  np.cos(angle_z), 0],
            #             [             0,             0, 1]])
            # l2w[:3, :3] = Rz

            w2l = np.linalg.inv(l2w)
            R = np.transpose(w2l[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2l[:3, 3]


            # 读取每帧lidar数据
            lidar_center = np.zeros([1,3],dtype=np.float32)
            lidar_center = (np.pad(lidar_center, ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]

            pcd_path = frame["path"]["pcd"]
            pcd_path = "/home/xuanyuan/lansheng/GT2_DATA/" + pcd_path
            point_cloud_data = np.load(pcd_path)["data"] 
            point_cloud_data = point_cloud_data.reshape((-1, 7))
            pcd = point_cloud_data[np.where(point_cloud_data[:,6]==0)] # 根据sendor_id划分各雷达数据
            pcd = pcd[np.where(pcd[:,5]==0)] # 删除自车点

            
            pcd_world = (np.pad(point_cloud_data[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
            pcds.append(pcd_world)
            print("pcd size: ",pcd_world.shape)
            point_cloud_data[...,0:3] = pcd_world[...,0:3] - lidar_center

            pano, intensities = lidar_to_pano_with_intensities(
                local_points_with_intensities=point_cloud_data,
                lidar_H=H_lidar,
                lidar_W=W_lidar,
                beam_inclinations=beam_inclinations,
                max_depth=150,
            )
            range_view = np.zeros((H_lidar, W_lidar, 3))
            range_view[:, :, 1] = intensities
            range_view[:, :, 2] = pano
            ray_drop = np.where(range_view.reshape(-1, 3)[:, 2] == 0.0, 0.0,
                                1.0).reshape(H_lidar, W_lidar, 1)
            image_lidar = np.concatenate(
                [
                    ray_drop,
                    np.clip(range_view[:, :, 1, None], 0, 1),
                    range_view[:, :, 2, None]
                ],
                axis=-1,
            )

             # 存取每帧相机信息
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=None, FovX=None, image=image_lidar,
                            image_path=None, image_name=None, width=W_lidar, height=H_lidar,beam_inclinations = beam_inclinations,lidar_center=lidar_center))
            # break   # 只取一帧
            
        
        # 用于初始化gs的点云
        pointcloud = np.concatenate(pcds, axis=0)
        indices = np.random.choice(pointcloud.shape[0], 1000000, replace=True) # 随机取100w个点
        pointcloud = pointcloud[indices]

        return cam_infos,pointcloud,ply_path
    
def readgtInfo(path, white_background, eval, extension=".png", ply_path=None): 
    print("Reading Training Transforms")
    cam_infos,pointcloud,ply_path = gt2_readCamerasFromTransforms(path, white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = waymo_readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    # if eval:
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx+1)%20 != 0]  # TODO hard code 每20帧作为一次测试 
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx+1)%20 == 0]   # 跳过第一帧
    # else:
    # train_cam_infos = cam_infos
    # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # if os.path.exists(ply_path):  
    #     os.remove(ply_path)#删除文件
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