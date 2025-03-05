import sys
import time
sys.path.append('../')
from utils.data_partition_utils import getBlockInfo, judgeWhichBlock
from utils.obj_utils import loadStaticObj

from loguru import logger
import torch
import os
import numpy as np
import math

import subprocess
from typing import NamedTuple
from argparse import ArgumentParser

from gaussian_renderer import render
from utils.lidar_utils import pano_to_lidar_with_intensities, filter_pcd, get_pcdi
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.lidar_utils import cal_beam_inclinations, load_yaml_str, load_extrinsics, pano_to_lidar, pano_to_lidar_with_intensities
from threading import Thread
import threading

try:
    cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
    result = (
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        np.argmin([int(x.split()[2]) for x in result[:-1]])
    )

    os.system("echo $CUDA_VISIBLE_DEVICES")
    cpu_count = os.cpu_count()
    torch.set_num_threads(cpu_count)
except Exception as e:
    logger.info(f"发生未知错误: {e}")
    
class LidarInfo(NamedTuple):
    uid: int
    image_width: int
    image_height: int
    FoVx: float
    FoVy: float
    beam_inclinations: torch.tensor
    lidar_center: torch.tensor
    camera_center: torch.tensor
    original_lidar_center: torch.tensor
    world_view_transform: torch.tensor
    full_proj_transform: torch.tensor
    def withNewInfo(self, new_world_view_transform: torch.tensor, new_lidar_center: torch.tensor, new_original_lidar_center: torch.tensor):
        return self._replace(world_view_transform=new_world_view_transform, lidar_center=new_lidar_center, original_lidar_center=new_original_lidar_center)

def load_lidar_extrinsics(source_path): 
    sensor_yaml = source_path + '/frame/sensors.yaml'
    sensors = load_yaml_str(sensor_yaml)
    baselidar2body = load_extrinsics(sensors["BASE_LIDAR"]["initial_transform"])
    body2baselidar = np.linalg.inv(baselidar2body)
    top2body = load_extrinsics(sensors["ROBO_TOP"]["initial_transform"])
    # sensor2top['TOP'] = np.identity(4)
    backsensor2top = load_extrinsics(sensors['ROBO_BACK']["initial_transform"])
    leftsensor2top = load_extrinsics(sensors['ROBO_LEFT_FRONT']["initial_transform"])
    rightsensor2top = load_extrinsics(sensors['ROBO_RIGHT_FRONT']["initial_transform"])

    sensor2baselidar = dict()
    sensor2baselidar['TOP'] = body2baselidar @ top2body 

    back2body = top2body @ backsensor2top 
    sensor2baselidar['BACK'] = body2baselidar @ back2body 

    left2body = top2body @ leftsensor2top
    sensor2baselidar['LEFT'] = body2baselidar @ left2body 

    right2body = top2body @ rightsensor2top 
    sensor2baselidar['RIGHT'] = body2baselidar @ right2body 

    frontcam2toplidar = load_extrinsics(sensors['CAMERA_FRONT']["initial_transform"]) 
    frontcam2baselidar = sensor2baselidar['TOP']  @ frontcam2toplidar
    return sensor2baselidar, baselidar2body, frontcam2baselidar

def debugGetGTPose():
    from scipy.spatial.transform import Rotation
    file_path = "/mnt_gx/usr/lansheng/regroup_92171_all/temp/GT2-00006_20240614150446_20240614150512_9/annotation/single_frame_cloud_poses.txt"
    trans_matrix = np.eye(4)
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            pcd_name = data[1]
            print("[ debug ] using ",pcd_name)

            quaternion = [
                float(data[4]),
                float(data[5]),
                float(data[6]),
                float(data[7]),
            ]
            rot_matrix = Rotation.from_quat(quaternion).as_matrix()
            trans_matrix[:3, :3] = rot_matrix
            trans_matrix[0, 3] = data[8]
            trans_matrix[1, 3] = data[9]
            trans_matrix[2, 3] = data[10]
            break
    return trans_matrix

class LidarSimOnline:
    def __init__(self, source_path, caseid, sensorid, sensor2baselidar, model_path, iterations, block_area_info): # , framestamps
        self.sensor2baselidar = sensor2baselidar
        self.block_area_info = block_area_info
        parser = ArgumentParser(description="Simulation Render")
        self.model_params = ModelParams(parser)
        self.pipe_params = PipelineParams(parser)

        self.args, _ = parser.parse_known_args(sys.argv[1:])
        self.args.gpu = '0'
        self.iteration = iterations # default -1
        self.args.caseid = caseid
        self.args.sensorid = sensorid

        self.args.source_path = source_path
        self.args.model_path = model_path
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

        if self.args.caseid == "None":
            from scene.GT_ParaLane_dataloader import GT_Dataloader
        else:
            from scene.GT_Dynamic_dataloader import GT_Dataloader
            
        self.model_args = self.model_params.extract(self.args)
        self.loadLidarInfo()
        self.static_gaussian_model_dict = {} #self.renderSet(self.model_args, self.block_area_info)
        self.curr_block_id = None
        self.thread_lock = threading.Lock()

    def loadLidarInfo(self):
        self.W_lidar = int(360/0.2)
        self.H_lidar = int(32)
        beam_inclinations = cal_beam_inclinations()
        self.beam_inclinations = torch.tensor(beam_inclinations,dtype=torch.float32).cuda()
        lidar_map = {0:'TOP', 1:'BACK', 3:'LEFT', 4:'RIGHT'}
        self.lidar_name = lidar_map[self.model_args.sensorid]

    def renderSet(self, model_args, block_area_info):
        '''
        不考虑原场景内的动态物体的设置 / 或者适用于只有静态背景的ParaLane数据集
        '''
        # load scene gs
        model_dict = {}
        for block_id, area in block_area_info.items():
            with torch.no_grad():
                model_gaussians = GaussianModel(model_args.feat_dim, model_args.n_offsets, model_args.voxel_size, model_args.update_depth, model_args.update_init_factor, model_args.update_hierachy_factor, model_args.use_feat_bank, 
                                        model_args.appearance_dim, model_args.ratio, model_args.add_opacity_dist, model_args.add_cov_dist, model_args.add_color_dist, model_args.color_channel)
                loaded_iter = 4000 # default 
                if self.iteration == -1:
                    loaded_iter = searchForMaxIteration(os.path.join(model_args.model_path,"0", str(block_id)))
                else:
                    loaded_iter = self.iteration
                # load gs
                model_gaussians.load_ply_sparse_gaussian(os.path.join(model_args.model_path,
                                                                str(0),
                                                                str(block_id),
                                                                "iteration_" + str(loaded_iter),
                                                                "point_cloud.ply"))
                model_gaussians.load_mlp_checkpoints(os.path.join(model_args.model_path,
                                                                str(0),
                                                                str(block_id),
                                                                "iteration_" + str(loaded_iter)), mode = 'unite')
                model_gaussians.eval()
            model_dict[block_id] = model_gaussians
        return model_dict

    def multiprocessLoadGSModel(self, block_id_list):
        # 删除历史block模型
        with self.thread_lock:
            keys_to_remove = []
            for key in self.static_gaussian_model_dict.keys():
                if key not in block_id_list:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.static_gaussian_model_dict[key]
            torch.cuda.empty_cache()
        # 加载新block模型
        for block_id in block_id_list:
            if block_id not in self.block_area_info: continue # model不存在的block id 
            if block_id in self.static_gaussian_model_dict: continue # 加载过了不重复加载
            with torch.no_grad():
                model_gaussians = GaussianModel(self.model_args.feat_dim, self.model_args.n_offsets, self.model_args.voxel_size, self.model_args.update_depth, self.model_args.update_init_factor, self.model_args.update_hierachy_factor, self.model_args.use_feat_bank, 
                                        self.model_args.appearance_dim, self.model_args.ratio, self.model_args.add_opacity_dist, self.model_args.add_cov_dist, self.model_args.add_color_dist, self.model_args.color_channel)
                loaded_iter = 4000 # default 
                if self.iteration == -1:
                    loaded_iter = searchForMaxIteration(os.path.join(self.model_args.model_path,"0", str(block_id)))
                else:
                    loaded_iter = self.iteration
                # load gs
                model_gaussians.load_ply_sparse_gaussian(os.path.join(self.model_args.model_path,
                                                                str(0),
                                                                str(block_id),
                                                                "iteration_" + str(loaded_iter),
                                                                "point_cloud.ply"))
                model_gaussians.load_mlp_checkpoints(os.path.join(self.model_args.model_path,
                                                                str(0),
                                                                str(block_id),
                                                                "iteration_" + str(loaded_iter)), mode = 'unite')
                model_gaussians.eval()

            with self.thread_lock:
                self.static_gaussian_model_dict[block_id] = model_gaussians


    def renderSimulation(self, sim_baselidar_to_world_pose, block_id, objs=None, save_name="test"):
        sim_world2sensor = np.linalg.inv(sim_baselidar_to_world_pose @ self.sensor2baselidar[self.lidar_name])
        sim_world_view_transform = torch.tensor(sim_world2sensor, dtype=torch.float32).transpose(0, 1).cuda()
        sim_lidar_center = sim_world_view_transform.inverse()[3, :3]
        sim_original_lidar_center = sim_lidar_center
        view = LidarInfo(uid=0, image_width=self.W_lidar, image_height=self.H_lidar, FoVx=1.0, FoVy=1.0,
                        beam_inclinations=self.beam_inclinations, camera_center=sim_lidar_center, lidar_center=sim_lidar_center, original_lidar_center=sim_original_lidar_center,
                        world_view_transform=sim_world_view_transform, full_proj_transform=sim_world_view_transform)

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            gs_model = None
            with self.thread_lock:
                gs_model = self.static_gaussian_model_dict[block_id]
            render_pkg = render(view, background, self.pipe_params, gs_model, self.model_args.max_depth, insert_objs = objs, retain_grad=False)

        rendering = render_pkg["render"]#.detach().cpu().numpy()
        render_intensity = rendering[0:1,...]
        render_raydrop = rendering[1:2,...]
        render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
        depth = render_pkg["depth"]

        render_intensity = render_intensity*render_raydrop_mask # 直接使用gt的raydrop mask
        depth = depth*render_raydrop_mask
        depth_numpy = depth.detach().cpu().numpy()
        intensity_numpy = render_intensity.detach().cpu().numpy()
        point_with_intensity = pano_to_lidar_with_intensities(depth_numpy[0, :, :],intensity_numpy[0], lidar_K=None, beam_inclinations=view.beam_inclinations.detach().cpu().numpy())
        # if True: # 密度滤波
        #     make_raydrop = filter_pcd(point_with_intensity[:,:3])
        #     point_with_intensity = point_with_intensity[make_raydrop]
        # convert to baselidar coordinate
        sensor2baselidar = self.sensor2baselidar[self.lidar_name]
        points = point_with_intensity[:,:3]
        distance = np.linalg.norm(points[...,:3], axis=1).reshape(points.shape[0],1)
        
        points = (np.pad(points[...,:3], ((0,0),(0, 1)), constant_values=1) @ sensor2baselidar.T)[:,:3]  
        point_with_intensity[:,:3] = points
        
        distance_mask = distance > 3
        distance_mask = distance_mask.reshape(distance.shape[0])
        point_with_intensity = point_with_intensity[distance_mask]
        distance = distance[distance_mask]
        # np.savetxt(os.path.join(self.save_path, '{}_{}'.format(self.lidar_name, save_name) + ".txt"), point_with_intensity, fmt='%.4f', comments='')
        data = get_pcdi(point_with_intensity, self.model_args.sensorid, distance)
        
        return data

if __name__ == "__main__":
    if False:
        debugRenderOriginView(True)
    else:
        source_path = '/mnt_gx/usr/lansheng/regroup_92171_all' #
        log = 'None' #
        sensorid = 0# 1 / 3 / 4
        iterations = -1
        model_path = '/mnt_gx/usr/lansheng/workspace/LiDAR-2DGS-dynamic-debug/LiDAR_GS/outputs/regroup_92171_all/TOP'
        save_name = "test1223_time"
        sensor2baselidar, baselidar2body, frontcam2baselidar= load_lidar_extrinsics(source_path) if log == 'None' else load_lidar_extrinsics(source_path + '/car_cfgs/' + log)
        print(sensor2baselidar)

        debug_b2w = True
        if debug_b2w:
            sim_baselidar_to_world_pose = debugGetGTPose()  
            # baselidar_to_world_pose = body_to_world_pose @ baselidar2body
        else:
            sim_baselidar_to_world_pose = None

        debug_insert_obj = False
        if debug_insert_obj:
            objs = [] # 以列表形式输入 支持多obj渲染
            sim_pose = np.eye(4)
            sim_pose[0, 3] = -248
            sim_pose[1, 3] = -329
            sim_pose[2, 3] = -1.7
            theta = -np.pi / 6 # 顺时针30度
            rotation_matrix = np.array([
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1]
            ])
            sim_pose[0:3, 0:3] = (sim_baselidar_to_world_pose[0:3,0:3] @ sim_pose[0:3, 0:3]) @ rotation_matrix
            path = "/mnt_gx/usr/lansheng/Gobj/chain/cone_chain_002_1/points3D.ply"
            objs.append(loadStaticObj(path, sim_pose=sim_pose))
        else:
            objs = None

        # 判断所给的pose是在哪个block里（长序列会分block训练）
        block_info_path = "/mnt_gx/usr/lansheng/workspace/LiDAR-2DGS-dynamic-debug/LiDAR_GS/outputs/regroup_92171_all/TOP"
        block_info_with_extend, block_info_without_extend, block_area_without_extend, block_height = getBlockInfo(block_info_path)
        if log == 'None': # Paralane
            init_cam_front_pose = np.loadtxt(os.path.join(model_path, "para_lane_init_pos.txt"), dtype=float, delimiter=',')
            sim_frontcam2world = init_cam_front_pose @ sim_baselidar_to_world_pose @ frontcam2baselidar
            block_id = judgeWhichBlock(sim_frontcam2world, block_area_without_extend, log=log)
        else:
            block_id = judgeWhichBlock(sim_baselidar_to_world_pose, block_area_without_extend)
        gs_sim_interface = LidarSimOnline(source_path, log, sensorid, sensor2baselidar, model_path, iterations, block_area_without_extend)

        if block_id != gs_sim_interface.curr_block_id:
            load_block_id = [str(int(block_id)-1), block_id, str(int(block_id)+1),\
                            str(int(block_id) - block_height), str(int(block_id) - block_height + 1), str(int(block_id) - block_height - 1),\
                            str(int(block_id) + block_height), str(int(block_id) + block_height + 1), str(int(block_id) + block_height - 1)]  # 该逻辑暂时只适配paralane的blockid TODO
            gs_loader_thread = Thread(target=gs_sim_interface.multiprocessLoadGSModel, args=(load_block_id, ))
            gs_loader_thread.daemon = True  # 将线程设置为守护线程主线程结束，自动关闭子线程
            gs_loader_thread.start()
        gs_sim_interface.curr_block_id = block_id
        while gs_sim_interface.curr_block_id not in gs_sim_interface.static_gaussian_model_dict.keys(): # 主线程得等待一定把模型加载进来
            time.sleep(0.1)# 等待100ms

        start_time = time.time()
        gs_sim_interface.renderSimulation(sim_baselidar_to_world_pose=sim_baselidar_to_world_pose, block_id=block_id, objs=objs, save_name=save_name)
        end_time = time.time()
        print("cost time:", end_time-start_time)
