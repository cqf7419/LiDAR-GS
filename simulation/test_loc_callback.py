from scipy.spatial import cKDTree
import gserver
import os
import time
import sys
import subprocess

sys.path.append('../')
import numpy as np

from simulation.lidar_sim_online import LidarSimOnline, debugGetGTPose, load_lidar_extrinsics
from utils.obj_utils import loadStaticObj
from utils.data_partition_utils import judgeWhichBlock, getBlockInfo
import pyrender

from typing import NamedTuple


from scipy.spatial.transform import Rotation


def load_pcd_pose():
    all_pcd_pose = []
    cases = ["GT2-00006_20240614143004_20240614143030_8", "GT2-00006_20240614150446_20240614150512_9", "GT2-00006_20240614154135_20240614154201_10"]
    for case in cases:
        file_path = "/data/regroup_92171_all_new/regroup_92171_all" + "/temp/"+ case +"/annotation/single_frame_cloud_poses.txt"
        pose = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                pcd_name = data[1]

                quaternion = [
                    float(data[4]),
                    float(data[5]),
                    float(data[6]),
                    float(data[7]),
                ]
                rot_matrix = Rotation.from_quat(quaternion).as_matrix()

                trans_matrix = np.eye(4)
                trans_matrix[:3, :3] = rot_matrix
                trans_matrix[0, 3] = data[8]
                trans_matrix[1, 3] = data[9]
                trans_matrix[2, 3] = data[10]
                pose.append(trans_matrix)
        all_pcd_pose.append(pose)
    print("pcd pose shape",len(all_pcd_pose),len(all_pcd_pose[0]))
    return all_pcd_pose

if __name__ == "__main__":
    source_path = '/data/gs_data/big_scene/regroup_92171_all_new/regroup_92171_all' #
    log = 'None' #
    sensorid = 0 # 1 / 3 / 4 ["TOP", "BACK", "LEFT", "RIGHT"]
    iterations = -1
    model_path_top = '/data/gs_data/big_scene/lidar_model/lidar_model/regroup_92171_all/TOP/'
    model_path_back = '/data/gs_data/big_scene/lidar_model/lidar_model/regroup_92171_all/BACK/'
    model_path_left = '/data/gs_data/big_scene/lidar_model/lidar_model/regroup_92171_all/LEFT/'
    model_path_right = '/data/gs_data/big_scene/lidar_model/lidar_model/regroup_92171_all/RIGHT/'
    save_name = "test1"
    sensor2baselidar, baselidar2body, frontcam2baselidar = load_lidar_extrinsics(source_path)

    objs = [] # 以列表形式输入 支持多obj渲染
    sim_pose = np.eye(4)
    sim_pose[0, 3] = -249
    sim_pose[1, 3] = -329
    sim_pose[2, 3] = -1
    path = "/data/gs_data/big_scene/lidar_model/lidar_model/regroup_92171_all/point_cloud.ply"
    objs.append(loadStaticObj(path, sim_pose=sim_pose))
    # 判断所给的pose是在哪个block里（长序列会分block训练）
    server = gserver.GaussianSplattingServer("gs_lidar", "/data/regroup_92171_all_new/regroup_92171_all/frame/frames_basic.yaml")
    block_info_path = "/data/gs_data/big_scene/lidar_model/lidar_model/regroup_92171_all/TOP/"
    block_info_with_extend, block_info_without_extend, block_area_without_extend = getBlockInfo(block_info_path)
    gs_sim_interface_top = LidarSimOnline(source_path, log, 0, sensor2baselidar, model_path_top, iterations, block_area_without_extend) #  ,block_info
    gs_sim_interface_back = LidarSimOnline(source_path, log, 1, sensor2baselidar, model_path_back, iterations, block_area_without_extend) #  ,block_info
    gs_sim_interface_left = LidarSimOnline(source_path, log, 3, sensor2baselidar, model_path_left, iterations, block_area_without_extend) #  ,block_info
    gs_sim_interface_right = LidarSimOnline(source_path, log, 4, sensor2baselidar, model_path_right, iterations, block_area_without_extend) #  ,block_info
    init_cam_front_pose = np.loadtxt(os.path.join("/data/gs_data/big_scene/cross_lane_46dffbd5/cross_lane_data_1223_1", "para_lane_init_pos.txt"), dtype=float, delimiter=',')
    
    running = True
    while running:
        if not server.is_ready():
            print("wait for ready")
            continue
        
        ego_pose = server.get_ego_pose()
                
        print(ego_pose)
        agent_poses = server.get_agent_poses()
        
        if len(agent_poses) > 0:
            objs = []
            for agent_pose in agent_poses:
                objs.append(loadStaticObj(path, sim_pose=agent_pose))
        else:
            objs = []
            
        start = time.time()

        sim_frontcam2world = init_cam_front_pose @ ((ego_pose @ baselidar2body) @frontcam2baselidar)
        block_id = judgeWhichBlock(sim_frontcam2world, block_area_without_extend)
        print(block_id)

        top = gs_sim_interface_top.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        back = gs_sim_interface_back.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        left = gs_sim_interface_left.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        right = gs_sim_interface_right.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        
        print(time.time() - start)
        # pcd = np.load("/data/regroup_92171_all_new/regroup_92171_all/pcds/GT2-00006_20240614154135_20240614154201_10/1718350895899616.npz")["data"]
        # zero_matrix = np.zeros((pcd.shape[0], 2))
        # pcd = np.concatenate([pcd, zero_matrix], axis=1)
        
        # {"ROBO_TOP": top, "ROBO_BACK": back, "ROBO_LEFT": left, "ROBO_RIGHT": right}
        server.write_lidar("ROBO_POINTCLOUDS", {"ROBO_TOP": top, "ROBO_BACK": back, "ROBO_LEFT": left, "ROBO_RIGHT": right})