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
import pygame
import pyrender

from typing import NamedTuple


from scipy.spatial.transform import Rotation


def update_position(matrix, direction, step_size=1):
    R = matrix[:3, :3]
    euler_angles = Rotation.from_matrix(R).as_euler('zyx', degrees=False)
    heading = euler_angles[1]
    position = matrix[:3, 3]

    roll_direction = R[:, 0]

    if direction == 'A':
        position -= roll_direction * step_size
    elif direction == 'D':
        position += roll_direction * step_size
    elif direction == 'W':
        position[0] -= step_size*np.cos(heading)
        position[1] -= step_size*np.sin(heading)
    elif direction == 'S':
        position[0] += step_size*np.cos(heading)
        position[1] += step_size*np.sin(heading)

    matrix[:3, 3] = position

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Matrix Control with WASD")
    source_path = '/data/regroup_92171_all_new/regroup_92171_all' #
    log = 'None' #
    sensorid = 0 # 1 / 3 / 4 ["TOP", "BACK", "LEFT", "RIGHT"]
    iterations = -1
    model_path = '/data/lidar_model/regroup_92171_all/RIGHT/'
    save_name = "test1"
    sensor2baselidar, baselidar2body= load_lidar_extrinsics(source_path)

    init_pose = np.load("/home/administrator/GStudio/simulation/init_body2world.npy")
    init_pose = debugGetGTPose() # init_pose @ baselidar2body

    debug_insert_obj = True
    if debug_insert_obj:
        objs = [] # 以列表形式输入 支持多obj渲染
        sim_pose = np.eye(4)
        sim_pose[0, 3] = -249
        sim_pose[1, 3] = -329
        sim_pose[2, 3] = -1
        path = "/home/administrator/LiDAR_GS/point_cloud.ply"
        objs.append(loadStaticObj(path, sim_pose=sim_pose))
    else:
        objs = None
        
    # 判断所给的pose是在哪个block里（长序列会分block训练）
    server = gserver.GaussianSplattingServer("gs_lidar", "/data/regroup_92171_all_new/regroup_92171_all/frame/frames_basic.yaml")
    block_info_path = "/home/administrator/GStudio/cross_lane_eval_output_track"
    block_info_with_extend, block_info_without_extend, block_area_without_extend, __ = getBlockInfo(block_info_path)
    block_id = judgeWhichBlock(init_pose, block_area_without_extend)
    block_info = block_info_without_extend[block_id]
    # gs_sim_interface_top = LidarSimOnline(source_path, log, 0, sensor2baselidar, model_path, iterations, block_id) #  ,block_info
    # gs_sim_interface_back = LidarSimOnline(source_path, log, 1, sensor2baselidar, model_path, iterations, block_id) #  ,block_info
    # gs_sim_interface_left = LidarSimOnline(source_path, log, 3, sensor2baselidar, model_path, iterations, block_id) #  ,block_info
    gs_sim_interface_right = LidarSimOnline(source_path, log, 4, sensor2baselidar, model_path, iterations, block_id) #  ,block_info
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    update_position(init_pose, 'W')
                elif event.key == pygame.K_s:
                    update_position(init_pose, 'S')
                elif event.key == pygame.K_a:
                    update_position(init_pose, 'A')
                elif event.key == pygame.K_d:
                    update_position(init_pose, 'D')
                elif event.key == pygame.K_ESCAPE:
                    running = False
        start = time.time()
        # top = gs_sim_interface_top.renderSimulation(init_pose, objs=objs, save_name=save_name)
        # back = gs_sim_interface_back.renderSimulation(init_pose, objs=objs, save_name=save_name)
        # left = gs_sim_interface_left.renderSimulation(init_pose, objs=objs, save_name=save_name)
        right = gs_sim_interface_right.renderSimulation(init_pose, objs=objs, save_name=save_name)
        print(time.time() - start)
        # {"ROBO_TOP": top, "ROBO_BACK": back, "ROBO_LEFT": left, "ROBO_RIGHT": right}
        server.write_lidar("ROBO_POINTCLOUDS", {"ROBO_RIGHT": right})
    pygame.quit()
    sys.exit()