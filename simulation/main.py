import gserver
import argparse
import sys
import os
import time

from utils.data_partition_utils import getBlockInfo, judgeWhichBlock

sys.path.append('../')

from simulation.lidar_sim_online import LidarSimOnline, load_lidar_extrinsics
from utils.obj_utils import loadStaticObj
import numpy as np
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Launch camera simulation")
    parser.add_argument('--sim_params')
    parser.add_argument('--gs_data_path')
    parser.add_argument('--gs_model_path')
    args = parser.parse_args()
    
    source_path = args.gs_data_path
    model_path = args.gs_model_path
    model_path_top =  os.path.join(model_path, "TOP") # '/data/lidar_model/regroup_92171_all/TOP/'
    model_path_back = os.path.join(model_path, "BACK") # '/data/lidar_model/regroup_92171_all/BACK/'
    model_path_left = os.path.join(model_path, "LEFT") # '/data/lidar_model/regroup_92171_all/LEFT/'
    model_path_right = os.path.join(model_path, "RIGHT") # '/data/lidar_model/regroup_92171_all/RIGHT/'
    insert_model_path = os.path.join(source_path, 'point_cloud.ply')
    block_info_path = os.path.join(model_path, "TOP")
    log = 'None'
    iterations = -1
    
    sensor2baselidar, baselidar2body, frontcam2baselidar = load_lidar_extrinsics(source_path)
    block_info_with_extend, block_info_without_extend, block_area_without_extend = getBlockInfo(block_info_path)
    init_cam_front_pose = np.loadtxt(os.path.join(source_path, "para_lane_init_pos.txt"), dtype=float, delimiter=',')

    
    server = gserver.GaussianSplattingServer("gs_lidar", os.path.join(source_path, "frame/frames_basic.yaml"))    
    gs_sim_interface_top = LidarSimOnline(source_path, log, 0, sensor2baselidar, model_path_top, iterations, block_area_without_extend) #  ,block_info
    gs_sim_interface_back = LidarSimOnline(source_path, log, 1, sensor2baselidar, model_path_back, iterations, block_area_without_extend) #  ,block_info
    gs_sim_interface_left = LidarSimOnline(source_path, log, 3, sensor2baselidar, model_path_left, iterations, block_area_without_extend) #  ,block_info
    gs_sim_interface_right = LidarSimOnline(source_path, log, 4, sensor2baselidar, model_path_right, iterations, block_area_without_extend) #  ,block_info
    
    logger.info("lidar is wait for localization...")
    while not server.is_finish():
        if not server.is_ready():
            time.sleep(0.1)
            continue
        
        ego_pose = server.get_ego_pose()
        agent_poses = server.get_agent_poses()
        
        if len(agent_poses) > 0:
            objs = []
            for agent_pose in agent_poses:
                objs.append(loadStaticObj(insert_model_path, sim_pose=agent_pose))
        else:
            objs = []
            
        sim_frontcam2world = init_cam_front_pose @ ((ego_pose @ baselidar2body) @frontcam2baselidar)
        block_id = judgeWhichBlock(sim_frontcam2world, block_area_without_extend)
        logger.info(f"block_id is {block_id}")
            
        top = gs_sim_interface_top.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        back = gs_sim_interface_back.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        left = gs_sim_interface_left.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        right = gs_sim_interface_right.renderSimulation(ego_pose @ baselidar2body, block_id, objs=objs)
        
        logger.info("send ROBO_POINTCLOUDS...")
        server.write_lidar("ROBO_POINTCLOUDS", {"ROBO_TOP": top, "ROBO_BACK": back, "ROBO_LEFT": left, "ROBO_RIGHT": right})
        
    logger.info("close lidar node...") 

if __name__ == "__main__":
    main()