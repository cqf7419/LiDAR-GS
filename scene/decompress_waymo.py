import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import math
import imageio
import argparse
import json
from tqdm import tqdm
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
from os.path import join, exists
import pickle as pkl
import pdb


camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT', 
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}
camera_names = ["FRONT", "FRONT_LEFT","FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT" ]

image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    # dataset_pb2.LaserName.FRONT: 'FRONT',
    # dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    # dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',s
    # dataset_pb2.LaserName.REAR: 'REAR',
}

opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def parse_seq_rawdata(seq_path, seq_save_dir, start_idx=None, end_idx=None):
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')
    os.makedirs(seq_save_dir, exist_ok=True)
    
    # set start and end timestep
    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())
    start_idx = start_idx or 0
    end_idx = end_idx or num_frames - 1


    print("Processing LiDAR data...")
                
    datafile = WaymoDataFileReader(seq_path)
    all_beam_inclinations=[]
    all_extrinsic = []
    all_frame_pose = []
    for frame_id, frame in enumerate(datafile):
        pts_3d = [] # LiDAR point cloud in world frame
        pts_2d = [] # LiDAR point cloud projection in camera [camera_name, w, h] 
        
        cam_proj_1 = []
        cam_proj_2 = []
        pts_3d_1 = []
        pts_3d_2 = []

        for laser_name, laser_name_str in laser_names_dict.items():
            laser = utils.get(frame.lasers, laser_name)
            laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
            # ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
            # ri_2, camera_projection_2, range_image_pose_2 = utils.parse_range_image_and_camera_projection(laser, second_response=True)

            #------------
            beam_inclinations = utils.compute_beam_inclinations(laser_calibration, 64)
            # beam_inclinations = np.flip(beam_inclinations)
            all_beam_inclinations.append(beam_inclinations[None,...])
            extrinsic = np.array(laser_calibration.extrinsic.transform).reshape(4,4)
            all_extrinsic.append(extrinsic[None,...])
            frame_pose = np.array(frame.pose.transform).reshape(4,4)
            all_frame_pose.append(frame_pose[None,...])
            #------------

            # # LiDAR spherical coordinate -> polar -> cartesian
            # pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
            # pcl_2, pcl_attr_2 = utils.project_to_pointcloud(frame, ri_2, camera_projection_2, range_image_pose_2, laser_calibration)

            # result_1 = np.concatenate((pcl, pcl_attr[:,1:2]), axis=1)
            # pts_3d_1.append(result_1)
            # result_2 = np.concatenate((pcl_2, pcl_attr_2[:,1:2]), axis=1)
            # pts_3d_2.append(result_2)


        # pts_3d_1 = np.concatenate(pts_3d_1, axis=0)
        # pts_3d_2 = np.concatenate(pts_3d_2, axis=0)
        # pts_3d = np.vstack([pts_3d_1, pts_3d_2])
        # np.savez(f'{seq_save_dir}/{str(frame_id).zfill(3)}.npz', data=pts_3d)

    all_beam_inclinations = np.concatenate(all_beam_inclinations, axis=0)
    all_extrinsic = np.concatenate(all_extrinsic, axis=0)
    all_frame_pose = np.concatenate(all_frame_pose, axis=0)
    print(all_beam_inclinations.shape)
    print(all_extrinsic.shape)
    print(all_frame_pose.shape)
    laser_calibrations_save_path = join(seq_save_dir, "laser_calibrations")
    os.makedirs(laser_calibrations_save_path, exist_ok = True)
    np.savez(f'{laser_calibrations_save_path}/laser_calibrations.npz', beam_inclinations=all_beam_inclinations, extrinsic=all_extrinsic, frame_pose=all_frame_pose)


                                            
    print("Processing LiDAR data done...")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--partition', type=int, required=False,
                       default="1")

    args = parse.parse_args()

    raw_data_dir = "/data1/public_dataset/waymo/waymo_v140/individual_files/training"
    target_root_dir = "/mnt_gx/lidar_data/datapublic_old/waymo_train"


    target_dir = join(target_root_dir, "pcds_new")
    os.makedirs(target_dir, exist_ok = True)

    laser_calibrations_dir = join(target_root_dir, "temp")
    os.makedirs(laser_calibrations_dir, exist_ok = True)

    case_names = os.listdir("/mnt_gx/lidar_data/datapublic_old/waymo_train/temp")

    selected_segment = "/mnt_gx/usr/lansheng/selected_waymo.txt"
    with open(selected_segment, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    i_range = int(len(case_names))
    cont=0
    for i in tqdm(range(i_range)):
        case_name = case_names[i]
        if case_name not in lines: continue
        # if os.path.exists(join(target_dir, case_name)): continue

        occ_path = os.path.join("/mnt_gx/lidar_data/datapublic_old/waymo_train/temp", case_name, "occ")
        if os.path.exists(occ_path) == False:
            os.rmdir(os.path.join("/mnt_gx/lidar_data/datapublic_old/waymo_train/temp", case_name))
            continue
        case_raw_info_path = join(raw_data_dir, case_name + ".tfrecord")
        # parse_seq_rawdata(case_raw_info_path,  join(target_dir, case_name) )
        parse_seq_rawdata(case_raw_info_path,  join(laser_calibrations_dir, case_name) )
        cont+=1
        print(cont)

# # use
# for case_name in os.listdir(pcds_dir):
#     meta_infos = pkl.load(open(join(target_root_dir, "meta_infos_org", case_name + ".pkl"), "rb"))
#     frames = meta_infos["frames"]

#     pointcloud_path = join(target_dir, case_name, "pointcloud.npz")
#     pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
#     pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  
#     print(len(frames))
#     print(len(pts2d_dict))
#     assert(len(frames) == len(pts2d_dict))

#     # print(frames[0]["path"])
#     for i in range(len(frames)):
#         pcds = np.load(join(target_root_dir, frames[i]["path"]["pcd"]))["data"]
#         print("point cloud shape in pcds : ",pcds.shape)
#         print("parsed pointcloud shape : ",pts3d_dict[i].shape)
#         print("parsed camera projection shape : ",pts2d_dict[i].shape)

#         print("first raw from pcds frame 0 : ",  pcds[0])
#         print("first raw from parsed frame 0 : ",pts3d_dict[i][0])
#         np.savetxt("/mnt_gx/ziqian_data/old_pcd.txt", pcds)
#         np.savetxt("/mnt_gx/ziqian_data/new_pcd.txt", pts3d_dict[i])

#         break
#     break




#### read beam ##########
# import tensorflow as tf
# from waymo_open_dataset import dataset_pb2
# import os
# import numpy as np

# selected_segment = "/mnt_gx/usr/lansheng/selected_waymo.txt"
# with open(selected_segment, 'r') as f:
#     lines = f.readlines()
# lines = [line.strip() for line in lines]

# root_path = "/mnt_gx/lidar_data/datapublic_old/waymo_train/temp"
# for segment in os.listdir(root_path):
#     if segment not in lines: continue
#     beam_path = os.path.join(root_path, segment, "beam_inclinations")
#     if os.path.exists(beam_path) == True:
#         continue

#     occ_path = os.path.join(root_path, segment, "occ")
#     if os.path.exists(occ_path) == False:
#         continue
#         #os.rmdir(os.path.join(root_path, segment))
        
#     pathname = os.path.join("/data1/public_dataset/waymo/waymo_v120/tfrecord_training", segment+".tfrecord")
#     if os.path.exists(pathname):
#         print(pathname)
#         dataset = tf.data.TFRecordDataset(pathname, compression_type='')
#         data0= None
#         for frame_idx, data in enumerate(dataset):
#             if frame_idx == 0: data0 = data

#         frame = dataset_pb2.Frame()
#         frame.ParseFromString(bytearray(data0.numpy()))
#         calibrations = sorted(frame.context.laser_calibrations,key=lambda c: c.name)
#         beam_inclinations=None
#         for c in calibrations:
#             if c.name != dataset_pb2.LaserName.TOP: continue
#             beam_inclinations = tf.constant(c.beam_inclinations)
#             beam_inclinations = list(beam_inclinations.numpy())
#             beam_inclinations = [f'{i:e}' for i in beam_inclinations]

#         save_path = "/mnt_gx/lidar_data/datapublic_old/waymo_train/temp/"+segment
#         save_path = os.path.join(save_path, "beam_inclinations")
#         os.makedirs(save_path, exist_ok=True)
#         np.save(os.path.join(save_path, "beam_inclinations.npy"), beam_inclinations)