import os
import sys
import pickle
import math
import json
import numpy as np
from collections import OrderedDict
from .colmap_data_io_utils import readExtrinsicsText, qvec2rotmat
from scipy.spatial.transform import Rotation

chunk_max_image = 150
chunk_min_image = 30

def getInfo(args):
    pkl_file_name = args.caseid + ".pkl"
    meta_file = os.path.join(args.source_path, "meta_infos", pkl_file_name)
    meta_info = pickle.load(open(meta_file, "rb"))
    print("Total frame number: ", len(meta_info["frames"]))
    
    max_x = -sys.maxsize
    max_y = -sys.maxsize
    min_x = sys.maxsize
    min_y = sys.maxsize
    time_with_pose = {}
    for idx, info in enumerate(meta_info["frames"]):
        if 'optimized_pose' in info: 
            lidar_to_world = info["optimized_pose"] 
        else: 
            lidar_to_world = info["lidar2world"] 
        min_x = min(min_x, float(lidar_to_world[0, 3]))
        min_y = min(min_y, float(lidar_to_world[1, 3]))
        max_x = max(max_x, float(lidar_to_world[0, 3]))
        max_y = max(max_y, float(lidar_to_world[1, 3]))
        time_with_pose[info["log_time_stamp"]] = (float(lidar_to_world[0, 3]), float(lidar_to_world[1, 3]))

    min_x = math.floor(min_x)
    min_y = math.floor(min_y)
    max_x = math.ceil(max_x)
    max_y = math.ceil(max_y)
    print("Delta x: ", max_x - min_x)
    print("Delta y: ", max_y - min_y)
    return time_with_pose, min_x, min_y, max_x, max_y


def dataPartition(args):
    if args.caseid != "None" and args.caseid != "pesudo":
        use_downsample = False
        chunk_size = 60 # 40
        expand_size = 10
        time_with_pose, min_x, min_y, max_x, max_y = getInfo(args)
    else:
        exit(1)

    block_id_with_rect = {}
    block_time_with_extend = {}
    block_time_without_extend = {}
    block_height = 0
    if len(time_with_pose) < chunk_max_image:   # The situation where the log is short
        block_time_with_extend[0] = []
        for time, pose in time_with_pose.items():
            block_time_with_extend[0].append(time)
        block_time_without_extend = block_time_with_extend
        block_id_with_rect[0] = [min_x, min_y, max_x, max_y]
    else:
        # obtain nose to tail block id
        block_id_with_extend_rect = {}
        block_height = math.ceil((max_y - min_y) / chunk_size)
        cols = 0
        for curr_min_x in range(min_x, max_x, chunk_size):
            cols += 1
            rows = 0
            for curr_min_y in range(min_y, max_y, chunk_size):
                curr_max_x = curr_min_x + chunk_size
                curr_max_y = curr_min_y + chunk_size

                rows += 1
                if cols % 2 == 0:
                    curr_block_id = cols * block_height - rows + 1
                else:
                    curr_block_id = (cols - 1) * block_height + rows

                block_id_with_rect[curr_block_id] = [curr_min_x, curr_min_y, curr_max_x, curr_max_y]
                block_id_with_extend_rect[curr_block_id] = [
                    curr_min_x - expand_size, curr_min_y - expand_size,
                    curr_max_x + expand_size, curr_max_y + expand_size]

        # correspond time with block id
        block_id_with_timelist = {}
        last_pos = np.array([0, 0])
        for timestamp, pos in time_with_pose.items():
            curr_x = float(pos[0])
            curr_y = float(pos[1])
            curr_pos = np.array([curr_x, curr_y])
            if use_downsample and np.linalg.norm(curr_pos - last_pos) < 0.1: # TODO This is a problem. When the car is stationary, the dynamic object motion will introduce bugs
                continue

            last_pos = curr_pos
            for block_id, rect in block_id_with_extend_rect.items():
                if curr_x > rect[0] and curr_x < rect[2] and curr_y > rect[1] and curr_y < rect[3]:
                    block_id_with_timelist.setdefault(block_id, []).append(timestamp)
        block_id_with_timelist = OrderedDict(sorted(block_id_with_timelist.items()))

        # merge and get train block
        last_key = list(block_id_with_timelist.keys())[-1]
        prev_values = []
        print("Train block info")
        for key, values in block_id_with_timelist.items():
            curr_value = values + prev_values
            if key != last_key:
                if len(curr_value) < chunk_min_image: 
                    prev_values += values  
                    continue

            if use_downsample and len(curr_value) > chunk_max_image * 1.5:  # TODO  掉头转弯？？这种原地操作不应该删掉 而且动态物体有些帧也会被跳掉！
                curr_value.sort()
                interval = max(int(len(curr_value) / chunk_max_image), 2)
                curr_value = [curr_value[i] for i in range(0, len(curr_value), interval)]

            curr_value = sorted(list(set(curr_value)))
            block_time_with_extend[key] = curr_value
            prev_values = []

        # remove invalid area
        new_block_id_with_rect = {}
        for block_id in block_id_with_rect.keys():
            if block_id in block_time_with_extend.keys():
                new_block_id_with_rect[block_id] = block_id_with_rect[block_id]
        block_id_with_rect = new_block_id_with_rect

        # get render block
        print("Render block info")
        for timestamp, pos in time_with_pose.items():
            pos_x = float(pos[0])
            pos_y = float(pos[1])
            distance = sys.maxsize
            result_block_id = -1

            for block_id, area in block_id_with_rect.items():
                min_x = float(area[0])
                min_y = float(area[1])
                max_x = float(area[2])
                max_y = float(area[3])
                mid_x = (min_x + max_x) / 2
                mid_y = (min_y + max_y) / 2

                curr_distance = (pos_x - mid_x) * (pos_x - mid_x) + (pos_y - mid_y) * (pos_y - mid_y)
                if curr_distance < distance:
                    distance = curr_distance
                    result_block_id = block_id

            block_time_without_extend.setdefault(result_block_id, []).append(timestamp)
        block_time_without_extend = OrderedDict(sorted(block_time_without_extend.items()))

    # save json files
    block_info = {'block_time_with_extend': block_time_with_extend, 'block_time_without_extend': block_time_without_extend,\
                'block_id_with_rect': block_id_with_rect, 'block_height': block_height}
    block_info_json = os.path.join(args.model_path, 'block_info.json')
    with open(block_info_json, 'w', encoding='utf-8') as file:
        json.dump(block_info, file, ensure_ascii=False, indent=4)
    return block_time_with_extend, block_time_without_extend

def getBlockInfo(root_model_path):    
    block_info_json = os.path.join(root_model_path, 'block_info.json')
    if os.path.exists(block_info_json):
        with open(block_info_json, "r", encoding="utf-8") as file:
            block_info = json.load(file)
        block_time_with_extend = block_info['block_time_with_extend']
        block_time_without_extend = block_info['block_time_without_extend']
        block_id_with_rect = block_info['block_id_with_rect']
        block_height = int(block_info['block_height'])
    else:
        info_json_file = os.path.join(root_model_path, 'block_time_with_extend.json')
        with open(info_json_file, "r", encoding="utf-8") as file:
            block_time_with_extend = json.load(file)
        info_json_file = os.path.join(root_model_path, 'block_time_without_extend.json')
        with open(info_json_file, "r", encoding="utf-8") as file:
            block_time_without_extend = json.load(file)
        info_json_file = os.path.join(root_model_path, 'block_area_without_extend.json')
        with open(info_json_file, "r", encoding="utf-8") as file:
            block_id_with_rect = json.load(file)
        block_height = 0
        
    return block_time_with_extend, block_time_without_extend, block_id_with_rect, block_height


def judgeWhichBlock(sim_baselidar_to_world_pose, block_id_with_rect, log=None):
    pos_x = sim_baselidar_to_world_pose[0, 3]
    if log is None or log == 'None':  # Paralane
        pos_y = sim_baselidar_to_world_pose[2, 3]
    else:
        pos_y = sim_baselidar_to_world_pose[1, 3]
    result_block_id = -1
    for block_id, area in block_id_with_rect.items():
        min_x = float(area[0])
        min_y = float(area[1])
        max_x = float(area[2])
        max_y = float(area[3])    
        if pos_x >= min_x and pos_x <= max_x and pos_y >= min_y and pos_y <= max_y:
            result_block_id = block_id
            break
    if result_block_id != -1:
        return result_block_id
    
    distance = sys.maxsize
    for block_id, area in block_id_with_rect.items():
        min_x = float(area[0])
        min_y = float(area[1])
        max_x = float(area[2])
        max_y = float(area[3])
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        curr_distance = (pos_x - mid_x) * (pos_x - mid_x) + (pos_y - mid_y) * (pos_y - mid_y)
        if curr_distance < distance:
            distance = curr_distance
            result_block_id = block_id
    return result_block_id