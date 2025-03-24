import numpy as np
from pathlib import Path
import json
import torch
import pandas as pd
import os
import pickle
import sys
import open3d as o3d
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.lidar_utils import lidar_to_pano_with_intensities, cal_beam_inclinations, load_yaml_str, load_extrinsics, pano_to_lidar, pano_to_lidar_with_intensities
from extern.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import imageio
import pdb

def getTempBaselidar2world(args, train_frame_times):
    root_path = args.source_path
    case = args.caseid
    meta_info_path = root_path + "/meta_infos/" + case+ ".pkl"
    frames_data = None
    with open(meta_info_path, 'rb') as pickle_file:
        all_data = pickle.load(pickle_file)
        frames_data = all_data['frames']
    
    max_frame_num = len(train_frame_times)

    baselidar2world = []
    count_ind = 0
    for frame in frames_data:
        if count_ind == max_frame_num: break
        if frame['log_time_stamp'] != int(train_frame_times[count_ind]): continue
        l2w = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
        baselidar2world.append(l2w)
        count_ind += 1

    if count_ind < max_frame_num: raise ValueError("Missing Frame or Abnormal Loading.")
    
    return  baselidar2world

class Waymo_Dataloader:
    '''
    eg. 
        root_path =  "/mnt_gx/lidar_data/datapublic_old/waymo_train"
        caseid = "segment-15832924468527961_1564_160_1584_160_with_camera_labels"
    '''
    def __init__(self, args, train=True, train_frame_times=None, dtype=np.float32):
        self.train = train
        self.root_path = args.source_path
        self.case = args.caseid
        self.meta_info_path = self.root_path + "/meta_infos/" + self.case+ ".pkl"
        
        # read meta_info
        self.start_frame = 0 # 从第0帧开始  留着这个接口 目前给的case都是50帧不用特殊处理
        self.frames_data = None
        with open(self.meta_info_path, 'rb') as pickle_file:
            all_data = pickle.load(pickle_file)
            self.frames_data = all_data['frames']
            print("[ Info ] this case have {} frames totally".format(len(self.frames_data)))

        self.beam_inclinations = np.load(os.path.join(self.root_path, "temp", self.case, "beam_inclinations", "beam_inclinations.npy")).astype(np.float32).tolist() #TODO cal_beam_inclinations()
        self.laser_calibrations = np.load(os.path.join(self.root_path, "laser_calibrations", self.case, "laser_calibrations/laser_calibrations.npz"))
        self.extrinsic = self.laser_calibrations['extrinsic'][0] # laser_to_vehicle
        print("beam_inclinations", self.beam_inclinations)

        self.W_lidar = int(2650)
        self.H_lidar = int(64)
        self.train_frame_times = train_frame_times
        self.max_frame_num = len(train_frame_times)
        self.max_depth = args.max_depth # helios 5515 的最大深度是150 看情况调整，大于100的点也很少基本，误差更大
        print("max_depth", self.max_depth)
        self.aabb_min = np.ones(3, dtype=np.float32)*(10000000)
        self.aabb_max = np.ones(3, dtype=np.float32)*(-10000000)

        self.sensor2baselidar = dict() # 记录每个lidar到toplidar的变换矩阵


        self.pcds = [] # 原始每帧点云
        self.pcds_label = [] # onemodle的语义结果 ==10为地面 ==0为背景 
        self.l2ws = [] # 每帧的l2w
        self.timestep_2_frameid = dict() # 通过timsestep查询对应训练的帧的id _ 0 to 50
        self.frameid_2_timestep = [] # 通过frame id 反查询对应训练帧的timestep
        self.baselidar2world = []
        self.pcds, self.l2ws = self.load_pcds(frames=self.frames_data, train_frame_times=self.train_frame_times, frame_num=self.max_frame_num)

        self.obj_id_list = self.load_dynamic_obj_id_list() 

        self.obj_frames_id = dict()  # 通过obj_id 查询实例出现在哪几帧（列表）(frame id : 0-50)
        self.obj_pcd = dict() # 通过obj_id 查询实例的拼接后的完整的pcd
        self.obj_o2l = dict() # 通过obj_id 和对应那一帧的frame id查询实例的o2l ， 字典嵌套了一个字典
        if self.obj_id_list is not None:
            for obj_id in self.obj_id_list:
                self.obj_pcd[str(obj_id)] = self.load_dynamic_pcd(str(obj_id))
        if train:
            self.static_pcd = self.load_static_scene(use_pcd=True)
        self.range_views, self.masks = self.load_rangeview(self.H_lidar,self.W_lidar)


    def load_static_scene(self, use_pcd = True):
        '''
        waymo的数据没有拿到每帧的点云的动态语义信息，因此直接加载静态拼接结果， 可能会比较耗时
        '''  

        static_pcd_path = os.path.join(self.root_path, "recon_related", self.case, "static_recon_voxels.pcd")
        static_pcd = np.array(o3d.io.read_point_cloud(static_pcd_path).points, dtype=np.float32)
        within_aabb = np.all((static_pcd >= self.aabb_min) & (static_pcd <= self.aabb_max), axis=1)
        final_static_pcd = static_pcd[within_aabb]

        return final_static_pcd

    def load_dynamic_obj_id_list(self):
        '''
        从文件夹加载动态物体ID
        return: 一个存储ID的list
        '''
        dynamic_obj_path = self.root_path + '/temp/' + self.case + '/occ/preproc/dynamic/objects'
        directory_path = Path(dynamic_obj_path)
        obj_id_list = None
        if directory_path.exists():
            obj_id_list = [int(item.name) for item in directory_path.iterdir() if item.is_dir()]
        
        return obj_id_list

    def load_dynamic_pcd(self, object_id):
        '''
        加载对应obj_id的各帧点云拼成一个完整obj的pcd, 同时每个obj会在哪几帧出现也会在这里处理和存储
        reutrn : obj拼接后的点云 
        '''
        cor_frames = []
        obj_b2ls = dict() 

        dynamic_obj_path = self.root_path + '/temp/' + self.case + '/occ/preproc/dynamic/objects/' + object_id
        info_json = dynamic_obj_path + '/info.json'
        # print("[ Debug ]: info_json:", info_json)
        data = None
        with open(info_json, 'r') as file:
            datas = json.load(file)
            data = datas["tracks"]

        pcd_path = dynamic_obj_path + '/vertices.txt' # stitch.pcd已经拼接好的的点云  vertice.txt是过滤过的 但不是每个都有 
        # print("[ Debug ]: pcd_path:", pcd_path)
        if os.path.exists(pcd_path):
            obj_pcd = np.loadtxt(pcd_path).astype(np.float32)
        else:
            pcd_path = dynamic_obj_path + '/stitch.pcd'
            obj_pcd = np.array(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)
        for key, value in data.items():
            b2l = np.array(value["T_b2l"], dtype=np.float32).reshape(4,4)
            if int(key) < 200:   # obj存在两种命名格式 前后版本需要兼容以下
                key = key.zfill(3)
            if key in self.timestep_2_frameid:  # TODO 
                cor_frames.append(self.timestep_2_frameid[key])
                obj_b2ls[str(self.timestep_2_frameid[key])] = b2l
            else:
                if key > self.frameid_2_timestep[-1]: break # 超过训练的timestep了 
                else: continue # 可能存在跳帧
        self.obj_frames_id.update({object_id : cor_frames}) 
        self.obj_o2l.update({object_id : obj_b2ls})

        return obj_pcd

    def load_pcds(self, frames, train_frame_times, frame_num=50, selected_sensor=0):
        '''
        加载每帧点云 作为gt 以及计算边界 其中selected_sensor = 0 / 1 / 3 / 4 分别代表 ROTOTOP, ROBO_BACK, ROBO_LEFT_FRONT, ROBO_RIGHT_FRONT
        return : 原始每帧点云_baselidar系 
        '''
        pcds = []
        pcds_label = []
        l2ws = []
        count_ind = 0
        for f_id, frame in enumerate(frames):
            if count_ind == frame_num: break
            if frame['log_time_stamp'] != int(train_frame_times[count_ind]): continue
            # 这里的 lidar2world 是baselidar系
            l2w = np.array(frame['lidar2world'])
            self.baselidar2world.append(l2w)
            sl2w = l2w @ self.extrinsic # vehicle2wordl @ lidar2vehicle
            l2ws.append(sl2w)

            pcd_path = self.root_path + "/" + frame["path"]["pcd"]
            pcd_path = pcd_path.replace("pcds","pcds_new")


            pcd = np.load(pcd_path)["data"]#.reshape((-1,7))


            pcd_world = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
            aabb_min = np.min(pcd_world, axis=0)
            aabb_max = np.max(pcd_world, axis=0)
            # print("[ debug ] aabb:", aabb_min, aabb_max)
            self.aabb_min = np.minimum(self.aabb_min, aabb_min)
            self.aabb_max = np.maximum(self.aabb_max, aabb_max)


            vehicle_to_laser = np.linalg.inv(self.extrinsic)
            sensor_pcd = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ vehicle_to_laser.T)[:,:3]      
            pcd[:,:3] = sensor_pcd[:,:3]
            pcd[:,3] /= 30.0  
            pcds.append(pcd)

            if self.judgeLoadtype(): # 判断记录时间戳还是lidar文件名
                timestep = pcd_path.split('/')[-1].split('.')[0]
            else:
                timestep = str(frame["log_time_stamp"])
            self.timestep_2_frameid.update({timestep : count_ind})
            self.frameid_2_timestep.append(timestep)
            count_ind += 1

        if count_ind < frame_num: raise ValueError("Missing Frame or Abnormal Loading.")
        return pcds, l2ws#, pcds_label
        
    def judgeLoadtype(self):
        dynamic_path = os.path.join(self.root_path, "temp", self.case, "occ/preproc/dynamic/objects")
        first_obj = os.listdir(dynamic_path)[0]
        for file in os.listdir(os.path.join(dynamic_path, first_obj)):
            base_name, extension = os.path.splitext(file)
            if extension.lower() == '.pcd':
                if int(base_name) <= 200:
                    return True
                else:
                    return False
        return False

        
    def load_rangeview(self,H_lidar,W_lidar):
        '''
        return [H,W,3] rangeiew
        '''
        range_views = []
        masks = []
        for frame_idx in range(self.max_frame_num):
            pano, intensities, mask = lidar_to_pano_with_intensities(  
                    local_points_with_intensities=self.pcds[frame_idx],
                    lidar_H=H_lidar,
                    lidar_W=W_lidar,
                    beam_inclinations=self.beam_inclinations,
                    max_depth=self.max_depth, 
                    #ground = (self.pcds_label[frame_idx]==10),
                    #is_correction = True,
                    #sensor_id = self.selected_sensor,
                    #s2b = self.sensor2baselidar[self.lidar_map[self.selected_sensor]]
                )
            # if frame_idx==0:  ## test code
            #     imageio.imwrite("./rangeview.png", (pano/80*255).astype(np.uint8))
            #     temp_pcd = pano_to_lidar(pano, beam_inclinations=self.beam_inclinations)
            #     np.savetxt("./temp_pcd.txt", temp_pcd, fmt='%.4f', comments='')
            #     np.savetxt("./temp_pcd_gt.txt", self.pcds[frame_idx], fmt='%.4f', comments='')
            #     world_pcd = (np.pad(self.pcds[frame_idx][...,:3], ((0,0),(0, 1)), constant_values=1) @ self.l2ws[frame_idx].T)[:,:3]    
            #     np.savetxt("./temp_pcd_gt_world.txt", world_pcd, fmt='%.4f', comments='')  
            range_view = np.zeros((H_lidar, W_lidar, 3))
            range_view[:, :, 1] = intensities
            range_view[:, :, 2] = pano
            ray_drop = np.where(range_view.reshape(-1, 3)[:, 2] <= 0.0, 0.0,
                                1.0).reshape(H_lidar, W_lidar, 1)
            image_lidar = np.concatenate(
                [
                    ray_drop,
                    np.clip(range_view[:, :, 1, None], 0, 1),
                    range_view[:, :, 2, None]
                ],
                axis=-1,
            )
            range_views.append(image_lidar)
            masks.append(mask)
        return range_views,masks

    def get_frames_nums(self):
        return self.max_frame_num

    def get_static_pcd(self):
        return self.static_pcd
    
    def get_obj_pcd(self, object_id):
        return self.obj_pcd[str(object_id)]

    def get_rangeview(self,frame_idx):
        '''
        return [H,W,3] numpy 
        '''
        return self.range_views[frame_idx]
    def get_mask(self,frame_idx):
        '''
        return [H,W,3] numpy 
        '''
        return self.masks[frame_idx]

    def getlidar2world(self):
        '''
        return list 所有帧的sensorlidar2world
        '''
        return self.l2ws
    
    def getlidar2world_newcar(self):
        '''
        return 车型迁移后新的外参
        '''
        return self.newcar_l2ws

    def get_obj2lidar(self,occurred_frame_idx,object_id, newcar_render=None):        
        '''
        return [4,4] numpy 返回obj2lidar的矩阵
        baselidar2sensor @ obj2baselidar
        '''
        vehicle_to_laser = np.linalg.inv(self.extrinsic)
        return vehicle_to_laser @ self.obj_o2l[str(object_id)][str(occurred_frame_idx)] # TODO 
        # return self.obj_o2l[str(object_id)][str(occurred_frame_idx)]

    def get_sensor2baselidar(self,sensorid):
        '''
        字典返回每个雷达系到baselidar系的矩阵
        '''
        return self.sensor2baselidar[self.lidar_map[sensorid]]
        
    def get_sensor2baselidar_newcar(self,sensorid):
        return self.new_sensor2baselidar[self.lidar_map[sensorid]]

    def get_obj_frames(self,object_id):
        '''
        列表返回obj出现的帧
        '''
        return self.obj_frames_id[str(object_id)]

    def get_dynamic_obj_id_list(self):
        '''
        列表返回移动的obj的id
        '''
        return self.obj_id_list

    def get_beam_inclination(self):
        return self.beam_inclinations

    def get_lidar_res(self):
        return self.W_lidar, self.H_lidar

