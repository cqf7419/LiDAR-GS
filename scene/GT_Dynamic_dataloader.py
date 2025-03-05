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

class GT_Dataloader:
    '''
    eg. 
        root_path =  /mnt_gx/ziqian_data/recon_cases_91115_default
        case = GT2-00007_20240402105310_20240402105440_128279484
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


        self.beam_inclinations = cal_beam_inclinations()

        self.W_lidar = int(360/0.2)
        self.H_lidar = int(32)
        self.train_frame_times = train_frame_times
        self.max_frame_num = len(train_frame_times)
        self.max_depth = args.max_depth # helios 5515 的最大深度是150 看情况调整，大于100的点也很少基本，误差更大
        self.aabb_min = np.ones(3, dtype=np.float32)*(10000000)
        self.aabb_max = np.ones(3, dtype=np.float32)*(-10000000)

        self.sensor2baselidar = dict() # 记录每个lidar到toplidar的变换矩阵
        self.load_lidar_extrinsics()

        self.pcds = [] # 原始每帧点云
        self.pcds_label = [] # onemodle的语义结果 ==10为地面 ==0为背景 
        self.l2ws = [] # 每帧的l2w
        self.timestep_2_frameid = dict() # 通过timsestep查询对应训练的帧的id _ 0 to 50
        self.frameid_2_timestep = [] # 通过frame id 反查询对应训练帧的timestep
        self.lidar_map = {0:'TOP', 1:'BACK', 3:'LEFT', 4:'RIGHT'}
        self.selected_sensor = args.sensorid # 0 / 1 / 3 /4
        self.lidar_name = self.lidar_map[self.selected_sensor]
        self.ref50_baselidar2world = None
        self.baselidar2world = []
        self.pcds, self.l2ws, self.pcds_label = self.load_pcds(frames=self.frames_data, train_frame_times=self.train_frame_times, frame_num=self.max_frame_num, selected_sensor=self.selected_sensor)

        self.obj_id_list = self.load_dynamic_obj_id_list() 

        self.obj_frames_id = dict()  # 通过obj_id 查询实例出现在哪几帧（列表）(frame id : 0-50)
        self.obj_pcd = dict() # 通过obj_id 查询实例的拼接后的完整的pcd
        self.obj_o2l = dict() # 通过obj_id 和对应那一帧的frame id查询实例的o2l ， 字典嵌套了一个字典
        if self.obj_id_list is not None:
            for obj_id in self.obj_id_list:
                self.obj_pcd[str(obj_id)] = self.load_dynamic_pcd(str(obj_id))
        if train:
            self.static_pcd = self.load_static_scene(use_pcd=False)
        self.range_views, self.masks = self.load_rangeview(self.H_lidar,self.W_lidar)

        self.newcar_l2ws = []
        self.new_sensor2baselidar = None
        self.T_GT2_baselidar_GT2V1_baselidar = None
        if args.newcar_render == "GT2V1":
            print("### use new car info ###")
            self.T_GT2_baselidar_GT2V1_baselidar = np.array([
            [1,0,0,-0.2],
            [0,1,0,0],
            [0,0,1,0.2],
            [0,0,0,1]
            ])
            self.newcar_l2ws, self.new_sensor2baselidar = self.load_newcar_info(frames=self.frames_data, train_frame_times=self.train_frame_times, frame_num=self.max_frame_num, selected_sensor=self.selected_sensor)

    def load_lidar_extrinsics(self):
        '''
        加载每个lidar的相对变化到top
        '''
        sensor_yaml = self.root_path + '/car_cfgs/' + self.case + '/frame/sensors.yaml'
        sensors = load_yaml_str(sensor_yaml)
        baselidar2body = load_extrinsics(sensors["BASE_LIDAR"]["initial_transform"])
        body2baselidar = np.linalg.inv(baselidar2body)
        top2body = load_extrinsics(sensors["ROBO_TOP"]["initial_transform"])
        # sensor2top['TOP'] = np.identity(4)
        backsensor2top = load_extrinsics(sensors['ROBO_BACK']["initial_transform"])
        leftsensor2top = load_extrinsics(sensors['ROBO_LEFT_FRONT']["initial_transform"])
        rightsensor2top = load_extrinsics(sensors['ROBO_RIGHT_FRONT']["initial_transform"])

        self.sensor2baselidar['TOP'] = body2baselidar @ top2body 
        # print("[ debug ] top2baselidar:",self.sensor2baselidar['TOP'])

        back2body = top2body @ backsensor2top 
        self.sensor2baselidar['BACK'] = body2baselidar @ back2body 
        # print("[ debug ] back2baselidar:",self.sensor2baselidar['BACK'])

        left2body = top2body @ leftsensor2top
        self.sensor2baselidar['LEFT'] = body2baselidar @ left2body 
        # print("[ debug ] left2baselidar:",self.sensor2baselidar['LEFT'])

        right2body = top2body @ rightsensor2top 
        self.sensor2baselidar['RIGHT'] = body2baselidar @ right2body 
        # print("[ debug ] right2baselidar:",self.sensor2baselidar['RIGHT'])

    def load_static_scene(self, use_pcd = True):
        '''
        但这个函数非常耗时，是否必要待定
        由于是整个场景的点，可能会太多视野外的，初始化显存不够 接近上千万的点  这是5个激光拼出来的点.也会造成太密的问题 且有效信息变得更少了
        eg.  /mnt_gx/ziqian_data/recon_cases_91115_default/temp/GT2-00007_20240402105310_20240402105440_128279484/occ/preproc/ground.pcd
        '''
        ground_pcd_path = self.root_path + '/temp/' + self.case + '/occ/preproc/ground.pcd' 
        if use_pcd and os.path.exists(ground_pcd_path):
            ground_pcd = o3d.io.read_point_cloud(ground_pcd_path).points
            ground_pcd = np.array(ground_pcd,dtype=np.float32)
            ground_pcd_normal = o3d.io.read_point_cloud(ground_pcd_path).normals
            ground_pcd_normal =  np.array(ground_pcd_normal, dtype=np.float32)
            within_aabb = np.all((ground_pcd >= self.aabb_min) & (ground_pcd <= self.aabb_max), axis=1)
            ground_pcd = ground_pcd[within_aabb]
            ground_pcd_normal = ground_pcd_normal[within_aabb]
            # 晒掉其他传感器的点，没什么用还占显存
            if_sensor_pcds = np.zeros((ground_pcd_normal.shape[0]), dtype=bool)
            for sensor2world in self.l2ws:
                world2sensor = np.linalg.inv(sensor2world)
                sensor_ground_pcd_normal = (np.pad(ground_pcd_normal, ((0,0),(0, 1)), constant_values=1) @ world2sensor.T)[:,:3]
                if_sensor_pcd = np.linalg.norm(sensor_ground_pcd_normal, ord=2, axis=1) < 0.1
                if_sensor_pcds = if_sensor_pcds | if_sensor_pcd
            ground_pcd = ground_pcd[if_sensor_pcds]

            other_pcd_path = self.root_path + '/temp/' + self.case + '/occ/preproc/other_all_with_normal.pcd' 
            other_pcd = o3d.io.read_point_cloud(other_pcd_path).points
            other_pcd = np.array(other_pcd, dtype=np.float32)
            other_pcd_normal = o3d.io.read_point_cloud(other_pcd_path).normals
            other_pcd_normal =  np.array(other_pcd_normal, dtype=np.float32)
            within_aabb = np.all((other_pcd >= self.aabb_min) & (other_pcd <= self.aabb_max), axis=1)
            other_pcd = other_pcd[within_aabb]
            other_pcd_normal = other_pcd_normal[within_aabb]
            if_sensor_other_pcds = np.zeros((other_pcd_normal.shape[0]), dtype=bool)
            for sensor2world in self.l2ws:
                world2sensor = np.linalg.inv(sensor2world)
                sensor_other_pcd_normal = (np.pad(other_pcd_normal, ((0,0),(0, 1)), constant_values=1) @ world2sensor.T)[:,:3]
                if_sensor_other_pcd = (np.linalg.norm(sensor_other_pcd_normal, ord=2, axis=1)) < 0.1
                if_sensor_other_pcds = if_sensor_other_pcds | if_sensor_other_pcd
            other_pcd = other_pcd[if_sensor_other_pcds]

            static_pcd = np.concatenate((ground_pcd, other_pcd), axis=0)                
        else:
            static_pcd = []
            for ind in range(0, self.max_frame_num, 2):
                each_frame_pcd_init = self.pcds[ind]
                each_frame_pcd_label_init = self.pcds_label[ind]
                l2w = self.l2ws[ind]
                static_each_frame_pcd = each_frame_pcd_init[each_frame_pcd_label_init.flatten()>8]
                static_each_frame_pcd = (np.pad(static_each_frame_pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
                static_pcd.append(static_each_frame_pcd[:,:3])
            static_pcd = np.concatenate(static_pcd, axis=0)

        return static_pcd

    def load_dynamic_obj_id_list(self):
        '''
        从文件夹加载动态物体ID
        path = /mnt_gx/ziqian_data/recon_cases_91115_default/temp/GT2-00007_20240402105310_20240402105440_128279484/occ/preproc/dynamic/objects
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
            each_frame_pcd_path = dynamic_obj_path +'/'+ value["name"]
            each_frame_pcd = o3d.io.read_point_cloud(each_frame_pcd_path).points
            each_frame_pcd = np.array(each_frame_pcd, dtype=np.float32)
            # if each_frame_pcd.shape[0]<10: # TODO 待定 出现的很少 可能是很小很远 或者错检 ？？
            #     continue
            if key in self.timestep_2_frameid:
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
        for frame in frames:
            if count_ind == frame_num: break
            if frame['log_time_stamp'] != int(train_frame_times[count_ind]): continue
            # 这里的 lidar2world 是baselidar系
            l2w = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
            self.baselidar2world.append(l2w)
            if count_ind == frame_num-1: self.ref50_baselidar2world = l2w # 把最后一帧的baselidar2world记录下来，用于物体植入时矫正物体的坐标系方向
            sl2w = l2w @ self.sensor2baselidar[self.lidar_map[selected_sensor]] # sensor_lidar_2_world
            l2ws.append(sl2w)

            pcd_path = self.root_path + "/" + frame["path"]["pcd"]
            pcd_sematic_label_path = pcd_path.replace("pcds","pre_semantic_labels")
            if os.path.exists(pcd_sematic_label_path) == False:
                pcd_sematic_label_path = pcd_path.replace("pcds","onemodel_infer/pre_semantic_labels")
                # pcd_sematic_label_path = pcd_path.replace("pcds","semantic_labels")# 这是类别更多的结果 逻辑不通用 暂时先不用

            pcd = np.load(pcd_path)["data"].reshape((-1,7))
            pre_sematic_label = np.load(pcd_sematic_label_path)['data'].reshape((-1,1))
            pcd = np.concatenate((pcd, pre_sematic_label), axis=1)
            pcd = pcd[np.where(pcd[:,6]==selected_sensor)] # 根据sensor_id 划分各雷达数据
            pcd = pcd[np.where(pcd[:,5]==0)] # 删除自车点
            distances = np.linalg.norm(pcd[...,:3], axis=1) # 过滤
            if selected_sensor==0:
                condition1 = (((pcd[:, 0] >= 0) | (distances >= 4)) & (distances < self.max_depth)) #(pcd[:, 2] < 15)
            else:
                condition1 = (distances < self.max_depth) # & (pcd[:, 2] < 15)  # 这个15是为了和mesh仿真的方法对齐以作比较
            pcd = pcd[condition1]
            pcds_label.append(pcd[:,7:8])

            pcd_world = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
            aabb_min = np.min(pcd_world, axis=0)
            aabb_max = np.max(pcd_world, axis=0)
            # print("[ debug ] aabb:", aabb_min, aabb_max)
            self.aabb_min = np.minimum(self.aabb_min, aabb_min)
            self.aabb_max = np.maximum(self.aabb_max, aabb_max)

            # 此时pcd还是在baselidar系 要去求对应每个lidar的rangview需要转到对应lidar
            baselidar2sensor = np.linalg.inv(self.sensor2baselidar[self.lidar_map[selected_sensor]])
            sensor_pcd = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ baselidar2sensor.T)[:,:3]      
            pcd[:,:3] = sensor_pcd[:,:3]
            pcd[:,3] /= 255.0  # helios 的intensity是0-255与其他数据不同
            pcds.append(pcd[:,:4])

            timestep = frame["path"]["pcd"].split('/')[-1].split('.')[0]
            self.timestep_2_frameid.update({timestep : count_ind})
            self.frameid_2_timestep.append(timestep)
            count_ind += 1

        if count_ind < frame_num: raise ValueError("Missing Frame or Abnormal Loading.")
        return pcds, l2ws, pcds_label
        
    def load_rangeview(self,H_lidar,W_lidar):
        '''
        return [H,W,3] rangeiew
        '''
        range_views = []
        masks = []
        for frame_idx in range(self.max_frame_num):
            pano, intensities, mask = lidar_to_pano_with_intensities(  # TODO 把地面矫正放到外面，就能给每个lidar做
                    local_points_with_intensities=self.pcds[frame_idx],
                    lidar_H=H_lidar,
                    lidar_W=W_lidar,
                    beam_inclinations=self.beam_inclinations,
                    max_depth=self.max_depth, # 手册上是 0.2 - 150 
                    ground = (self.pcds_label[frame_idx]==10),
                    is_correction = True,
                    sensor_id = self.selected_sensor,
                    s2b = self.sensor2baselidar[self.lidar_map[self.selected_sensor]]
                )

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

    def evaluate_with_rawpcd(self,frame_idx, TOP_pcd, BACK_pcd, LEFT_pcd, RIGHT_pcd, rangeview_pcd=None):
        '''
            和原始点云做评价
        '''
        frame = self.frames_data[frame_idx] 
        pcd_path = self.root_path + "/" + frame["path"]["pcd"]
        pcd = np.load(pcd_path)["data"].reshape((-1,7))
        selected_sensors = [0,1,3,4]
        gt_pcds = []
        for selected_sensor in selected_sensors:
            each_pcd = pcd[np.where(pcd[:,6]==selected_sensor)] # 根据sensor_id 划分各雷达数据
            each_pcd = each_pcd[np.where(each_pcd[:,5]==0)] # 删除自车点
            distances = np.linalg.norm(each_pcd[...,:3], axis=1) # 过滤
            if selected_sensor==0:
                condition1 = (((each_pcd[:, 0] >= 0) | (distances >= 4)) & (distances < self.max_depth) & (each_pcd[:, 2] < 15))
            else:
                condition1 = ((distances < self.max_depth) & (each_pcd[:, 2] < 15))
            each_pcd = each_pcd[condition1]
            gt_pcds.append(each_pcd)       
        gt_pcd = np.concatenate(gt_pcds,axis=0)[:,:3]

        TOP_pcd = (np.pad(TOP_pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['TOP'].T)[:,:3]      
        BACK_pcd = (np.pad(BACK_pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['BACK'].T)[:,:3]   
        LEFT_pcd = (np.pad(LEFT_pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['LEFT'].T)[:,:3]   
        RIGHT_pcd = (np.pad(RIGHT_pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['RIGHT'].T)[:,:3]   
        render_pcd = np.vstack((TOP_pcd, BACK_pcd, LEFT_pcd, RIGHT_pcd))

        np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/render_pcd.txt", render_pcd, delimiter=",", fmt="%.2f")
        np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/raw_pcd.txt", gt_pcd, delimiter=",", fmt="%.2f")

        rangeview_gt_pcd=None
        if rangeview_pcd is not None:
            TOP_gt_pcd = (np.pad(rangeview_pcd[0], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['TOP'].T)[:,:3]      
            BACK_gt_pcd = (np.pad(rangeview_pcd[1], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['BACK'].T)[:,:3]   
            LEFT_gt_pcd = (np.pad(rangeview_pcd[2], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['LEFT'].T)[:,:3]   
            RIGHT_gt_pcd = (np.pad(rangeview_pcd[3], ((0,0),(0, 1)), constant_values=1) @ self.sensor2baselidar['RIGHT'].T)[:,:3]   
            rangeview_gt_pcd = np.vstack((TOP_gt_pcd, BACK_gt_pcd, LEFT_gt_pcd, RIGHT_gt_pcd))
            ## 由于早期代码有bug，之前留下的数据rangeview真值有问题 这里暂时保留异常值处理
            dists = np.linalg.norm(rangeview_gt_pcd, axis=1)
            rangeview_gt_pcd = rangeview_gt_pcd[dists<51]
            np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/rangeview_pcd.txt", rangeview_gt_pcd, delimiter=",", fmt="%.2f")

        chamLoss = chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(render_pcd[None, ...]).cuda(),
            torch.FloatTensor(gt_pcd[None, ...]).cuda())
        chamfer_dis = dist1.mean() + dist2.mean()

        chamfer_dis2 = 0
        if gt_pcd is not None:
            chamLoss1 = chamfer_3DDist()
            _dist1, _dist2, _idx1, _idx2 = chamLoss1(
                torch.FloatTensor(render_pcd[None, ...]).cuda(),
                torch.FloatTensor(rangeview_gt_pcd[None, ...]).cuda())
            chamfer_dis2 = _dist1.mean() + _dist2.mean()

        diff_cd = 0
        # if gt_pcd is not None:
        #     chamLoss2 = chamfer_3DDist()
        #     ext_dist1, ext_dist2, ext_idx1, ext_idx2 = chamLoss2(
        #         torch.FloatTensor(gt_pcd[None, ...]).cuda(),
        #         torch.FloatTensor(rangeview_gt_pcd[None, ...]).cuda())
        #     diff_cd = ext_dist1.mean() + ext_dist2.mean()

        return chamfer_dis, chamfer_dis2, diff_cd

    def test_pcd(self,frame_idx):
        '''
        临时的一个测试函数
        '''
        rangeview_test = self.get_rangeview(frame_idx)
        raydrop = rangeview_test[:,:,0]
        depth = rangeview_test[:,:,2]*raydrop
        print("depth max:",depth.max())
        pcd_test = pano_to_lidar(depth, beam_inclinations = self.beam_inclinations)
        # np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/rangeview_pcd_test.txt", pcd_test, delimiter=",", fmt="%.2f")

    def load_newcar_info(self, frames, train_frame_times, frame_num=50, selected_sensor=0):
        '''
        读取新车型的雷达配置文件
        eg. /mnt_gx/ziqian/lidar_simulation_infos/GT2V1/sensors.yaml
        '''
        sensor_yaml = "/mnt_gx/ziqian/lidar_simulation_infos/GT2V1/sensors.yaml"
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
        # 此处sensor2baselidar的baselidar系是GT2V1的baselidar
        # T_GT2V1_baselidar_GT2_baselidar = np.linalg.inv(self.T_GT2_baselidar_GT2V1_baselidar)
        new_l2w = []
        count_ind = 0
        for frame in frames:
            if count_ind == frame_num: break
            if frame['log_time_stamp'] != train_frame_times[count_ind]: continue
            # 这里的 lidar2world 是baselidar系
            l2w = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
            sl2w = l2w @ (self.T_GT2_baselidar_GT2V1_baselidar @ sensor2baselidar[self.lidar_map[selected_sensor]]) # sensor_lidar_2_world
            new_l2w.append(sl2w)
            count_ind +=1
        if count_ind < frame_num: raise ValueError("Missing Frame or Abnormal Loading.")
        
        return new_l2w, sensor2baselidar

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
        if newcar_render == "GT2V1":
            baselidar2sensor = np.linalg.inv(self.new_sensor2baselidar[self.lidar_map[self.selected_sensor]])
            T_GT2V1_baselidar_GT2_baselidar = np.linalg.inv(self.T_GT2_baselidar_GT2V1_baselidar)
            return baselidar2sensor @ (T_GT2V1_baselidar_GT2_baselidar @ self.obj_o2l[str(object_id)][str(occurred_frame_idx)])
            # return baselidar2sensor @ self.obj_o2l[object_id][str(occurred_frame_idx)]
        else:
            baselidar2sensor = np.linalg.inv(self.sensor2baselidar[self.lidar_map[self.selected_sensor]])
            return baselidar2sensor @ self.obj_o2l[str(object_id)][str(occurred_frame_idx)]

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

# if __name__ == '__main__':
#     root_path =  "/mnt_gx/ziqian_data/recon_cases_91115_default"
#     case = "GT2-00007_20240402105310_20240402105440_128279484"
#     GT_DATA = GT_Dataloader(root_path, case)
#     # print(GT_DATA.frames_data[0]["path"]["pcd"])
#     GT_DATA.load_dynamic_obj(root_path,case)

