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
from scipy.spatial.transform import Rotation

class GT_Dataloader:
    '''
    eg. 
        root_path =  /mnt_gx/usr/lansheng/regroup_92171_all
        case = GT2-00007_20240402105310_20240402105440_128279484
    '''
    def __init__(self, args, train=True, train_frame_times=None, dtype=np.float32):
        self.train = train
        self.root_path = args.source_path
        self.track_list = ["track_0", "track_1", "track_2"]
        self.meta_info_path = [self.root_path + "/pack/scene_3/" + track_list + "/meta_infos/" + track_list + ".pkl" for track_list in self.track_list]
        print(self.meta_info_path)
        # read meta_info
        self.start_frame = 0 # 从第0帧开始  留着这个接口 目前给的case都是50帧不用特殊处理
        self.frames_data = []
        for meta_info_path in  self.meta_info_path:
            with open(meta_info_path, 'rb') as pickle_file:
                all_data = pickle.load(pickle_file)
                self.frames_data.append(all_data['frames'])
                print("[ Info ]"+ meta_info_path +" this case have {} frames totally".format(len(all_data['frames'])))


        self.beam_inclinations = cal_beam_inclinations()
        print("[ Info ] beam inclination:",self.beam_inclinations)

        self.W_lidar = int(360/0.2)
        self.H_lidar = int(32)
        self.train_frame_times = train_frame_times
        self.max_frame_num = len(train_frame_times)
        print("max_frame_num",self.max_frame_num)
        self.max_depth = args.max_depth # helios 5515 的最大深度是150 看情况调整，大于100的点也很少基本，误差更大
        self.aabb_min = np.ones(3, dtype=np.float32)*(100000) # temp
        self.aabb_max = np.ones(3, dtype=np.float32)*(-100000)

        self.sensor2baselidar = dict() # 记录每个lidar到toplidar的变换矩阵
        self.load_lidar_extrinsics()

        self.pcds = [] # 原始每帧点云
        self.pcds_label = [] # 对应每个点的标签 ==10为地面 ==0为背景 
        self.l2ws = [] # 每帧的l2w
        self.timestep_2_frameid = dict() # 通过timsestep查询对应训练的帧的id _ 0 to 50
        self.frameid_2_timestep = [] # 通过frame id 反查询对应训练帧的timestep
        self.lidar_map = {0:'TOP', 1:'BACK', 3:'LEFT', 4:'RIGHT'}
        self.selected_sensor = args.sensorid # 0 / 1 / 3 /4
        self.lidar_name = self.lidar_map[self.selected_sensor]
        print("[ dataloader ] selected lidar :",self.lidar_name)
        self.all_pcd_pose = self.load_pcd_pose()
        self.pcds, self.l2ws, self.pcds_label = self.load_pcds(self.frames_data, self.train_frame_times, frame_num=self.max_frame_num, selected_sensor=self.selected_sensor)
        print("[ debug ] self.pcds_label.shape ",len(self.pcds_label),self.pcds_label[0].shape)

        self.obj_id_list = None
        if train:
            self.static_pcd = self.load_static_scene()
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
            self.newcar_l2ws, self.new_sensor2baselidar = self.load_newcar_info(self.frames_data, frame_num=self.max_frame_num, selected_sensor=self.selected_sensor)
    
    def load_lidar_extrinsics(self):
        '''
        加载每个lidar的相对变化到top
        '''
        sensor_yaml = self.root_path + '/frame/sensors.yaml'
        print("[ debug ] sensor yaml:", sensor_yaml)
        sensors = load_yaml_str(sensor_yaml)
        baselidar2body = load_extrinsics(sensors["BASE_LIDAR"]["initial_transform"])
        body2baselidar = np.linalg.inv(baselidar2body)
        top2body = load_extrinsics(sensors["ROBO_TOP"]["initial_transform"])
        # sensor2top['TOP'] = np.identity(4)
        backsensor2top = load_extrinsics(sensors['ROBO_BACK']["initial_transform"])
        leftsensor2top = load_extrinsics(sensors['ROBO_LEFT_FRONT']["initial_transform"])
        rightsensor2top = load_extrinsics(sensors['ROBO_RIGHT_FRONT']["initial_transform"])

        self.sensor2baselidar['TOP'] = body2baselidar @ top2body 
        print("[ debug ] top2baselidar:",self.sensor2baselidar['TOP'])

        back2body = top2body @ backsensor2top 
        self.sensor2baselidar['BACK'] = body2baselidar @ back2body 
        print("[ debug ] back2baselidar:",self.sensor2baselidar['BACK'])

        left2body = top2body @ leftsensor2top
        self.sensor2baselidar['LEFT'] = body2baselidar @ left2body 
        print("[ debug ] left2baselidar:",self.sensor2baselidar['LEFT'])

        right2body = top2body @ rightsensor2top 
        self.sensor2baselidar['RIGHT'] = body2baselidar @ right2body 
        print("[ debug ] right2baselidar:",self.sensor2baselidar['RIGHT'])

    def load_static_scene(self, load_pcd=False):
        '''
        由于是整个场景的点，可能会太多视野外的，初始化显存不够 接近上千万的点  这是5个激光拼出来的点.也会造成太密的问题 且有效信息变得更少了
        eg.  /mnt_gx/ziqian_data/recon_cases_91115_default/temp/GT2-00007_20240402105310_20240402105440_128279484/occ/preproc/ground.pcd
        '''
        if load_pcd:
            cases = ["GT2-00006_20240614143004_20240614143030_8", "GT2-00006_20240614150446_20240614150512_9", "GT2-00006_20240614154135_20240614154201_10"]
            static_pcds = []
            for case in cases:
                ground_pcd_path = self.root_path + '/temp/' + case + '/occ/preproc/ground.pcd' 
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

                other_pcd_path = self.root_path + '/temp/' + case + '/occ/preproc/other_all_with_normal.pcd' 
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
                static_pcds.append(static_pcd)
                print("[ debug ] each static pcd shape:",static_pcd.shape)
            static_pcds = np.concatenate(static_pcds, axis=0)
        else:
            static_pcd = []
            for ind in range(0, self.max_frame_num, 3):
                each_frame_pcd_init = self.pcds[ind]
                each_frame_pcd_label_init = self.pcds_label[ind]
                l2w = self.l2ws[ind]
                static_each_frame_pcd = each_frame_pcd_init[each_frame_pcd_label_init.flatten()>8]
                static_each_frame_pcd = (np.pad(static_each_frame_pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
                static_pcd.append(static_each_frame_pcd[:,:3])
            static_pcds = np.concatenate(static_pcd, axis=0)
            print("[ debug ] all static pcd shape:",static_pcds.shape)

        return static_pcds

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
        return None

    def load_pcd_pose(self):
        all_pcd_pose = []
        cases = ["GT2-00006_20240614143004_20240614143030_8", "GT2-00006_20240614150446_20240614150512_9", "GT2-00006_20240614154135_20240614154201_10"]
        for case in cases:
            file_path = self.root_path + "/temp/"+ case +"/annotation/single_frame_cloud_poses.txt"
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


    def load_pcds(self, frames_list, train_frame_times, frame_num=50, selected_sensor=0):
        '''
        加载每帧点云 作为gt 以及计算边界 其中selected_sensor = 0 / 1 / 3 / 4 分别代表 ROTOTOP, ROBO_BACK, ROBO_LEFT_FRONT, ROBO_RIGHT_FRONT
        return : 原始每帧点云_baselidar系 
        '''
        pcds = []
        pcds_label = []
        l2ws = []
        count_ind = 0
        print("[ debug ] max frame_num", frame_num)
        for ind, frames in enumerate(frames_list):
            print("load track_{}".format(ind))
            if count_ind == frame_num: break
            for frame_ind, frame in enumerate(frames):
                if count_ind == frame_num: break
                if frame['log_time_stamp'] != int(train_frame_times[count_ind]): continue
                # 这里的 lidar2world 是baselidar系
                # l2w = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
                l2w = self.all_pcd_pose[ind][frame_ind]
                sl2w = l2w @ self.sensor2baselidar[self.lidar_map[selected_sensor]] # sensor_lidar_2_world
                l2ws.append(sl2w)
                # if i==0: print("[ debug ] sensor2world 0-ind:",sl2w)

                pcd_path = self.root_path + "/" + frame["path"]["pcd"]
                pcd_sematic_label_path = pcd_path.replace("pcds","pre_semantic_labels")

                pcd = np.load(pcd_path)["data"].reshape((-1,7))
                pre_sematic_label = np.load(pcd_sematic_label_path)['data'].reshape((-1,1))
                pcd = np.concatenate((pcd, pre_sematic_label), axis=1)
                pcd = pcd[np.where(pcd[:,6]==selected_sensor)] # 根据sensor_id 划分各雷达数据
                pcd = pcd[np.where(pcd[:,5]==0)] # 删除自车点
                distances = np.linalg.norm(pcd[...,:3], axis=1) # 过滤
                if selected_sensor==0:
                    condition1 = (((pcd[:, 0] >= 0) | (distances >= 4)) & (distances < self.max_depth) ) # hardcode 简单的过滤
                else:
                    condition1 = ((distances < self.max_depth) )
                pcd = pcd[condition1]
                pcds_label.append(pcd[:,7:8])

                pcd_world = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
                aabb_min = np.min(pcd_world, axis=0)
                aabb_max = np.max(pcd_world, axis=0)
                self.aabb_min = np.minimum(self.aabb_min, aabb_min)
                self.aabb_max = np.maximum(self.aabb_max, aabb_max)

                # 此时pcd还是在baselidar系 要去求对应每个lidar的rangview需要转到对应lidar
                baselidar2sensor = np.linalg.inv(self.sensor2baselidar[self.lidar_map[selected_sensor]])
                sensor_pcd = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ baselidar2sensor.T)[:,:3]      

                pcd[:,:3] = sensor_pcd[:,:3]
                pcd[:,3] /= 255.0  # helios 的intensity是0-255与其他数据不同
                pcds.append(pcd[:,:4])
                # if count_ind == 0:
                #     ground_pcd = pcd[pcd[:,7]==10]
                #     np.savetxt("./test_left_ground_pcd.txt", ground_pcd[:,:3])

                timestep = frame["path"]["pcd"].split('/')[-1].split('.')[0]
                self.timestep_2_frameid.update({timestep : count_ind})
                self.frameid_2_timestep.append(timestep)
                count_ind += 1

        if count_ind < frame_num: print("count_ind,",count_ind);raise ValueError("Missing Frame or Abnormal Loading.")
        print("final aabb:",self.aabb_min, self.aabb_max)
        return pcds, l2ws, pcds_label

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
                    max_depth=self.max_depth, # 手册上是 0.2 - 150 
                    ground = (self.pcds_label[frame_idx]==10),
                    is_correction = True,
                    sensor_id = self.selected_sensor,
                    pre_labels = self.pcds_label[frame_idx],
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
            # if frame_idx == 1:
            #     import imageio 
            #     imageio.imwrite('output_mask.png', (mask * 255).astype(np.uint8))
        return range_views, masks

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

        # np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/render_pcd.txt", render_pcd, delimiter=",", fmt="%.2f")
        # np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/raw_pcd.txt", gt_pcd, delimiter=",", fmt="%.2f")

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
            # np.savetxt("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/rangeview_pcd.txt", rangeview_gt_pcd, delimiter=",", fmt="%.2f")

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

    def load_newcar_info(self, frames, frame_num=50, selected_sensor=0):
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
        T_GT2V1_baselidar_GT2_baselidar = np.linalg.inv(self.T_GT2_baselidar_GT2V1_baselidar)
        new_l2w = []
        for i in range(frame_num*len(self.track_list)):
            frame = frames[i] 
            # 这里的 lidar2world 是baselidar系
            l2w = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
            sl2w = l2w @ (T_GT2V1_baselidar_GT2_baselidar @ sensor2baselidar[self.lidar_map[selected_sensor]]) # sensor_lidar_2_world
            new_l2w.append(sl2w)
        
        return new_l2w, sensor2baselidar
    def get_frames_nums(self):
        return len(self.pcds_label)

    def get_static_pcd(self):
        return self.static_pcd
    
    def get_obj_pcd(self, object_id):
        return None

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
        return None

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
        return self.obj_frames_id[object_id]

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

