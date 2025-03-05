import numpy as np
import pickle
import os
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
import math
try:
    from simple_knn._C import distCUDA2
except ImportError:
    print("没有simple_knn")

def load_T(case, log, ind):
    path = "/mnt_gx/dsc/GStudio/eval_output/{}_{}/pos/pos{}.txt".format(case, log, ind)
    # path = "/mnt_gx/dsc/GStudio/eval_output/{}_{}/pos{}.txt".format(case, log, ind)
    data = np.loadtxt(path) #+ np.array([0,0,0.02])
    return data

def get_obj_type():
    objs = [
        # "manhole/manhole_bad_000_1",
        # "manhole/manhole_bad_001_1",
        # "manhole/manhole_bad_002_1",
        # "manhole/manhole_bad_003_1",
        # "manhole/manhole_bad_004_1",
        # "manhole/manhole_bad_005_1",
        # "cone/big_cone_001_1",
        # "cone/small_cone_001_1",
        # "chain/cone_chain_001_1",
        "chain/cone_chain_002_1",
        "chain/cone_chain_003_1",
        "chain/cone_chain_004_1",
        "chain/cone_chain_005_1",
        "chain/cone_chain_006_1"
    ]
    return objs

def load_pcd_with_labels(file_path, return_label=True):
    with open(file_path, 'r') as f:
        # 跳过 PCD 文件头信息，直到数据开始行
        lines = f.readlines()
        data_started = False
        points = []
        labels = []
        
        for line in lines:
            if not data_started:
                if line.startswith("DATA"):
                    data_started = True
                continue
            
            # 分割每一行数据，假设格式是 x y z label object
            parts = line.strip().split()
            if len(parts) >= 4:  # 确保有足够的字段（x, y, z, label）
                x, y, z, label = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                points.append([x, y, z])
                labels.append([label])
    if return_label==False: return np.array(points)
    return np.array(points), np.array(labels)

def voxelize_sample(data=None, label=None, voxel_size=0.05):
    '''
    voxel降采样
    第四维是label信息，为了简化采样过程，label也会
    '''
    # np.random.shuffle(data)
    data_voxel, indices = np.unique(np.round(data[:,:3]/voxel_size), axis=0, return_index=True)#*voxel_size
    data_voxel = data_voxel * voxel_size
    if label is not None:
        label = label[indices]
        res = np.concatenate((data_voxel, label), axis=1)
    else:
        res = data_voxel

    return res

def load_ground_pcd(case, log):
    root_path = "/mnt_gx/lidar_data/datanext/"
    meta_info_path = root_path + case + "/meta_infos/" + log + ".pkl"
    with open(meta_info_path, 'rb') as pickle_file:
        all_data = pickle.load(pickle_file)
        frames_data = all_data['frames']
    
    ground_pcds = []
    ref50_baselidar2world = None
    for i in range(0,50):
        if i == 49: 
            frame = frames_data[i]
            ref50_baselidar2world = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
        if i%2 == 0:
            frame = frames_data[i]
            l2w = np.array(frame['optimized_pose']) if 'optimized_pose' in frame else np.array(frame['lidar2world'])
            pcd_path = root_path + case + "/" + frame["path"]["pcd"]
            pcd_sematic_label_path = pcd_path.replace("pcds","pre_semantic_labels")
            if os.path.exists(pcd_sematic_label_path) == False:
                pcd_sematic_label_path = pcd_path.replace("pcds","onemodel_infer/pre_semantic_labels")
            pcd = np.load(pcd_path)["data"].reshape((-1,7))
            pcd[...,:3] = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
            pre_sematic_label = np.load(pcd_sematic_label_path)['data'].reshape((-1,1))
            pcd = np.concatenate((pcd, pre_sematic_label), axis=1)
            pcd = pcd[np.where(pcd[:,5]==0)] # 删除自车点
            pcd = pcd[np.where(pcd[:,7]==10)] # 地面
            ground_pcds.append(pcd[:,:3])
    ground_pcd = np.concatenate(ground_pcds,axis=0)
    return ground_pcd, ref50_baselidar2world

def generateInsertGobjGT(case, log, pos_num=2):
    '''
    生成符合感知bev模型训练需要的obj真值数据(高于地面)
    '''
    ground_pcd,ref50_baselidar2world = load_ground_pcd(case, log)
    # 使用 RANSAC 方法进行平面分割
    distance_threshold = 0.03  # 允许的距离阈值
    ransac_n = 3                # RANSAC 选择的点数
    num_iterations = 1000       # RANSAC 的迭代次数
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(ground_pcd)
    plane_model, inliers = plane_pcd.segment_plane(distance_threshold=distance_threshold,
                            ransac_n=ransac_n,
                            num_iterations=num_iterations)
    a, b, c, d = plane_model

    save_root_path = "/mnt_gx/lidar_data/datanext/ali_simulation_data"

    objs_type = get_obj_type()
    for obj_type in objs_type:
        path = "/mnt_gx/usr/lansheng/Gobj/{}/points3D.pcd".format(obj_type)
        if os.path.exists(path) == False:
            print("没有pcd文件")
            path = path.replace(".pcd",".ply")
            pcd = o3d.io.read_point_cloud(path)
            obj_point = np.array(pcd.points)[:,:3] 
            obj_label = None
            print(obj_point.shape)
        else:
            obj_point, obj_label = load_pcd_with_labels(path)
            print(obj_point.shape)
        for pos_ind in range(pos_num):
            obj_type_pose = obj_type + "_pos{}".format(pos_ind)
            print("==>", obj_type_pose)
            offset = load_T(case, log, pos_ind)
            sim_pose = np.eye(4)
            sim_pose[0, 3] = offset[0]
            sim_pose[1, 3] = offset[1]
            sim_pose[2, 3] = offset[2]
            if offset.shape[0] >= 6:
                theta = -1.0 * offset[5] / 180.0 * np.pi # 顺时针
                rotation_matrix = np.array([
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]
                ])
            sim_pose[0:3, 0:3] = (ref50_baselidar2world[0:3,0:3] @ sim_pose[0:3, 0:3]) @ rotation_matrix
            point = (np.hstack([obj_point, np.ones((obj_point.shape[0], 1))])@ sim_pose.T)[:,:3]
            point = voxelize_sample(data=point, label=obj_label, voxel_size=0.05)

            x = point[:, 0]
            y = point[:, 1]
            z = point[:, 2]
            z_plane = (-d - a * x - b * y) / c if c != 0 else np.full_like(z, np.nan) 

            above_plane = z > z_plane
            point = point[above_plane]
            save_path = os.path.join(save_root_path,case, case + "_" + obj_type_pose.split("/")[-1], "obj_pcd", log)
            os.makedirs(save_path, exist_ok=True)
            np.savetxt(os.path.join(save_path, "points.txt"),point, fmt='%.4f', comments='')
            np.savez(os.path.join(save_path, "points.npz"), data=point)

def generateInsertGobjBEVGTv2(case, log, uuid, obj_path, sim_pose):
    '''
    生成符合感知bev模型训练需要的obj真值数据(高于地面)
    '''
    ground_pcd,ref50_baselidar2world = load_ground_pcd(case, log)
    # 使用 RANSAC 方法进行平面分割
    distance_threshold = 0.05  # 允许的距离阈值
    ransac_n = 3                # RANSAC 选择的点数
    num_iterations = 1000       # RANSAC 的迭代次数
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(ground_pcd)
    plane_model, inliers = plane_pcd.segment_plane(distance_threshold=distance_threshold,
                            ransac_n=ransac_n,
                            num_iterations=num_iterations)
    a, b, c, d = plane_model

    save_root_path = "/mnt_gx/lidar_data/datanext/ali_simulation_data"

    obj_point, obj_label = load_pcd_with_labels(obj_path)

    point = (np.hstack([obj_point, np.ones((obj_point.shape[0], 1))])@ sim_pose.T)[:,:3]
    point = voxelize_sample(data=point, label=obj_label, voxel_size=0.05)

    x = point[:, 0]
    y = point[:, 1]
    z = point[:, 2]
    z_plane = (-d - a * x - b * y) / c if c != 0 else np.full_like(z, np.nan) 

    above_plane = z > z_plane
    point = point[above_plane]
    save_path = os.path.join(save_root_path, case,"obj_pcd", log+"_"+uuid)
    os.makedirs(save_path, exist_ok=True)
    np.savetxt(os.path.join(save_path, "points.txt"),point, fmt='%.4f', comments='')
    np.savez(os.path.join(save_path, "points.npz"), data=point) 

def generateInsertGobjBBox(case, log, uuid, obj_path, sim_pose):
    _, file_extension = os.path.splitext(obj_path)
    if file_extension == ".ply":
        pcd = o3d.io.read_point_cloud(obj_path)
        obj_point =  np.array(pcd.points)[:,:3]
    else:
        obj_point, obj_label = load_pcd_with_labels(obj_path, return_label=True)
    if len(sim_pose.shape) != 3: 
        print("generateInsertGobjBBox input error : sim_pose")
        exit(1)
    for model2world in sim_pose:
        point = (np.hstack([obj_point, np.ones((obj_point.shape[0], 1))])@ model2world.T)[:,:3]
        min_bbox = np.min(point, axis=0)  
        max_bbox = np.max(point, axis=0) 

def loadStaticObj(path, l2w = None, args=None, pos_ind=0, sim_pose=None, extension="ply", dense=True, dropout=0.):
    '''
    加载静态的obj物体, path为obj存放路路径, 加载进场景有多种方式:
    1、obj在baselidar系下。那么需要传l2w进去
    2、obj在world系, 需要加载pos的具体位置,pos_ind根据txt读取
    3、obj在仿真系统,需要逐帧改变pose, 给到sim_pose
    返回一个在场景内的obj的字典,包含gs所需的基本属性
    '''
    if extension == "pcd":
        obj_point, obj_label = load_pcd_with_labels(path, return_label=True)
    else:
        pcd = o3d.io.read_point_cloud(path)
        if l2w is not None:
            obj = np.array(pcd.points)/1000.0 + np.array([[30, 4.0, -2.05]])
            obj_point = (np.pad(obj[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
        elif args is not None:
            casename = args.root_path.split('/')[-1]
            offset = load_T(casename, args.case, pos_ind)
            sim_pose = np.eye(4)
            sim_pose[0, 3] = offset[0]
            sim_pose[1, 3] = offset[1]
            sim_pose[2, 3] = offset[2]
            if offset.shape[0] >= 6:
                theta = -1.0 * offset[5] / 180.0 * np.pi # 顺时针30度
                rotation_matrix = np.array([
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]
                ])
                sim_pose[0:3, 0:3] = (args.ref50_baselidar2world[0:3,0:3] @ sim_pose[0:3, 0:3]) @ rotation_matrix
            obj_point =  np.array(pcd.points)[:,:3]
        else:
            obj_point =  np.array(pcd.points)[:,:3]
            if dense: obj_point = voxelize_sample(data=obj_point, voxel_size=0.025)

    obj_point_homogeneous = np.hstack([obj_point, np.ones((obj_point.shape[0], 1))])
    obj_tensor = torch.tensor(obj_point_homogeneous,dtype=torch.float32).cuda()

    obj_color = torch.ones([obj_tensor.shape[0],2],dtype=torch.float32).cuda() # 0-1
    if dropout > 0.0:
        dropout_label_index = np.where(obj_label==133)[0] # 133 是飘带 123 是锥桶
        random_raydrop = torch.rand([dropout_label_index.shape[0],1],dtype=torch.float32).cuda()>dropout # dropout>0.99 基本上扫不出绳子
        obj_color[dropout_label_index,1:2] = obj_color[dropout_label_index,1:2]*random_raydrop
        del random_raydrop
        torch.cuda.empty_cache()

    obj_scaling = torch.ones([obj_tensor.shape[0],3],dtype=torch.float32).cuda()
    if dense:
        cal_sacle = torch.clamp_min(distCUDA2(obj_tensor[:,:3]).float().cuda(), 0.005)
        obj_scaling = torch.sqrt(cal_sacle)[...,None].repeat(1,3)
        del cal_sacle
        torch.cuda.empty_cache()
        obj_opacity = torch.ones([obj_tensor.shape[0],1],dtype=torch.float32).cuda()*10
    else:
        obj_scaling = obj_scaling*0.005
        obj_opacity = torch.ones([obj_tensor.shape[0],1],dtype=torch.float32).cuda()*1000


    rot_numpy = np.load(path.replace(path.split('/')[-1], "pcd_with_rot.npy"))[:,3:]
    obj_rots = torch.tensor(rot_numpy,dtype=torch.float32).cuda()#torch.zeros((obj_tensor.shape[0], 4),dtype=torch.float32).cuda()

    objs = {
        "xyz": obj_tensor,
        "color": obj_color,
        "opacity": obj_opacity,
        "scaling": obj_scaling,
        "rot": obj_rots
    }
    if sim_pose is not None:
        if sim_pose.shape == (4,4):
            objs["sim_pose"] = torch.tensor(sim_pose,dtype=torch.float32).cuda()
        elif len(sim_pose.shape)==3 and sim_pose.shape[0]>1 and sim_pose[0].shape == (4,4):
            objs["dynamic_pose"] = torch.tensor(sim_pose,dtype=torch.float32).cuda()
        else:
            print("错误的sim_pose格式")
            eixt(0)
    return objs


if __name__ == "__main__":
    # case = "union_cases_93769_default"
    # log = "GT2-00002_20240731152818_20240731152918_128997753"
    # ind = 0
    # T = load_T(case, log, ind)
    # print(T.shape)
    # print(T[0])
    
    # pcd = load_ground_pcd(case,log)
    # np.savetxt("/ground.txt",pcd, fmt='%.4f', comments='')

    with open('/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/case_94205/temp_logname1217.txt', 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(',')

            case = "union_cases_94205_default"#parts[0]
            log = parts[0]#parts[1]
            print(case,log)
            generateInsertGobjGT(case,log,pos_num=3)