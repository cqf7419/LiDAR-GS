import open3d as o3d
from tqdm import tqdm
import numpy as np

def quaternion_from_vectors(v1, v2):
    # 计算旋转轴
    axis = np.cross(v1, v2)
    axis_length = np.linalg.norm(axis)

    # 如果轴的长度是0，说明两个向量相同，返回单位四元数
    if axis_length == 0:
        return np.array([1, 0, 0, 0])  # 单位四元数

    # 归一化旋转轴
    axis = axis / axis_length

    # 计算旋转角度
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止超出范围

    # 构造四元数
    q = np.array([
        np.cos(theta / 2),                       # w
        axis[0] * np.sin(theta / 2),            # x
        axis[1] * np.sin(theta / 2),            # y
        axis[2] * np.sin(theta / 2)             # z
    ])

    return q

objs = [
    "3d_real_car"
    # "manhole/manhole_bad_000_1",
    # "manhole/manhole_bad_001_1",
    # "manhole/manhole_bad_002_1",
    # "manhole/manhole_bad_003_1",
    # "manhole/manhole_bad_004_1",
    # "manhole/manhole_bad_005_1",
    # "cone/big_cone_001_1",
    # "cone/small_cone_001_1",
    # "chain/cone_chain_001_1",
    # "chain/cone_chain_002_1",
    # "chain/cone_chain_003_1",
    # "chain/cone_chain_004_1",
    # "chain/cone_chain_005_1",
    # "chain/cone_chain_006_1"
]

# 读取点云
for obj in tqdm(objs):
    path = "/mnt_gx/usr/lansheng/Gobj/" + obj 
    if obj == "3d_real_car":
        obj_name = "/point_cloud.ply"
    else:
        obj_name = "/points3D.ply"
    load_path = path + obj_name
    pcd = o3d.io.read_point_cloud(load_path)  # 替换为您的点云文件路径

    # 估算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 获取法向量
    normals = np.asarray(pcd.normals)

    # 我们定义的参考法向量
    reference_vector = np.array([0, 0, 1])
    # 将法向量转换为四元数
    quaternions = []
    for normal in normals:
        q = quaternion_from_vectors(reference_vector, normal)
        quaternions.append(q)

    # 将结果保存为 NumPy 数组
    quaternions = np.array(quaternions)

    # 保存到文件 (例如以 .npy 格式)
    point =  np.array(pcd.points)[:,:3]
    print(quaternions.shape)
    point_with_normal = np.concatenate((point, quaternions), axis=1)
    np.save(f"{path}/pcd_with_rot.npy", point_with_normal)
