import open3d as o3d
import numpy as np
import time
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json
from utils.lidar_utils import pano_to_lidar
import imageio
from utils.lidar_utils import PointsMeter
from utils.loss_utils import l1_loss, ssim
import cv2
import os
import matplotlib.pyplot as plt

def show(gt,render):
    # 创建直方图
    plt.figure(figsize=(10, 6))

    bins = 40  # 您可以根据需要调整bins的数量

    # 绘制直方图
    plt.hist(gt, bins=bins, alpha=0.5, label='gt')
    plt.hist(render, bins=bins, alpha=0.5, label='render')

    # 添加标题和标签
    plt.title('Histogram of Two [2650,] NumPy Arrays')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.legend()
    plt.show()

exp_path = "/mnt_gx/usr/lansheng/workspace/lidar-gs/outputs/waymo_seq1137/waymo1137/2024-08-14_17:01:30"
dataset_type = "/test"
frames = 2
iter = 3000
def pano2lidar(i,beam_inclinations):
    # pano_path = "/home/xuanyuan/lansheng/Scaffold-GS-lidar/outputs/waymo_seq1137/lidar_gs_0627/2024-07-10_17:28:08/train/ours_10000/renders/depth_" + str(i).zfill(5) + ".png"  
    gt_pano_path = exp_path + dataset_type + "/ours_" + str(iter) + "/gt/depth_" + str(i).zfill(5) + ".npy"  
    # gt = imageio.imread(gt_pano_path)
    gt = np.load(gt_pano_path)
    ray_drop = (gt>0)
    # cv2.imwrite("./ray_drop.png", ray_drop*255.0)
    gt = gt*ray_drop
    print(ray_drop.shape)
    pano_path = exp_path + dataset_type + "/ours_" + str(iter) + "/renders/depth_" + str(i).zfill(5) + ".npy"  
    # render = imageio.imread(pano_path)
    render = np.load(pano_path)
    # render_mask = (render>7)
    render = render*ray_drop
    # render = render*render_mask

    point_with_intensity = pano_to_lidar(render, lidar_K=None, beam_inclinations=beam_inclinations)

    points_meter = PointsMeter(scale=1, intrinsics=None, beam_inclinations=beam_inclinations)  # 这里有问题  读图片会被精度截断
    points_meter.update(render[None,...], gt[None,...])
    cd_fs = points_meter.measure()
    cd_test = cd_fs[0]
    fscore_test = cd_fs[1]
    print("point meter: ",cd_test,fscore_test)
    # gt_intensity_path = exp_path + "/test/ours_20000/gt/intensity_" + str(i).zfill(5) + ".png"  
    # gt_intensity = imageio.imread(gt_intensity_path)/255.0
    # intensity_path = exp_path + "/test/ours_20000/renders/intensity_" + str(i).zfill(5) + ".png"  
    # intensity = imageio.imread(intensity_path)/255.0
    # mae = np.abs(gt_intensity*ray_drop-intensity * ray_drop).mean()
    # print("intensity mar:",mae)

    return point_with_intensity[:,:3]

def generate_point_cloud_frame(frame_id):
    num_points = 1000
    # 这里生成一个简单的旋转点云，实际数据来源可能是传感器或文件
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(-1, 1, num_points)
    x = np.sin(theta + frame_id * 0.1)
    y = np.cos(theta + frame_id * 0.1)
    points = np.vstack((x, y, z)).T
    return points

def generate_point_cloud_colors(points):
    distances = np.linalg.norm(points, axis=1)  # 计算每个点到原点的距离
    max_distance = np.max(distances) or 1  # 最大距离（避免除以零）
    norm_distances = distances / max_distance  # 归一化距离到 [0, 1]

    # 使用距原点的距离设置颜色
    # 采用简单的蓝到红的映射：较近的点为蓝色，较远的点为红色
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = norm_distances  # 红色分量（距离越大，红色越强）
    colors[:, 2] = 1 - norm_distances  # 蓝色分量（距离越小，蓝色越强）
    
    return colors

# 创建一个用于存储50帧点云的列表
path = "/mnt_gx/usr/lansheng/workspace/lidar-gs/data/waymo_seq1137/transforms_train.json"
with open(path) as json_file:
    contents = json.load(json_file)
    beam_inclinations = contents["beam_inclinations"]
point_clouds = [pano2lidar(i,beam_inclinations) for i in range(frames)]
# point_clouds = [generate_point_cloud_frame(i) for i in range(47)]
# 可视化
time.sleep(2)
vis = o3d.visualization.Visualizer()
vis.create_window()

# 准备显示的点云
pcd = o3d.geometry.PointCloud()
points = point_clouds[0]
pcd.points = o3d.utility.Vector3dVector(points)
colors = generate_point_cloud_colors(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
# pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 设置初始颜色

vis.add_geometry(pcd)

# view_ctl = vis.get_view_control()
# view_ctl.set_up([0, 1, 0])       # z轴向上
# view_ctl.set_front([0, 0, -1])   # y轴向下
# view_ctl.set_lookat([0, 0, 0])   # 环视的中心点，即原点

for i, pc in enumerate(point_clouds):
        pcd.points = o3d.utility.Vector3dVector(pc)
        colors = generate_point_cloud_colors(pc)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        output_filename = "gt{}.ply".format(i)  # 可以选择不同的文件格式，例如 .pcd, .xyz, 等
        o3d.io.write_point_cloud(output_filename, pcd)
        # 刷新点云
        # vis.update_geometry(pcd)
        # vis.poll_events()
        # vis.update_renderer()
        # # if i == 0: time.sleep(5) 
        # time.sleep(10)  # 模拟帧率，可根据实际情况调整

# vis.destroy_window()

