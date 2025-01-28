import numpy as np
import math


def cal_beam_inclinations():
    # 根据gt2上使用的helios 5515的硬件规则生成
    # fov 
    beam_inclinations = []
    level1 = np.linspace(-55, -10, num=15, endpoint=False)
    beam_inclinations.extend(list(level1))
    level2 = np.linspace(-10, -8, num=1, endpoint=False)
    beam_inclinations.extend(list(level2))
    level3 = np.linspace(-8, 4, num=9, endpoint=False)
    beam_inclinations.extend(list(level3))
    level4 = np.linspace(4, 7, num=2, endpoint=False)
    beam_inclinations.extend(list(level4))
    level5 = np.linspace(7, 15, num=5)
    beam_inclinations.extend(list(level5))
    # print(beam_inclinations)
    # print("32线雷达",len(beam_inclinations))

    result = []
    for x in beam_inclinations:
        result.append(math.radians(x))#(x*math.pi/180.)
    # print(result)
    return np.array(result).tolist()

def find_closest_label(beam_labels, angle):
    from bisect import bisect_left
    if (angle >= beam_labels[-1]):
        return len(beam_labels) - 1
    elif angle <= beam_labels[0]:
        # return beam_labels[0]
        return 0
    pos = bisect_left(beam_labels, angle)
    before = beam_labels[pos - 1]
    after = beam_labels[pos]
    if after - angle < angle - before:
        # return after
        return pos
    else:
        # return before
        return pos - 1


def lidar_to_pano_with_intensities(local_points_with_intensities: np.ndarray,
                                   lidar_H: int,
                                   lidar_W: int,
                                   lidar_K=None,
                                   cam_pos = None,
                                   beam_inclinations=None,
                                   max_depth=80):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        beam_inclinations:beam_inclinations.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    usage:
        lidar_to_pano_with_intensities(
                                        points,
                                        H=32,
                                        W=1800,
                                        beam_inclinations,
                                        max_depth=150
        )
        返回每帧pcd的rangeview
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    if beam_inclinations is not None:
        use_beam_inclinations = True
    else:
        use_beam_inclinations = False
        fov_up, fov = lidar_K
        fov_down = fov - fov_up

    # Compute dists to lidar center.
    if cam_pos != None:
        dists = np.linalg.norm(local_points-cam_pos, axis=1)
    else:
        dists = np.linalg.norm(local_points, axis=1)


    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for (local_points, dist, local_point_intensity) in zip(
            local_points,
            dists,
            local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        c = int(round(beta / (2 * np.pi / lidar_W)))

        if use_beam_inclinations:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2))
            r = find_closest_label(beam_inclinations, alpha)
            r = lidar_H - r
        else:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
            r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities
'''
process:
    1、 珊格化lidar fov
    2、 计算每个高斯到相机中心的方向，即可求的高斯中心所在rangeview曲面珊格位置
    3、 对于单一一个珊格，按照深度排序投进来的高斯
    4、 根据透明度渲染深度和intensity（当作rgb）
'''
# # [0.698746,1.852252,-2.116155] [0.706313, 1.852198, -2.119826]   [0.711732,1.845759,-2.115992]
# p1 = np.array([0.698746,1.852252,-2.116155],dtype=np.float32)
# p2 = np.array([0.706313, 1.852198, -2.119826],dtype=np.float32)
# p3 = np.array([0.711732,1.845759,-2.115992],dtype=np.float32)
# p4 = np.array([28.397535,-7.744916,8.693146],dtype=np.float32)
# beta1 = np.pi - np.arctan2(p1[1], p1[0])
# beta2 = np.pi - np.arctan2(p2[1], p2[0])
# beta3 = np.pi - np.arctan2(p3[1], p3[0])
# print(beta1,beta2,beta3)
# print(beta2-beta1,beta3-beta2)
# alpha = np.arctan2(p4[2], np.sqrt(p4[0]**2 + p4[1]**2))
# print(math.degrees(alpha))

# H = 32
# W = 1800
# intrinsics = (-55, 15)  # fov_up, fov
# l_beam_inclinations = cal_beam_inclinations() # helio 5515
# beam_inclinations = np.array(l_beam_inclinations)
# # lidar_paths = [os.path.join(dataspace, data['frames'][i]['path']['pcd']) for i in range(frame_num)]
# generate_train_data(H=H,
#                     W=W,
#                     beam_inclinations=beam_inclinations,
#                     lidar_paths=lidar_paths,
#                     out_dir=out_path,
#                     points_dim=7,
#                     max_depth=150) # helio max 150


## test waymo
calib_path = '/home/xuanyuan/lansheng/waymo_test/0000000.txt'
with open(calib_path, 'r') as f:
    lines = f.readlines()
beam_inclinations = np.array(
    [float(info) for info in lines[5].split(' ')[1:]])
# lidar_path = '/home/xuanyuan/lansheng/waymo_test/0000000.bin'
lidar_path = "/home/xuanyuan/lansheng/Scaffold-GS-lidar/data/waymo/waymo_extract/lidar_0/"+"0000000"+".bin"
point_cloud = np.fromfile(lidar_path, dtype=np.float32)
point_cloud = point_cloud.reshape((-1, 6))
pano, intensities = lidar_to_pano_with_intensities(
    local_points_with_intensities=point_cloud,
    lidar_H=64,
    lidar_W=2650,
    # lidar_K=intrinsics,
    beam_inclinations=beam_inclinations,
    max_depth=80,
)
range_view = np.zeros((64, 2650, 3))
range_view[:, :, 1] = intensities
range_view[:, :, 2] = pano

# import matplotlib.pyplot as plt
# plt.imshow(range_view[:,:,2], cmap='gray')
# plt.colorbar(label='Distance (m)')
# plt.show()

range_view_dist = range_view[:, :, 2]*255/80. 
import cv2
cv2.imwrite('range_view_dist.png', range_view_dist)
