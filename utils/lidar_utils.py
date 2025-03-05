import torch
import numpy as np


from extern.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extern.fscore import fscore
import os
import math
import yaml
from scipy.spatial.transform import Rotation

def filter_pcd(pcd, r = 0.35, d = 3):
    from scipy.spatial import KDTree
    radius = r
    tree = KDTree(pcd)
    density = np.array([len(tree.query_ball_point(point,radius)) for point in pcd])
    condition = density>=d
    marked = np.zeros_like(density,dtype=bool)
    marked[condition] = True
    # print(marked.shape)
    return marked

def cal_beam_inclinations():
    '''
    根据gt2上使用的helios 5515的硬件规则生成 fov
    ''' 
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


    result = []
    for x in beam_inclinations:
        result.append(math.radians(x))#(x*math.pi/180.)
    # print(result)
    return np.array(result)

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
                                   max_depth=80,
                                   ground=None,
                                   is_correction=False,
                                   sensor_id=None,
                                   pre_labels = None,
                                   s2b = None):

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

    if ground is None:
        ground = np.zeros(local_point_intensities.shape, dtype=bool)
    if pre_labels is None:
        pre_labels = np.ones(local_point_intensities.shape)*100
    
    # print("[ debug ] ground_correction.shape and type ",ground.shape,type(ground))
    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    mask = np.zeros((lidar_H, lidar_W))
    for (local_points, dist, local_point_intensity, is_ground, pre_label) in zip(
            local_points,
            dists,
            local_point_intensities,
            ground,
            pre_labels
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
            if is_correction : # 只对地面做个简单的矫正 其他的扫描可能得从运动本身去估计
                if is_ground : 
                    if sensor_id==0 and alpha<0: # 主雷达（前）是平着装的 不需要额外考虑安装角度
                        dist = np.abs(dist*np.sin(alpha) / (np.abs(np.sin(beam_inclinations[r]))+1e-16))
                        if dist >= max_depth:
                            continue
                    else:
                        if s2b is not None:
                            tmp = local_points @ (s2b.T)[:3,:3] # 旋转到baselidar系
                            new_alpha = np.arctan2(tmp[2], np.sqrt(tmp[0]**2 + tmp[1]**2)) 
                            delta_alpha = new_alpha - alpha
                            if new_alpha<0:
                                dist = np.abs(dist*np.sin(alpha+delta_alpha) / (np.abs(np.sin(beam_inclinations[r]+delta_alpha))+1e-16))
                                if dist >= max_depth:
                                    continue
                        
            r = lidar_H - r - 1 
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
            mask[r, c] = 1 if pre_label>8 else 0
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
            mask[r, c] = 1 if pre_label>8 else 0

    return pano, intensities, mask
def lidar_to_pano_with_grad(pcd: torch.tensor,
                                   grad:torch.tensor,
                                   lidar_H: int,
                                   lidar_W: int,
                                   lidar_K=None,
                                   cam_pos = None,
                                   beam_inclination=None,
                                   max_depth=80):

    local_points = (pcd[:, :3] - cam_pos).detach().cpu().numpy()
    local_point_gradx = grad[:, 0].detach().cpu().numpy() # W 
    local_point_grady = grad[:, 1].detach().cpu().numpy() # H
    beam_inclinations = beam_inclination.detach().cpu().numpy()
    if beam_inclinations is not None:
        use_beam_inclinations = True
    else:
        use_beam_inclinations = False
        fov_up, fov = lidar_K
        fov_down = fov - fov_up


    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for (local_points, gradx, grady) in zip(
            local_points,
            local_point_gradx,
            local_point_grady,
    ):
        # Check max depth.


        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        c = int(round(beta / (2 * np.pi / lidar_W)))

        if use_beam_inclinations:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2))
            r = find_closest_label(beam_inclinations, alpha)
            r = lidar_H - r - 1
        else:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
            r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = gradx 
            intensities[r, c] = grady
        else:
            pano[r, c] = max(gradx,pano[r, c])
            intensities[r, c] = max(grady,intensities[r, c])

    return pano, intensities


def pano_to_lidar_with_intensities(pano: np.ndarray,
                                   intensities,
                                   lidar_K=None,
                                   beam_inclinations=None):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)
        beam_inclinations: beam_inclinations (H,)

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """

    H, W = pano.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    beta = -(i - W / 2.0) / W * 2.0 * np.pi
    if beam_inclinations is not None:
        alpha = np.expand_dims(beam_inclinations[::-1], 1).repeat(W, 1)
    else:
        fov_up, fov = lidar_K
        alpha = (fov_up - j / H * fov) / 180.0 * np.pi
    dirs = np.stack([
        np.cos(alpha) * np.cos(beta),
        np.cos(alpha) * np.sin(beta),
        np.sin(alpha),
    ], -1)
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2)

    # Filter empty points.
    idx = np.where(pano != 0.0)
    
    local_points_with_intensities = local_points_with_intensities[idx]
    # print("pano shape: ",local_points_with_intensities.shape)
    return local_points_with_intensities

def pano_to_lidar(pano, lidar_K=None, beam_inclinations=None):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
        beam_inclinations=beam_inclinations,
    )
    return local_points_with_intensities[:, :3]

def load_extrinsics(yaml_item):
    t=np.array(yaml_item["translation"]).astype(float)
    q=np.array(yaml_item["rpy"]).astype(float)
    r = Rotation.from_euler('xyz', (q[0], q[1], q[2]), degrees=True).as_matrix()
    T=np.hstack((r, t.reshape(3,1)))
    T=np.vstack((T, np.array([[0,0,0,1]])))
    return T

def load_yaml_str(calibration_file_path):
    param = open(calibration_file_path, "r").read()
    if not param:
        return ''
    if "%YAML:1.0" in param:
        param=param.replace("%YAML:1.0", "%YAML 1.0")
    param = param.replace('...', '').replace('!!opencv-matrix', '').replace('!<tag:yaml.org,2002:opencv-matrix>', '')
    params = yaml.load(param, Loader=yaml.FullLoader)
    return params


class PointsMeter:

    def __init__(self, scale, intrinsics, beam_inclinations=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics
        self.beam_inclinations = beam_inclinations

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths, Filter = False):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths)  # [B, H, W]
        chamLoss = chamfer_3DDist()
        pred_lidar = pano_to_lidar(pano=preds[0],
                                   lidar_K=self.intrinsics,
                                   beam_inclinations=self.beam_inclinations)
        gt_lidar = pano_to_lidar(pano=truths[0],
                                 lidar_K=self.intrinsics,
                                 beam_inclinations=self.beam_inclinations)
        if Filter:
            mask_raydrop = filter_pcd(pred_lidar)
            pred_lidar = pred_lidar[mask_raydrop]
            
        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(pred_lidar[None, ...]).cuda(),
            torch.FloatTensor(gt_lidar[None, ...]).cuda())
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "CD"),
                          self.measure()[0], global_step)

    def report(self):
        return f'CD f-score = {self.measure()}'


def write_pcd(save_filename, points, utime1=None, utime2=None, distance=None, ring=None, intensity=None, semantic_flag=None):
    pcd_header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z utime1 utime2 distance ring intensity semantic_flag\nSIZE 4 4 4 4 4 4 2 1 1\nTYPE F F F I I F U U U\nCOUNT 1 1 1 1 1 1 1 1 1\nWIDTH 98765\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 98765\nDATA ascii"
    if save_filename.split(".")[-1] != "pcd":
        raise ValueError("InvalidFileExtensionError")
    if utime1 is None or utime1.shape[0] != points.shape[0]:
        utime1 = np.zeros((points.shape[0], 1))
        # print("utime1 data Error")
    if utime2 is None or utime2.shape[0] != points.shape[0]:
        utime2 = np.zeros((points.shape[0], 1))
        # print("utime2 data Error")
    if distance is None or distance.shape[0] != points.shape[0]:
        distance = np.zeros((points.shape[0], 1))
        # print("distance data Error")
    if ring is None or ring.shape[0] != points.shape[0]:
        ring = np.zeros((points.shape[0], 1))
        # print("ring data Error")
    if intensity is None or intensity.shape[0] != points.shape[0]:
        intensity = np.zeros((points.shape[0], 1))
        # print("intensity data Error")
    if semantic_flag is None or semantic_flag.shape[0] != points.shape[0]:
        semantic_flag = np.zeros((points.shape[0], 1))
        # print("semantic_flag data Error")
    
    data = np.concatenate((points[:, :3], utime1, utime2, distance, ring, intensity, semantic_flag), axis=1)
    pcd_header=pcd_header.replace("98765", str(data.shape[0]))
    np.savetxt(save_filename, data, header=pcd_header, comments="", fmt='%.3f')

def write_pcdi(save_filename, pcdi, selected_lidar, distance=None):
    points = pcdi[:,:3]
    intensity = pcdi[:,3:4]*255
    # distance = np.linalg.norm(points[...,:3], axis=1).reshape(points.shape[0],1)
    ring = np.full(intensity.shape, selected_lidar)
    write_pcd(
        save_filename,
        points,
        ring = ring,
        intensity = intensity.astype(np.uint8),
        distance = distance
    )
    
def get_pcdi(pcdi, selected_lidar, distance=None):
    points = pcdi[:,:3]
    intensity = pcdi[:,3:4]*255
    ring = np.full(intensity.shape, selected_lidar)
    data = get_pcd(
        points,
        ring = ring,
        intensity = intensity.astype(np.uint8),
        distance=distance
    )
    
    return data


def get_pcd(points, utime1=None, utime2=None, distance=None, ring=None, intensity=None, semantic_flag=None):
    if utime1 is None or utime1.shape[0] != points.shape[0]:
        utime1 = np.zeros((points.shape[0], 1))
        # print("utime1 data Error")
    if utime2 is None or utime2.shape[0] != points.shape[0]:
        utime2 = np.zeros((points.shape[0], 1))
        # print("utime2 data Error")
    if distance is None or distance.shape[0] != points.shape[0]:
        distance = np.zeros((points.shape[0], 1))
        # print("distance data Error")
    if ring is None or ring.shape[0] != points.shape[0]:
        ring = np.zeros((points.shape[0], 1))
        # print("ring data Error")
    if intensity is None or intensity.shape[0] != points.shape[0]:
        intensity = np.zeros((points.shape[0], 1))
        # print("intensity data Error")
    if semantic_flag is None or semantic_flag.shape[0] != points.shape[0]:
        semantic_flag = np.zeros((points.shape[0], 1))
        # print("semantic_flag data Error")
    
    data = np.concatenate((points[:, :3], utime1, utime2, distance, ring, intensity, semantic_flag), axis=1)
    return data
