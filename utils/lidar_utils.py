import torch
import numpy as np


from extern.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extern.fscore import fscore
import os
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
    print("32线雷达",len(beam_inclinations))

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
                                   max_depth=80):

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

    def update(self, preds, truths):
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




def get_beam_inclinations(fov_up,fov,H):
    j = np.arange(H,dtype=np.float32)
    alpha = (fov_up - j/H*fov)/180*np.pi
    return alpha[::-1]