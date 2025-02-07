import numpy as np
from pathlib import Path
import json
import torch
import pandas as pd
import os

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

class waymo_dynamic:
    def __init__(self, context_dir, dtype=np.float32):
        self.context_dir = Path(context_dir)
        self.scene_size = 50

        range_images = np.load(self.context_dir/'range_images1.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:]
        self.first_masks = np.where(range_images[:,:,:,0]>0, True, False)
        self.first_dist = range_images[:,:,:,0]
        self.first_intensity = np.tanh(range_images[:,:,:,1])
        self.first_elongation = range_images[:,:,:,2]
        self.ray_object_indices = np.load(self.context_dir/'ray_object_indices.npy', allow_pickle=True)[:self.scene_size,:,:]
        self.normals = np.load(self.context_dir/'normals.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:,:]
        self.ray_origins = np.load(self.context_dir/'ray_origins.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:,:]
        self.ray_dirs = np.load(self.context_dir/'ray_dirs.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:,:]
        self.valid_normal_flag = np.load(self.context_dir/'valid_normal_flags.npy', allow_pickle=True)[:self.scene_size,:,:]
        self.objects_id_2_tsfm = np.load(self.context_dir/'objects_id_2_tsfm.npy', allow_pickle=True).item()
        self.objects_id_types_per_frame = np.load(self.context_dir/'objects_id_types_per_frame.npy', allow_pickle=True)
        self.objects_id_2_corners = np.load(self.context_dir/'objects_id_2_corners.npy', allow_pickle=True).item()
        self.objects_id_2_anchors = np.load(self.context_dir/'objects_id_2_anchors.npy', allow_pickle=True).item()
        self.objects_id_2_frameidx = np.load(self.context_dir/'objects_id_2_frameidx.npy', allow_pickle=True).item()
        self.objects_id_2_dynamic_flag = np.load(self.context_dir/'objects_id_2_dynamic_flag.npy', allow_pickle=True).item()
        # print(list(self.objects_id_2_dynamic_flag.keys()))
        self.object_ids_per_frame = np.load(self.context_dir/'object_ids_per_frame.npy', allow_pickle=True)

        df = pd.read_parquet(self.context_dir/"training_lidar_calibration.parquet", engine="pyarrow",columns=["[LiDARCalibrationComponent].beam_inclination.values"])
        self.beam_inclinations = df.iloc[4,0]
        self.l2w = []
        pcds = []
        with open(self.context_dir/'meta_info.json') as file:
            contents = json.load(file)

        self.first_vehicle2world = None
        for i in range(self.scene_size):
            frames = contents["frames"] # 共199帧
            frame = frames[i+50]
            l2w_M = np.array(frame["lidar2world"])
            self.l2w.append(l2w_M)

        self.map_id_2_type()
        self.convert_dynamic_object_id_2_global_idx()
        self.create_aabb_of_anchor_boxes_of_dynamic_vehicles()
        self.create_tsfm_for_object_idx_at_every_frame()


    def getlidar2world(self):
        return self.l2w


    def map_id_2_type(self):
        ## map a string id to int types
        self.object_id_2_type = {}
        for frame_idx in range(self.scene_size):
            for object_idx in range(len(self.objects_id_types_per_frame[frame_idx])):
                object_id = self.object_ids_per_frame[frame_idx][object_idx]
                object_type = self.objects_id_types_per_frame[frame_idx][object_idx]
                self.object_id_2_type.update({object_id:object_type})


    def convert_dynamic_object_id_2_global_idx(self):
        ###mapping string ids to int indices
        dynamic_object_counter=0
        object_id_2_global_idx = {}
        for frame_idx in range(self.scene_size):
            for object_id in self.object_ids_per_frame[frame_idx]:
                dynamic_flag = self.objects_id_2_dynamic_flag[object_id] if object_id in self.objects_id_2_dynamic_flag.keys() else False
                object_type = self.object_id_2_type[object_id] if object_id in self.object_id_2_type.keys() else -1
                if object_id not in object_id_2_global_idx.keys() and dynamic_flag and object_type==1:
                    object_id_2_global_idx.update({object_id:dynamic_object_counter})
                    dynamic_object_counter+=1
        self.object_id_2_global_idx = object_id_2_global_idx   
        
        # print("object_id_2_global_idx ", object_id_2_global_idx)
        
        self.dynamic_object_counter = dynamic_object_counter   
        print(f"detect {self.dynamic_object_counter} dynamic vehicles in the dataset")

    def create_aabb_of_anchor_boxes_of_dynamic_vehicles(self):
        self.aabb_vehicle = [torch.randn(6) for i in range(self.dynamic_object_counter)]
        for object_id in self.object_id_2_global_idx.keys():
            # anchor_bbox = self.objects_id_2_corners[object_id][0]
            anchor_bbox = self.objects_id_2_anchors[object_id]
            min = np.min(anchor_bbox,axis=0)
            max = np.max(anchor_bbox,axis=0)
            aabb = torch.tensor([min[0], min[1], min[2], max[0], max[1], max[2]])    
            self.aabb_vehicle[self.object_id_2_global_idx[object_id]] = aabb

    def create_tsfm_for_object_idx_at_every_frame(self):
        self.tsfm_vehicle = torch.randn(self.scene_size, self.dynamic_object_counter,4,4)
        self.mask_tsfm_vehicle = torch.zeros(self.scene_size, self.dynamic_object_counter).bool()
        for object_id in self.object_id_2_global_idx.keys():
            vehicle_idx = self.object_id_2_global_idx[object_id]
            occurred_frames = self.objects_id_2_frameidx[object_id]

            tsfms = self.objects_id_2_tsfm[object_id]
            for i in range(len(occurred_frames)):
                occurred_frame_idx = occurred_frames[i]
                tsfm = tsfms[i]

                self.tsfm_vehicle[occurred_frame_idx, vehicle_idx,:4,:4] = torch.tensor(tsfm)
                self.mask_tsfm_vehicle[occurred_frame_idx, vehicle_idx] = True


    def kabsch_transformation_estimation(self, x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0):
        if weights is None:
            weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

        if normalize_w:
            sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
            weights = (weights/sum_weights)

        weights = weights.unsqueeze(2)

        if best_k > 0:
            indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
            weights = weights[:,indices,:]
            x1 = x1[:,indices,:]
            x2 = x2[:,indices,:]

        if w_threshold > 0:
            weights[weights < w_threshold] = 0


        x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
        x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

        x1_centered = x1 - x1_mean
        x2_centered = x2 - x2_mean

        weight_matrix = torch.diag_embed(weights.squeeze(2))

        cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                            torch.matmul(weight_matrix, x2_centered))

        try:
            u, s, v = torch.svd(cov_mat)
        except Exception as e:
            r = torch.eye(3,device=x1.device)
            r = r.repeat(x1_mean.shape[0],1,1)
            t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

            return r, t, True

        tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

        determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

        rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

        # translation vector
        translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

        return rotation_matrix, translation_matrix, False


    def get_obj2world(self,occurred_frame_idx,object_id):        
        first_corners = self.objects_id_2_corners[object_id][occurred_frame_idx]
        x = np.linalg.norm(first_corners[0,:] - first_corners[4,:], axis=-1)
        y = np.linalg.norm(first_corners[0,:] - first_corners[3,:], axis=-1)
        z = np.linalg.norm(first_corners[0,:] - first_corners[1,:], axis=-1)
        anchor_corners = np.array([[0,0,0], [0,0,z],[0,y,z], [0,y,0],[x,0,0],[x,0,z],[x,y,z],[x,y,0]]) + np.mean(first_corners, axis=0)

        points_a = torch.from_numpy(first_corners)
        points_a = points_a.float()[None]
        
        points_b = torch.from_numpy(anchor_corners)
        points_b = points_b.float()[None]
        
        ##transform to aabb corners
        rotation_matrix, translation_matrix, _  =\
            self.kabsch_transformation_estimation(points_b, points_a)
        
        object_to_world = np.eye(4)
        object_to_world[:3 , :3] = rotation_matrix.numpy()
        object_to_world[:3 , 3] = first_corners[0,:]

        return object_to_world
    
    def get_mask(self,frame_idx,object_id):
        '''
        output:  static mask, dynamic vehice mask
        '''
        first_mask = self.first_masks[frame_idx]
        object_idx = self.ray_object_indices[frame_idx] # [H,W] 存放id， 若为-1则为背景 这个idx时全局的
        
        object_ids = self.object_ids_per_frame[frame_idx]
        object_ids = np.array(object_ids)
        idx = object_ids[object_idx]

        
        dynamic_flag = (idx==object_id)
        valid_normal_flag = self.valid_normal_flag[frame_idx]
        dynamic_only_mask = first_mask & dynamic_flag & valid_normal_flag

        static_mask = (~ dynamic_only_mask) & first_mask & valid_normal_flag
        return static_mask, dynamic_only_mask

    def get_static_mask(self,frame_idx):
        '''
        output:  all dynamic obj will be cut
        '''
        first_mask = self.first_masks[frame_idx]
        object_idx = self.ray_object_indices[frame_idx] # [H,W] 存放id， 若为-1则为背景 这个idx时全局的
        
        object_ids = self.object_ids_per_frame[frame_idx]
        object_ids = np.array(object_ids)
        idx = object_ids[object_idx]

        dynamic_flag = np.zeros(object_idx.shape,dtype=bool)
        obj_id_list = list(self.object_id_2_global_idx.keys())

        for id in obj_id_list:
            dynamic_flag |= (idx==id)
        valid_normal_flag = self.valid_normal_flag[frame_idx]
        dynamic_only_mask = first_mask & dynamic_flag & valid_normal_flag

        static_mask = (~ dynamic_only_mask) & first_mask & valid_normal_flag
        return static_mask

    def get_obj_frames(self,object_id):
        '''
        列表返回obj出现的帧
        '''
        occurred_frames = self.objects_id_2_frameidx[object_id]
        return occurred_frames

    def get_dynamic_obj_id_list(self):
        '''
        列表返回移动的obj的id
        '''
        return list(self.object_id_2_global_idx.keys())

    def get_all_obj_id_list(self):
        return list(self.objects_id_2_dynamic_flag.keys())
    
    def get_rangeview(self,frame_idx,H_lidar,W_lidar):
        range_view = np.zeros((64, 2650, 3))
        range_view[:, :, 1] = self.first_intensity[frame_idx]
        range_view[:, :, 2] = self.first_dist[frame_idx]
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
        return image_lidar

    def get_beam_inclination(self):
        df = pd.read_parquet(self.context_dir/"training_lidar_calibration.parquet", engine="pyarrow",columns=["[LiDARCalibrationComponent].beam_inclination.values"])
        return df.iloc[4,0]

    def get_static_pcd_each_frame(self,frame_id,lidar_to_world):
        lidar_point = pano_to_lidar(self.first_dist[frame_id], lidar_K=None, beam_inclinations=self.beam_inclinations)
        
        static_mask = self.get_static_mask(frame_id)
        static_mask = static_mask.reshape(-1)
        idx = np.where(static_mask != 0.0)
        lidar_point = lidar_point[idx]

        pcd_world = (np.pad(lidar_point[...,:3], ((0,0),(0, 1)), constant_values=1) @ lidar_to_world.T)[:,:3]
        return pcd_world

    def get_object_pcd_each_frame(self,obj_id,frame_id,lidar_to_object):
        lidar_point = pano_to_lidar(self.first_dist[frame_id], lidar_K=None, beam_inclinations=self.beam_inclinations)
        other_mask, obj_mask = self.get_mask(frame_id,obj_id)
        obj_mask = obj_mask.reshape(-1)
        idx = np.where(obj_mask != 0.0)
        lidar_point = lidar_point[idx]

        pcd_object = (np.pad(lidar_point[...,:3], ((0,0),(0, 1)), constant_values=1) @ lidar_to_object.T)[:,:3]

        return pcd_object


    def get_obj_pcd(self,frame_id,obj_id):
        lidar_point = pano_to_lidar(self.first_dist[frame_id], lidar_K=None, beam_inclinations=self.beam_inclinations)
        other_mask, obj_mask = self.get_mask(frame_id,obj_id)
        obj_mask = obj_mask.reshape(-1)
        idx = np.where(obj_mask != 0.0)
        return lidar_point[idx],lidar_point
    
    def get_static_pcd(self,frame_id):
        lidar_point = pano_to_lidar(self.first_dist[frame_id], lidar_K=None, beam_inclinations=self.beam_inclinations)
        static_mask = self.get_static_mask(frame_id)
        static_mask = static_mask.reshape(-1)
        idx = np.where(static_mask != 0.0)
        return lidar_point[idx]

# if __name__ == "__main__":
#     context_dir = "/mnt_gx/usr/lansheng/processed_data_dynamic/1005081002024129653_5313_150_5333_150"
#     data = waymo_dynamic(context_dir)
#     ids = data.get_dynamic_obj_id_list()
#     # print(data.get_all_obj_id_list())
#     # print(data.get_obj_frames(ids[0]))
#     s = data.get_static_mask(0)
#     import matplotlib.pyplot as plt
#     plt.imsave("static_all1.png",s,cmap='gray')


