#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, model_id, waymo_dynamic_model, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.block_id = args.block_id

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

        self.train_cameras = {}
        self.test_cameras = {}
        self.total_cameras = {}
        
        self.init_status = True
        scene_info = sceneLoadTypeCallbacks["GT"](model_id, waymo_dynamic_model, args.source_path, args.white_background, args.eval, args.newcar_render)
        
        if scene_info is None: 
            self.init_status = False
            return

        # else:
        #     assert False, "Could not recognize scene type!"

        self.gaussians.set_appearance(len(scene_info.total_cameras))
        
        self.cameras_extent = 1.0 #scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            self.total_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.total_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                            str(model_id),
                                                            str(self.block_id),
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                            str(model_id),
                                                           str(self.block_id),
                                                           "iteration_" + str(self.loaded_iter)), mode = 'unite')
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, model_id):
        point_cloud_path = os.path.join(self.model_path, str(model_id), str(self.block_id) ,"iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path, mode = 'unite')

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTotalCameras(self, scale=1.0):
        return self.total_cameras[scale]

# 存下每一帧的timestep 去确认obj出现在哪些帧