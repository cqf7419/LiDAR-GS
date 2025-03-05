import os
import torch
from utils.general_utils import get_expon_lr_func, build_rotation, quaternionRawMultiply


class PoseCorrection(torch.nn.Module):
    def __init__(self, num_poses):
        super().__init__()        
        self.pose_correction_trans = torch.nn.Parameter(torch.zeros(num_poses, 3).float().cuda()).requires_grad_(True)
        self.pose_correction_rots = torch.nn.Parameter(torch.tensor([[1, 0, 0, 0]]).repeat(num_poses, 1).float().cuda()).requires_grad_(True)

    def save_state_dict(self, path):
        state_dict = dict()
        state_dict['params'] = self.state_dict()
        torch.save(state_dict, os.path.join(path, "pose_correction.pth"))

    def load_state_dict(self, path):
        state_dict = torch.load(os.path.join(path, "pose_correction.pth"))
        super().load_state_dict(state_dict['params'])

    def training_setup(self):
        pose_correction_lr_init = 5e-6
        pose_correction_lr_final = 1e-6

        params = [
            {'params': [self.pose_correction_trans], 'lr': pose_correction_lr_init, 'name': 'pose_correction_trans'},
            {'params': [self.pose_correction_rots], 'lr': pose_correction_lr_init, 'name': 'pose_correction_rots'},
        ]        
        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-8, weight_decay=0.01)
        self.pose_correction_scheduler_args = get_expon_lr_func(
            lr_init=pose_correction_lr_init,
            lr_final=pose_correction_lr_final
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.pose_correction_scheduler_args(iteration)
            param_group['lr'] = lr
    
    def update_optimizer(self):
        self.optimizer.step()       
        self.optimizer.zero_grad(set_to_none=None)

    def forward(self, cam_id):
        pose_correction_trans = self.pose_correction_trans[cam_id]
        pose_correction_rot = self.pose_correction_rots[cam_id]
        pose_correction_rot = torch.nn.functional.normalize(pose_correction_rot.unsqueeze(0))
        pose_correction_rot = build_rotation(pose_correction_rot).squeeze(0)
        pose_correction_matrix = torch.cat([pose_correction_rot, pose_correction_trans[:, None]], dim=-1)
        padding = torch.tensor([[0, 0, 0, 1]]).float().cuda()
        pose_correction_matrix = torch.cat([pose_correction_matrix, padding], dim=0)
        return pose_correction_matrix

    def correct_gaussian_xyz(self, cam_id, xyz):
        pose_correction_trans = self.pose_correction_trans[cam_id]
        pose_correction_rot = self.pose_correction_rots[cam_id]
        pose_correction_rot = torch.nn.functional.normalize(pose_correction_rot.unsqueeze(0), dim=-1)
        pose_correction_rot = build_rotation(pose_correction_rot).squeeze(0)
        pose_correction_matrix = torch.cat([pose_correction_rot, pose_correction_trans[:, None]], dim=-1)
        padding = torch.tensor([[0, 0, 0, 1]]).float().cuda()
        pose_correction_matrix = torch.cat([pose_correction_matrix, padding], dim=0)
        xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) 
        xyz = xyz @ pose_correction_matrix.T
        xyz = xyz[:, :3]
        return xyz

    def correct_gaussian_rotation(self, cam_id, rotation):
        pose_correction_rot = self.pose_correction_rots[cam_id]
        pose_correction_rot = torch.nn.functional.normalize(pose_correction_rot.unsqueeze(0), dim=-1)
        rotation = quaternionRawMultiply(pose_correction_rot, rotation)
        return rotation

    def regularization_loss(self):
        loss_trans = torch.abs(self.pose_correction_trans).mean()
        rots_norm = torch.nn.functional.normalize(self.pose_correction_rots, dim=-1)
        loss_rots = torch.abs(rots_norm - torch.tensor([[1, 0, 0, 0]]).float().cuda()).mean()
        loss = loss_trans + loss_rots
        return loss
