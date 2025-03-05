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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.lovasz_softmax import Lovasz_softmax,lovasz_softmax_flat

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def get_ce_weights(gt_label, n_classes, max_weights = 50):
    _EPS = 1e-20  # To prevent division by zero
    # get inverse_frequency of each class from ground truth label
    counts =[]
    device = gt_label.device
    for label in range(n_classes):
        counts.append((gt_label == label).sum().item()+_EPS)
    counts = torch.tensor(counts).to(device)
    inv_freq = counts.sum() / counts
    seg_weight = torch.clamp(torch.sqrt(inv_freq), 0, max_weights)
    return seg_weight 

def raydrop_lossf(est,gt,lambda_bce=0.15,lambda_lov=0.15,reweight=True):
    """
        est: [B,C]
        gt : [B,]
        lambda_bce: 0.15 (default)
        lambda_lov: 0.15 (default)
        reweight : True(default)
    """
    # compute weights in an online fashion
    if reweight:
        seg_weights = get_ce_weights(gt, 2)
        # print("ce loss weight:",seg_weights.shape)
        criterion = torch.nn.CrossEntropyLoss(weight=seg_weights, ignore_index=-1)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # print("raydrop loss input : est shape ",est.shape)
    # print("raydrop loss input : gt shape ",gt.shape)
    ce_loss = criterion(est, gt)

    # score_softmax = self.softmax(est)
    score_softmax = est.softmax(dim=1)
    lovasz_loss = lovasz_softmax_flat(score_softmax, gt)

    return lambda_bce*ce_loss + lambda_lov*lovasz_loss