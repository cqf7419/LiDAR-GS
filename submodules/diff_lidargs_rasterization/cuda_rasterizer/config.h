/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 2 // using 2, lidar intensity + ray drop probability / Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 1
// #define MAX_DEPTH 80.0
// #define Ray_Divergence_Angle 0.002  // 2 mrad

#endif