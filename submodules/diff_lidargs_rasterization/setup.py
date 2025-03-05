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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
'''
v 0.0.0 完成forward和backward的重构和推导 rangeview
v 0.0.1 修复忽略depth梯度回传的问题
v 0.0.2+ 重构渲染 单位球切面坐标系渲染
.
.
.
(修复梯度bug,运算bug等)
v 0.0.9 
(整个pipeline, intensity+depth+raydrop)
v 1.0.0
'''
setup(
    name="diff_lidargs_rasterization",
    packages=['diff_lidargs_rasterization'],
    version='1.1.6', 
    ext_modules=[
        CUDAExtension(
            name="diff_lidargs_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
