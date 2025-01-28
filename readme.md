<h1 align="center">LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting</h1>
<!-- <h3 align="center">[CVPR 2024 - Highlight]</h3> -->
<p align="center">
   <a href="https://arxiv.org/abs/2410.05111.pdf">
      <img src='https://img.shields.io/badge/paper-pdf-green?style=for-the-badge' alt='Paper PDF'></a>
</p>
<!-- <p align="center">
   <a href="https://scholar.google.com.hk/citations?user=1ltylFwAAAAJ&hl=zh-CN&oi=sra">Tao Tang</a>
   ·
   <a href="https://wanggrun.github.io/">Guangrun Wang</a>
   ·
   <a href="https://scholar.google.com/citations?user=2w9VSWIAAAAJ&hl=en">Yixing Lao</a>
   ·
   <a href="https://damo.alibaba.com/labs/intelligent-transportation">Peng Chen</a>
   ·
   <a href="">Jie Liu</a>
    ·
   <a href="https://www.sysu-hcp.net/faculty/lianglin.html">Liang Lin</a>
   ·
   <a href="https://scholar.google.com.hk/citations?user=Jtmq_m0AAAAJ&hl=zh-CN&oi=sra">Kaicheng Yu</a>
   ·
   <a href="https://scholar.google.com/citations?user=voxznZAAAAAJ">Xiaodan Liang</a> -->
<p align="center">
<img src="./assets/teaser.png" alt="lidargs" style="zoom: 100%;" />
<img src="./assets/overview.png" alt="lidargs" style="zoom: 100%;" />
</p>

**[Abstract]**: LiDAR simulation plays a crucial role in closed-loop simulation for autonomous driving. Although recent advancements, such as the use of reconstructed mesh and Neural Radiance Fields (NeRF), have made progress in simulating the physical properties of LiDAR, these methods have struggled to achieve satisfactory frame rates and rendering quality. To address these limitations, we present **LiDAR-GS**, the first LiDAR Gaussian Splatting method, for real-time high-fidelity re-simulation of LiDAR sensor scans in public urban road scenes. The vanilla Gaussian Splatting, designed for camera models, cannot be directly applied to LiDAR re-simulation. To bridge the gap between passive camera and active LiDAR, our LiDAR-GS designs a differentiable laser beam splatting, grounded in the LiDAR range view model. This innovation allows for precise surface splatting by projecting lasers onto micro cross-sections, effectively eliminating artifacts associated with local affine approximations. Additionally, LiDAR-GS leverages Neural Gaussian Fields, which further integrate view-dependent clues, to represent key LiDAR properties that are influenced by the incident angle and external factors. Combining these practices with some essential adaptations, e.g., dynamic instances decomposition, our approach succeeds in simultaneously re-simulating depth, intensity, and ray-drop channels, achieving state-of-the-art results in both rendering frame rate and quality on publically available large scene datasets. 

## Updates
- [2025-01-28] 🎉🧧 Happy New Year's Eve! The core code is publicly available.

## Notes
Due to the following two reasons, the various functionalities of the code will be released progressively. \
1、The paper is under review \
2、
Information security review of code

Updated gradually: \
 1、Dynamic mode has not been updated yet \
2、More guides are being prepared \
3、The training of LiDAR-GS (2DGS version) 


## Video
[more results](https://github.com/cjlunmh/LiDAR-GS/blob/main/assets/video.mp4)

## Enrionment setup 
Ref to [dockerfile](https://github.com/cjlunmh/LiDAR-GS/blob/main/Dockerfile)

Then 
```
pip install submodules/simple-knn
pip install submodules/diff_lidargs_rasterization
(optional)
pip install submodules/diff_lidargs_surfel_rasterization 
```

## Prepare Dataset
Details coming soon ...

- Static dataset
  - ref to [Lidar-NeRF](https://github.com/tangtaogo/lidar-nerf)
- Dynamic dataset
  - ref to [DyNFL](https://github.com/prs-eth/Dynamic-LiDAR-Resimulation)


## Train
Guidance is coming soon ...

- Training of static dataset:
  ```
  bash train.sh
  ```
- Training of dynamic dataset: ...
- Inference: ...


## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@article{chen2024lidargs,
  title={LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting},
  author={Qifeng Chen, Sheng Yang, Sicong Du, Tao Tang, Peng Chen, Yuchi Huo},
  journal={arXiv preprint arXiv:2410.05111 },
  year={2024}
}
```



## Acknowledgments
We thank all authors from 
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
- [Lidar-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [AlignMiF](https://github.com/tangtaogo/alignmif)
- [DyNFL](https://github.com/prs-eth/Dynamic-LiDAR-Resimulation)

for presenting such an excellent work.