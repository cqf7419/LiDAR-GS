# LiDAR-GS-Waymo
这是一个兼容waymo common dynamic的分支

# train
```
pip install submodules/diff_lidargs_surfel_rasterization
bash _exp/train_waymo1.sh
```

# Note
⚠ 内源
这是一个兼容更多waymo dynamic的支持长序列训练的测试版本,其中rasterization 使用了2dgs版本，还未经过广泛的测试 \
可以通过简单的安装原始版本的lidargs的3dgs版本的rasterization， 需要在gaussian model函数和render函数在mlp输出的地方修改一下维度即可（参考static lidargs，这可能需要你对GS代码有一些了解）

# data
链接: https://pan.baidu.com/s/1JDciX4Fw7qcMckXjlENX4Q 提取码: dbqx 

