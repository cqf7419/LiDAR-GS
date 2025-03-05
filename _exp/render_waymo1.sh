cd /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS

python render.py \
-s /mnt_gx/lidar_data/datapublic_old/waymo_train \
-m /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS/outputs/waymo/segment-1083056852838271990_4080_000_4100_000_with_camera_labels \
--iteration 3000 \
--caseid segment-1083056852838271990_4080_000_4100_000_with_camera_labels \
--sensorid 0 \
--blockinfo /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS/outputs/waymo/segment-1083056852838271990_4080_000_4100_000_with_camera_labels