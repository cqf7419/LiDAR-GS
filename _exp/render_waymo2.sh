cd /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS

python render.py \
-s /mnt_gx/lidar_data/datapublic_old/waymo_train \
-m /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS/outputs/waymo/segment-1005081002024129653_5313_150_5333_150_with_camera_labels \
--iteration 4000 \
--caseid segment-1005081002024129653_5313_150_5333_150_with_camera_labels \
--sensorid 0 \
--blockinfo /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS/outputs/waymo/segment-1005081002024129653_5313_150_5333_150_with_camera_labels