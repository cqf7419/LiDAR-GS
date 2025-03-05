cd /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS

iterations=4000
output_dir="./outputs"
gpu=0

casename="waymo"
data="/mnt_gx/lidar_data/datapublic_old/waymo_train"

caseid="segment-1083056852838271990_4080_000_4100_000_with_camera_labels"
logdir="$casename/$caseid"
sensorid=0
python3 train.py -s ${data} --caseid ${caseid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir}/${logdir} --max_depth 80


