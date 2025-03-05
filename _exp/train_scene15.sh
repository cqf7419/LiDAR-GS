cd /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS
iterations=4000
output_dir="outputs"

gpu=0

data="/mnt_gx/ziqian_data/regroup_92171"
para_lane_scene="scene_15"
para_lane_track_lists=("track_1" "track_2") ## "track_0" 

caseid="pesudo"

for para_lane_track_list in "${para_lane_track_lists[@]}"; do
    echo "$para_lane_track_list"
    casename="$para_lane_scene/$para_lane_track_list"
    logdir="$casename/TOP"
    sensorid=0
    python3 train.py -s ${data} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir}/${logdir}

    logdir="$casename/BACK"
    sensorid=1
    python3 train.py -s ${data} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir}/${logdir}

    logdir="$casename/LEFT"
    sensorid=3
    python3 train.py -s ${data} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir}/${logdir}

    logdir="$casename/RIGHT"
    sensorid=4
    python3 train.py -s ${data} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir}/${logdir}

done