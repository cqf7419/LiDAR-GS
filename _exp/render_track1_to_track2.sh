cd /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/LiDAR_GS
output_dir="outputs"
data="/mnt_gx/ziqian_data/regroup_92171"
para_lane_scenes=("scene_3" "scene_4" "scene_5" "scene_6" "scene_11" "scene_12" "scene_15" "scene_20" "scene_21" "scene_25" "scene_30" "scene_31" "scene_34" "scene_36" "scene_37")
para_lane_track_lists=("track_2")
caseid="pesudo"

# track0 to track1
for para_lane_scene in "${para_lane_scenes[@]}"; do
    echo "$para_lane_scene"
    for para_lane_track_list in "${para_lane_track_lists[@]}"; do
        echo "$para_lane_track_list"
        casename="$para_lane_scene/track_1"
        logdir="$casename/TOP"
        blockinfo="$output_dir/$para_lane_scene/$para_lane_track_list/TOP"
        sensorid=0
        python3 render.py -s ${data} -m ${output_dir}/${logdir} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --iteration 4000 --blockinfo ${blockinfo}

        logdir="$casename/BACK"
        blockinfo="$output_dir/$para_lane_scene/$para_lane_track_list/BACK"
        sensorid=1
        python3 render.py -s ${data} -m ${output_dir}/${logdir} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --iteration 4000 --blockinfo ${blockinfo}

        logdir="$casename/LEFT"
        blockinfo="$output_dir/$para_lane_scene/$para_lane_track_list/LEFT"
        sensorid=3
        python3 render.py -s ${data} -m ${output_dir}/${logdir} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --iteration 4000 --blockinfo ${blockinfo}

        logdir="$casename/RIGHT"
        blockinfo="$output_dir/$para_lane_scene/$para_lane_track_list/RIGHT"
        sensorid=4
        python3 render.py -s ${data} -m ${output_dir}/${logdir} --para_lane_scene ${para_lane_scene} --para_lane_track_list ${para_lane_track_list} --caseid ${caseid} --sensorid ${sensorid} --iteration 4000 --blockinfo ${blockinfo}
    done
done
# caseid="pesudo"
# python render.py \
# -s /mnt_gx/lidar_data/datanext/union_cases_93769_default \
# -m /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/case_93769/output_1223/union_cases_93769_default/GT2V1-00006_20240814100331_20240814100431_129328959/TOP \
# --iteration 4000 \
# --caseid ${caseid} \
# --sensorid 0
# --insert_dynamic_obj
# --insert_static_obj
# --newcar_render GT2V1