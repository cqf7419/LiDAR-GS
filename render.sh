# casename="onemodel_cases_93366_default"
# caseid="GT2V5-00022_20240704154925_20240704155055_128884762"
casename="union_cases_93769_default"
caseid="GT2V1-00006_20240814100331_20240814100431_129328959"
sensorid=0

python render.py \
-s /mnt_gx/lidar_data/datanext/union_cases_93769_default \
-m /mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash/case_93769/output_1223/union_cases_93769_default/GT2V1-00006_20240814100331_20240814100431_129328959/TOP \
--iteration 4000 \
--caseid ${caseid} \
--sensorid ${sensorid} \
--insert_dynamic_obj
# --insert_static_obj
# --newcar_render GT2V1