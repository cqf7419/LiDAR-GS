iterations=4000
output_dir="./output1"
gpu=0

casename="waymo"
data="/home/xuanyuan/lansheng/waymo_train" # Replace it with the path where you unzipped the file

caseid="segment-1005081002024129653_5313_150_5333_150_with_camera_labels"
logdir="$casename/$caseid"
sensorid=0
python3 render.py -s ${data} -m ${output_dir}/${logdir} --caseid ${caseid} --iteration ${iterations} --max_depth 80 --blockinfo ${output_dir}/${logdir}


