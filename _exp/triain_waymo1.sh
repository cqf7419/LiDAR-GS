
# pip install submodules/diff_lidargs_surfel_rasterization
path="/home/xuanyuan/test2/LiDAR_GS(Waymo)"
cd $path
iterations=4000
output_dir="./outputs"
gpu=0

casename="waymo"
data="/home/xuanyuan/lansheng/waymo_train"

caseid="segment-1005081002024129653_5313_150_5333_150_with_camera_labels"
logdir="$casename/$caseid"
sensorid=0
python3 train.py -s ${data} --caseid ${caseid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir}/${logdir} --max_depth 80


