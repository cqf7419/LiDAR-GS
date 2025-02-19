
iterations=4000
time=$(date "+%Y-%m-%d_%H:%M:%S")
data_path="data/waymo/waymo_seq1067"  # data/waymo_dynamic/1005081002024129653_5313_150_5333_150
logdir='waymo_seq1067' #  1005081002024129653_5313_150_5333_150
label_id="waymo"
gpu=0

python train.py -s ${data_path} --data_label ${label_id} --gpu ${gpu} --iterations ${iterations} -m output/${logdir}/$time

