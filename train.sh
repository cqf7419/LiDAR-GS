
iterations=4000
time=$(date "+%Y-%m-%d_%H:%M:%S")
data_path="data/waymo"
logdir='waymo_seq1067'
label_id="waymo"
gpu=0

export TORCH_USE_CUDA_DSA=1 
python train.py -s ${data_path} --data_label ${label_id} --gpu ${gpu} --iterations ${iterations} -m output/${logdir}/$time

