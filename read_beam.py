import tensorflow as tf
from waymo_open_dataset import dataset_pb2
import os
import numpy as np

selected_segment = "/mnt_gx/usr/lansheng/selected_waymo.txt"
with open(selected_segment, 'r') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]

root_path = "/mnt_gx/lidar_data/datapublic_old/waymo_train/temp"
for segment in os.listdir(root_path):
    if segment not in lines: continue
    beam_path = os.path.join(root_path, segment, "beam_inclinations")
    if os.path.exists(beam_path) == True:
        continue

    occ_path = os.path.join(root_path, segment, "occ")
    if os.path.exists(occ_path) == False:
        continue
        #os.rmdir(os.path.join(root_path, segment))
        
    pathname = os.path.join("/data1/public_dataset/waymo/waymo_v120/tfrecord_training", segment+".tfrecord")
    if os.path.exists(pathname):
        print(pathname)
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        data0= None
        for frame_idx, data in enumerate(dataset):
            if frame_idx == 0: data0 = data

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data0.numpy()))
        calibrations = sorted(frame.context.laser_calibrations,key=lambda c: c.name)
        beam_inclinations=None
        for c in calibrations:
            if c.name != dataset_pb2.LaserName.TOP: continue
            beam_inclinations = tf.constant(c.beam_inclinations)
            beam_inclinations = list(beam_inclinations.numpy())
            beam_inclinations = [f'{i:e}' for i in beam_inclinations]

        save_path = "/mnt_gx/lidar_data/datapublic_old/waymo_train/temp/"+segment
        save_path = os.path.join(save_path, "beam_inclinations")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "beam_inclinations.npy"), beam_inclinations)