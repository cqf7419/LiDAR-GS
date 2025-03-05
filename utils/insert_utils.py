import pickle as pkl
import os 
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import collections
import math
import random
import pdb

def kalman_filter(observations, time_intervels, process_noise=0.1, measurement_noise=0.1, heuristic_init=True, vis=False):

    # 观测矩阵
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])

    # 过程噪声协方差矩阵
    Q = np.eye(6) * process_noise
    # 观测噪声协方差矩阵
    R = np.eye(2) * measurement_noise

    # 初始状态
    if len(observations) > 1 and heuristic_init and observations[0] is not None and observations[1] is not None:
        init_v = (observations[1] - observations[0]) / time_intervels[1]
        vx, vy = init_v[0, 0], init_v[1, 0]
        # 加速度初始值抖动过于剧烈，会导致误差发散
        # if len(observations) > 2:
        #     next_v = (observations[2] - observations[1]) / time_intervels[2]
        #     init_av = (next_v -init_v) / time_intervels[1]
        #     ax, ay = init_av[0, 0], init_av[1, 0]
        # else:
        #     ax, ay = 0, 0
    else:
        vx, vy = 0, 0
    
    ax, ay = 0, 0
    x = np.array([float(observations[0][0]), float(observations[0][1]), vx, vy, ax, ay]).reshape(6,1)
    P = np.eye(6)   # 初始状态协方差

    predicted_positions = []
    predicted_velocities = []

    predicted_positions.append(x[:2])
    predicted_velocities.append(x[2:4])
    # vis compare
    trans_velocities = []
    # time_intervels = time_intervels[1:] + [0.1]
    trans_velocities.append(x[2:4])
    error_x = []
    for idx , observation in enumerate(observations):

        if idx == 0: continue
        dt = time_intervels[idx]
        # 状态转移矩阵
        A = np.array([[1, 0, dt, 0, 0, 0],
                    [0, 1, 0, dt, 0, 0],
                    [0, 0, 1, 0, dt, 0],
                    [0, 0, 0, 1, 0 ,dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

        # 预测步骤
        x = A @ x  # 预测状态
        P = A @ P @ A.T + Q  # 预测协方差
        # 更新步骤
        if observation is not None:
            y = observation - H @ x  # 观测残差
            S = H @ P @ H.T + R  # 残差协方差
            K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
            x = x + K @ y  # 更新状态
            P = P - K @ H @ P # 更新协方差

        # 存储预测值
        predicted_positions.append(x[:2])
        predicted_velocities.append(x[2:4])
        error_x.append(np.trace(P))
        if idx != len(observations) - 1:
            diffx = observations[idx+1][0] - observations[idx][0]
            diffy = observations[idx+1][1] - observations[idx][1]
            trans_velocities.append([diffx/time_intervels[idx+1], diffy/time_intervels[idx+1]])
    if vis:
        import matplotlib.pyplot as plt
        predicted_velocities_x = np.array(predicted_velocities)[:,0]
        predicted_velocities_y = np.array(predicted_velocities)[:,1]
        pred_vel = np.sqrt(np.power(predicted_velocities_x, 2) + np.power(predicted_velocities_y, 2))
        trans_velocities_x = np.array(trans_velocities)[:,0]
        trans_velocities_y = np.array(trans_velocities)[:,1]
        trans_vel = np.sqrt(np.power(trans_velocities_x, 2) + np.power(trans_velocities_y, 2))
        plt.figure()
        plt.plot(range(len(predicted_velocities_x)), predicted_velocities_x, c='r', label='pred_vx')
        plt.plot(range(len(trans_velocities_x)), trans_velocities_x, c='b', label='trans_vx')

        plt.plot(range(len(predicted_velocities_y)), predicted_velocities_y, c='y',label="pred_vy")
        plt.plot(range(len(trans_velocities_y)), trans_velocities_y, c='g',label="trans_vy")
        plt.legend()
        plt.show()
        plt.savefig('./tmp_vis_kmfilter_velocity.png')
    return np.array(predicted_positions).squeeze(-1), np.array(predicted_velocities).squeeze(-1)


def generate_vel_gt(pkl_data, 
                    seq='GT2V1-00001_20240808081448_20240808081513_130322875',
                    gt_boxe_token = '9999999'):
    '''
    用于植入动态物体时, 计算bbox的速度
    '''
    # with open(file_path , 'rb') as f:
    data = pkl_data #pkl.load(f)
    instances = {}
    for frame_id, frame_info in enumerate(data[seq]['seq_info']):
        idx = frame_info['gt_boxes_token'].tolist().index(gt_boxe_token)
        if gt_boxe_token not in instances:
            instances[gt_boxe_token] = []
        instances[gt_boxe_token].append({'pose': frame_info['pose'], 
                                        'timestampe': frame_info['timestamp'],
                                        'bbox': frame_info['gt_boxes'][idx],
                                        'frame_id': frame_id})

    for token_name, instance in instances.items():
        pre_time = None
        centers = []
        time_intervels = []
        for ins in instance:
            
            if isinstance(ins['timestampe'], float):
                cur_time = ins['timestampe']
            else:
                cur_time = ins['timestampe'][0]
            if pre_time is None:
                pre_time = cur_time
                
            time_intervels.append(cur_time-pre_time)
            pose = ins['pose']
            center = ins['bbox'][:3]
            cur_center = (pose[:3,:3] @ center.reshape(1,3).T).T + pose[:3,3]
            cur_center = cur_center[0]
            centers.append(cur_center[:2].reshape(2, 1))
            pre_time = cur_time
        pred_center, pred_vel = kalman_filter(centers, time_intervels)
        # print(pred_vel) # [50,2]

        # to wirte vel info
        for frame_id in range(len(data[seq]['seq_info'])):
            gt_boxes_velocity_datas = data[seq]["seq_info"][frame_id]["gt_boxes_velocity"]
            new_gt_boxes_velocity_datas = np.concatenate((gt_boxes_velocity_datas, np.array([[pred_vel[frame_id][0],pred_vel[frame_id][1]]],dtype=np.float32)), axis=0)
            if frame_id==0: print(new_gt_boxes_velocity_datas.shape)
            data[seq]["seq_info"][frame_id]["gt_boxes_velocity"] = new_gt_boxes_velocity_datas

        return data