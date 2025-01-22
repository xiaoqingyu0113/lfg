import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import os

from lfg.ros import LFG, DTYPE
from lfg.derive import predict

from train import RealTrajectoryDataset

import torch
from torch.utils.data import DataLoader


'''

This test if GTSAM still works for the data in the interpolated wellprepared trajectory dataset used for training
'''

DTYPE = np.float64



def detection_loader(detection_file, traj_idx = 0):
    '''
    mimicks ros subscriber
    '''
    with open(detection_file, 'r') as f:
        detections = json.load(f)

    flattend_detections = []
    for camera_id, points in detections.items():
        for p in points:
                if p[0] != traj_idx:
                    continue
                p[3] = camera_id
                flattend_detections.append(p)

    flattend_detections.sort(key=lambda x: x[2]) # sort by timestamp

    for det in flattend_detections:
        yield det

def detection_parser(data):
    traj_idx, data_idx, timestamp, camera_id, u, v = data
    return timestamp, camera_id, u, v


def read_cam_calibration(filename):
    with open(filename,'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    K = np.array(data['camera_matrix']['data']).reshape(3,3)
    R = np.array(data['R_cam_world']).reshape(3,3)
    t = np.array(data['t_world_cam'])
    return K, R, -R@t


import time

if __name__ == '__main__':        
    detection_file = glob.glob('data/real/detections_tennis/data1*.json')[0]
    print(f'using {detection_file}')
    lfg = LFG(cam_params_dict=None, det_parser=detection_parser, min_graph_size=30)

    # load training data
    dataset = RealTrajectoryDataset('data/real/tennis_kpts',interpolate=500)
    print(f"Loaded dataset with {len(dataset)} samples.")
    total_data_size = len(dataset)
    split_ratio = 0.8
    train_data_size = int(total_data_size * split_ratio)
    test_data_size = total_data_size - train_data_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    single_traj = next(iter(test_loader))
    single_traj = single_traj[0, :, :].cpu().numpy()

    points = []
    trias = []
    initial_points = []
    for tria in single_traj:
        start_time = time.time()

        res = lfg.update(tria)
        print(f'update time: {time.time()-start_time:.3f}s')
        
        if res is not None:
            start_time = time.time()
            res_init = lfg.get_initial_estimate()
            initial_points.append(np.array(res_init).flatten())
            points.append(np.array(res).flatten())
            trias.append(tria)
            print(f'data saving time: {time.time()-start_time:.3f}s')

            print(f'got {len(points)} points')

        if len(points) > 1200:
            break

    points = np.array(points)
    trias = np.array(trias)
    initial_points = np.array(initial_points)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(points.shape[0]):
   

        predicted_points = predict(points[i, 0:3], points[i, 3:6], points[i, 6:9], 3.0, 300)

        ax.clear()
        ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], color='red') # current position
        ax.plot(points[:,0], points[:,1], points[:,2]) # gtsam smoothed
        ax.plot(trias[:,0], trias[:,1], trias[:,2]) # triangulated only

        
        # axes equal
        extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        # centers = np.mean(extents, axis=1)
        # max_range = np.ptp(extents, axis=1).max() / 2
        # for ctr, axis in zip(centers, 'xyz'):
        #     getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-10, 20)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-10, 15)

        save_filename = f'plots/increm_gtsam_tria/{i:04d}.png'
        if not os.path.exists(save_filename):
            os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)    
        print(save_filename)

    plt.show()