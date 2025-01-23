import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import os

from lfg.ros import LFG, DTYPE, LFG
from lfg.derive import predict

# DTYPE = np.float64
court_lines = {
        "baseline": [[0, -5.48, 0], [0, 5.48, 0]],             # Bottom baseline
        "service_line": [[5.48, -4.11, 0], [5.48, 4.11, 0]],     # Bottom service line
        "lside_line": [[0, -5.48, 0], [11.88, -5.48, 0]],         # Left side line
        "rside_line": [[0, 5.48, 0], [11.88, 5.48, 0]],           # Right side line
        "lserve_line_inner": [[0, -4.11, 0], [11.88, -4.11, 0]],   # Left service line inner
        "rserve_line_inner": [[0, 4.11, 0], [11.88, 4.11, 0]],     # Right service line inner
        "center_service_line": [[5.48, 0, 0], [11.8, 0, 0]],   # Left service line outer
    }

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

    return flattend_detections
    # for det in flattend_detections:
    #     yield det

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

    points = []
    trias = []
    initial_points = []
    flatten_detections = detection_loader(detection_file, traj_idx=0)
    for det in flatten_detections:
        start_time = time.time()
        tria = lfg.compute_position_prior(det)
        res = lfg.update(det)
        print(f'update time: {time.time()-start_time:.3f}s')
        
        if res is not None:
            start_time = time.time()
            res_init = lfg.get_initial_estimate()
            initial_points.append(np.array(res_init).flatten())
            points.append(np.array(res).flatten())
            trias.append(tria)
            print(f'data saving time: {time.time()-start_time:.3f}s')

            print(f'got {len(points)} points')

        if len(points) > 2000:
            break
    points = np.array(points)
    trias = np.array(trias)
    initial_points = np.array(initial_points)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    skip = 3
    for i in range(points.shape[0]):
        
        if i % skip != 0:
            continue

        # predicted_points = predict(initial_points[i, 0:3], initial_points[i, 3:6], initial_points[i, 6:9], 3.0, 300)
        predicted_points = predict(points[i, 0:3], points[i, 3:6], points[i, 6:9], 3.0, 300)

        ax.clear()
        ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], color='red') # current position
        # ax.plot(points[:,0], points[:,1], points[:,2]) # gtsam smoothed
        ax.plot(trias[:,0], trias[:,1], trias[:,2]) # triangulated only

        
        # axes equal
        extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        # centers = np.mean(extents, axis=1)
        # max_range = np.ptp(extents, axis=1).max() / 2
        # for ctr, axis in zip(centers, 'xyz'):
        #     getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)
        zoom_in = 0.5
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0*zoom_in, 30*zoom_in)
        ax.set_ylim(-15*zoom_in, 15*zoom_in)
        ax.set_zlim(-10*zoom_in, 15*zoom_in)
        ax.view_init(elev=20., azim=90)

        for name, kpts_court in court_lines.items():
            ax.plot(*np.array(kpts_court).T, color='k')
            ax.plot(*np.array(np.array([23.77, 0, 0]) - kpts_court).T, color='k')

        save_filename = f'plots/increm_gtsam/{i:04d}.png'
        if not os.path.exists(save_filename):
            os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)    
        print(save_filename)

    plt.show()