import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml


from lfg.ros import LFG, DTYPE
from lfg.derive import predict

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




if __name__ == '__main__':        
    detection_file = glob.glob('data/real/detections_tennis/data1*.json')[0]

    lfg = LFG(cam_params_dict=None, det_parser=detection_parser, min_graph_size=30)

    points = []
    trias = []
    for det in detection_loader(detection_file, traj_idx=0):
        tria = lfg.compute_position_prior(det)
        res = lfg.update(det)
        
        if res is not None:
            points.append(np.array(res).flatten())
            trias.append(tria)


    points = np.array(points)
    trias = np.array(trias)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(points.shape[0]):
        p = points[i, 0:3]
        v = points[i, 3:6]
        w = points[i, 6:9]

        predicted_points = predict(p, v, w, 2.0, 200)

        ax.clear()
        ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory
        ax.scatter(p[0], p[1], p[2], color='red') # current position
        ax.plot(points[:,0], points[:,1], points[:,2]) # gtsam smoothed
        ax.plot(trias[:,0], trias[:,1], trias[:,2]) # triangulated only

        
        # axes equal
        extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        centers = np.mean(extents, axis=1)
        max_range = np.ptp(extents, axis=1).max() / 2
        for ctr, axis in zip(centers, 'xyz'):
            getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)

        plt.savefig(f'plots/LFG/{i:04d}.png')        
    plt.show()