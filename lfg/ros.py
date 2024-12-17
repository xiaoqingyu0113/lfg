import numpy as np

import gtsam
from gtsam.symbol_shorthand import X, L, V, W
from lfg.derive import PriorFactor3, PositionFactor, VWFactor, predict
import cv2
import yaml
import os
CURRENT_DIR = os.path.dirname(__file__)  # Directory of the current script


DTYPE = np.float64

def param2proj(camera_param):
    'R is 3x3 and t is 3'
    K, R, t = camera_param
    return K@ np.hstack((R, t.reshape(3,1)))

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

class LFG:
    def __init__(self, cam_params_dict=None,
                  det_parser = detection_parser, 
                  min_graph_size = 30):
        self.det_parser = det_parser

        # for camera
        if cam_params_dict is None:
            config_folder = os.path.join(CURRENT_DIR, '../conf/camera')
            camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
            date = 'Dec13'
            cam_params = [read_cam_calibration(f'{config_folder}/{cname}_calibration_{date}.yaml') for cname in camera_names]
            cam_params_dict =  {'camera_'+str(i+1):cam_params[i] for i in range(6)}

        self.cam_params_dict = cam_params_dict

        # for triangulation
        self.prev_time =None
        self.prev_uv = None
        self.prev_camera_id = None
        self.prev_l_prior = None
        self.prev_interp_time = None
        self.prev_interp_l_prior = None


        # for issam
        self.gid = 0
        self.graph = gtsam.NonlinearFactorGraph()
        self.isam2 = gtsam.ISAM2(gtsam.ISAM2Params())
        self.initial_estimate = gtsam.Values()
        self.optim_estimate = None
        self.min_graph_size = min_graph_size
        self.pPriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.010, 0.010, 0.010], dtype=DTYPE))
        self.vwNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1], dtype=DTYPE))
        self.wPriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1], dtype=DTYPE))




    def compute_position_prior(self, det):
        timestamp, camera_id, u, v = self.det_parser(det)
        points_3d = None
        if self.prev_time is not None \
            and self.prev_camera_id != camera_id \
            and self.prev_camera_id != 'camera_1' \
            and camera_id != 'camera_1' \
            and self.prev_camera_id != 'camera_6' \
            and camera_id != 'camera_6' \
            and timestamp - self.prev_time < 0.010:

            prev_camparam = self.cam_params_dict[self.prev_camera_id]
            curr_camparam = self.cam_params_dict[camera_id]

            points4d = cv2.triangulatePoints(param2proj(prev_camparam), param2proj(curr_camparam), self.prev_uv, np.array([u, v]))
            points_3d = points4d / points4d[3]
            points_3d = points_3d[:3].flatten()

            repro_uv = param2proj(curr_camparam) @ np.concatenate((points_3d, [1]))
            repro_uv = repro_uv[:2] / repro_uv[2]

            repr_error = np.linalg.norm(repro_uv - np.array([u,v]))

            if repr_error > 120:
                points_3d = None

        return points_3d
    
    def update(self, det):
        t, camera_id, u, v = self.det_parser(det)

        w_prior = np.array([1,0,0], dtype=DTYPE)
        l_prior = self.compute_position_prior(det)

        if l_prior is None:
            self.prev_time = t
            self.prev_camera_id = camera_id
            self.prev_uv = np.array([u, v])

            return None

        if self.prev_l_prior is None:
            l_interp = l_prior
            t_interp = t
        else:
            l_interp = 0.5*(l_prior + self.prev_l_prior)
            t_interp = 0.5*(t + self.prev_time)
            
        

        self.graph.push_back(PriorFactor3(self.pPriorNoise, L(self.gid), l_interp))
        self.initial_estimate.insert(L(self.gid), l_interp)

        if self.gid == 0:
            self.graph.push_back(PriorFactor3(self.wPriorNoise, W(self.gid), w_prior))
            self.initial_estimate.insert(W(self.gid), w_prior)
            self.initial_estimate.insert(V(self.gid), 1e-3*np.random.rand(3).astype(DTYPE))
        else:
            self.graph.push_back(PositionFactor(self.pPriorNoise, L(self.gid-1), V(self.gid-1), L(self.gid), 0.0, t_interp-self.prev_interp_time))
            self.graph.push_back(VWFactor(self.vwNoise,L(self.gid-1), V(self.gid-1), W(self.gid-1), V(self.gid), W(self.gid), 0.0, t_interp - self.prev_interp_time, 0.200))

            if self.optim_estimate is None:
                self.initial_estimate.insert(V(self.gid), 1e-3*np.random.rand(3).astype(DTYPE))
                self.initial_estimate.insert(W(self.gid), np.array([1.0,0,0],dtype=DTYPE))
            else:
                v_prev = self.optim_estimate.atVector(V(self.gid-1))
                w_prev = self.optim_estimate.atVector(W(self.gid-1))
                self.initial_estimate.insert(V(self.gid), v_prev)
                self.initial_estimate.insert(W(self.gid), w_prev)

        if self.gid > self.min_graph_size:
            self.isam2.update(self.graph, self.initial_estimate)
            self.optim_estimate = self.isam2.calculateEstimate()
            self.graph.resize(0)
            self.initial_estimate.clear()

        
        # move outside
        self.prev_time = t
        self.prev_camera_id = camera_id
        self.prev_uv = np.array([u, v])
        self.prev_l_prior = l_prior
        self.prev_interp_time = t_interp
        self.prev_interp_l_prior = l_interp
        self.gid += 1

        if self.optim_estimate is not None:
            return self.optim_estimate.atVector(L(self.gid-1)), self.optim_estimate.atVector(V(self.gid-1)), self.optim_estimate.atVector(W(self.gid-1))
        else:
            return None
