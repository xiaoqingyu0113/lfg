#!/usr/bin/env python3
from lfg.ros import LFG
from lfg.derive import predict
import rospy
from std_srvs.srv import Empty, EmptyResponse
from ball_detection_new.msg import Detections
from ball_detection_new.msg import ImagePoint
from geometry_msgs.msg import PointStamped, Point,  PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler


import numpy as np
from typing import List, Optional
import os
import json
from collections import deque
from threading import Lock
import sys
sys.path.append(os.path.dirname(__file__))


'''
--------------------------------- handling multiple callbacks sequentialy -------------------------------------
'''

callback_queue = deque()
callback_lock = Lock()

def enqueue_callback(callback, data):
    with callback_lock:
        if len(callback_queue) > 3:
            callback_queue.popleft()
        callback_queue.append((callback, data))

def process_callback_queue():
    while not rospy.is_shutdown():
        with callback_lock:
            if callback_queue:
                callback, data = callback_queue.popleft()
                callback(data)

def detection_data_parser(data):
    '''
    return t, camera_id, u, v
    '''
    return data

class LFG_Node:
    def __init__(self, verbose=False):

        self.lfg = LFG(cam_params_dict=None, det_parser=detection_data_parser, min_graph_size=10)
        self.inference = None # set by LFG inference result
        self.curr_header = None # set by obs
        self.verbose = verbose

        self.pred_params = {'duration': 3.0,
                            'N': 100} # configure the time period and resolution of the prediction
        
        self.path_publisher = rospy.Publisher('/ball/rollout/path', Path, queue_size=1)
        self.ball_publisher = rospy.Publisher('/ball/rollout/pos', PoseStamped, queue_size=1)

        self.state_history = [] # List[Tuple(p,v,w)], save all latest estimation states at current
        self.bounce_idx = [] # save the indices of bounce happend in self.state_history 
        self.bc_muted_period = 30 # number of obs suppressed before publishing the estimation

        rospy.loginfo("LFG node ready!")

    def is_bounce(self):
        if len(self.state_history)< 2:
            return False
        
        p2, v2, w2 = self.state_history[-1]
        p1, v1, w1 = self.state_history[-2]

        if v2[2] > 0.0 and v1[2] < 0.0:
            curr_idx = len(self.state_history)-1
            if len(self.bounce_idx) > 0 and curr_idx - self.bounce_idx[-1] < 3: # two bounces should not occur within 5 frames, if it is, should be noise 
                # self.bounce_idx[-1] = curr_idx
                return False
            else:
                return True
        else:
            return False
        
    def allow_publish(self):
        # inference is not ready
        if self.inference is None:
            return False
        # don't publish result right after a small period of bounce. wait for more obs to stablize the est.
        if len(self.bounce_idx) > 0 and (len(self.state_history) -1) - self.bounce_idx[-1] < self.bc_muted_period:
            # if self.verbose:
            #     print(f'At {len(self.state_history)}. Muted by bounce because it is close to the last bounce {self.bounce_idx[-1]}')
            return False

        return True
        

    
    def add_to_graph(self, data, cam_id):
        t = data.header.stamp.to_sec()

        start_time = rospy.Time.now()
        for d in data.points:
            # this will only accept the first points, the second points will be filtered out by lfg objects internally.
            self.inference = self.lfg.update((t, cam_id, d.x, d.y))
            self.curr_header = data.header
            if self.inference is not None:
                self.state_history.append(self.inference) 
            if self.is_bounce():
                self.bounce_idx.append(len(self.state_history)-1)
                if self.verbose:
                    print(f'At {len(self.state_history)} Detected the {len(self.bounce_idx)}th bounce')
        _INFERENCE_TIME = (rospy.Time.now() - start_time).to_sec()
        if self.verbose and _INFERENCE_TIME > 0.010:
            print(f'\t - INFERENCE: At {len(self.state_history)}. Inference takes {_INFERENCE_TIME} seconds')

        if self.allow_publish():
            start_time = rospy.Time.now()
            self.publish_prediction()
            _PUBLISH_TIME = (rospy.Time.now() - start_time).to_sec()
            if self.verbose and _PUBLISH_TIME > 0.010:
                print(f'\t - PUBLISH: At {len(self.state_history)}. Publish takes {_PUBLISH_TIME} seconds')

    def publish_prediction(self):
        duration = self.pred_params['duration']
        N = self.pred_params['N']
        points = predict(*self.inference,duration, N) # future 2.0 secs, with 200 poitns (equally distributed)
        times = np.linspace(0, duration, N)

        poses_stamped = []
        for i in range(N):
            p = points[i,:]
            t = times[i]

            if 0 <= p[0] <= 28 and -4 <= p[1] <= 4:
                stamp = self.curr_header.stamp + rospy.Duration(t)
                header = Header(self.curr_header.seq, stamp, 'world')
                point = Point(p[0], p[1], p[2])
                pose = Pose(point, Quaternion(0,0,0,1))
                pose_stamped = PoseStamped(header, pose)
                poses_stamped.append(pose_stamped)

        if len(poses_stamped) > 0:
            header = self.curr_header
            header.frame_id = 'world'
            path = Path(header, poses_stamped)
            self.path_publisher.publish(path)
            self.ball_publisher.publish(poses_stamped[0])
                

    def callback1(self, data):
        self.add_to_graph(data, 'camera_1')
    def callback2(self, data):
        self.add_to_graph(data, 'camera_2')
    def callback3(self, data):
        self.add_to_graph(data, 'camera_3')
    def callback4(self, data):
        self.add_to_graph(data, 'camera_4')
    def callback5(self, data):
        self.add_to_graph(data, 'camera_5')
    def callback6(self, data):
        self.add_to_graph(data, 'camera_6')


    
def listener():
    rospy.init_node('LFG_Node', anonymous=True)

    lfg_node = LFG_Node(verbose=True)

    rospy.Subscriber("/camera_1/detector_1/detections", Detections, lambda data: enqueue_callback(lfg_node.callback1,data),queue_size=2)
    rospy.Subscriber("/camera_2/detector_2/detections", Detections, lambda data: enqueue_callback(lfg_node.callback2,data),queue_size=2)
    rospy.Subscriber("/camera_3/detector_3/detections", Detections, lambda data: enqueue_callback(lfg_node.callback3,data),queue_size=2)
    rospy.Subscriber("/camera_4/detector_4/detections", Detections, lambda data: enqueue_callback(lfg_node.callback4,data),queue_size=2)
    rospy.Subscriber("/camera_5/detector_5/detections", Detections, lambda data: enqueue_callback(lfg_node.callback5,data),queue_size=2)
    rospy.Subscriber("/camera_6/detector_6/detections", Detections, lambda data: enqueue_callback(lfg_node.callback6,data),queue_size=2)

    process_callback_queue()


if __name__ == '__main__':
    listener()

