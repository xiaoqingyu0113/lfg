#!/usr/bin/env python3
from lfg.ros import LFG, param2proj

from lfg.derive_mnnl import predict
import rospy
from std_srvs.srv import Empty, EmptyResponse
from ball_detection_new.msg import Detections
from ball_detection_new.msg import ImagePoint
from geometry_msgs.msg import PointStamped, Point,  PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header, Bool
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

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
        if len(callback_queue) > 50:
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
                            'N': 200} # configure the time period and resolution of the prediction
        
        self.path_publisher = rospy.Publisher('/ball/rollout/path', Path, queue_size=1)
        self.ball_publisher = rospy.Publisher('/ball/rollout/pos', PoseStamped, queue_size=1)
        self.courtline_publisher = rospy.Publisher('/tennis_court_markers', Marker, queue_size=1)
        self.bounce_publisher = rospy.Publisher('/ball/rollout/is_bounce', Bool, queue_size=1)

        self.state_history = [] # List[Tuple(p,v,w)], save all latest estimation states at current
        self.bounce_idx = [] # save the indices of bounce happend in self.state_history 
        self.bc_muted_period = 30 # number of obs suppressed before publishing the estimation

        self.robot_location = None
        self.prev_obs_time = None

        rospy.loginfo("LFG node ready!")
    
    def reset(self):
        self.inference = None # set by LFG inference result
        self.curr_header = None # set by obs

        self.state_history = [] # List[Tuple(p,v,w)], save all latest estimation states at current
        self.bounce_idx = [] # save the indices of bounce happend in self.state_history 
        self.lfg.reset()
        self.robot_odom = None

        rospy.loginfo("LFG node reset!")


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

        if  self.prev_obs_time is not None and t - self.prev_obs_time > 2.0:
            self.prev_obs_time = t
            rospy.loginfo('[RESET] large time gap between observations')
            self.reset()
            return
        self.prev_obs_time = t
        start_time = rospy.Time.now()

        if len(data.points) > 0:
            closet_detection = (t, cam_id,data.points[0].x,data.points[0].y)  
        else:
            return
        
        min_error = 10000

        # filter out points around the robot
        filtered_points = data.points
        if self.robot_location is not None:
            filtered_points = []
            for d in data.points:
                det = (t, cam_id, d.x, d.y)

                curr_camparam = self.lfg.cam_params_dict[cam_id]
                repro_uv = param2proj(curr_camparam) @ np.concatenate((self.robot_location + np.array([0,0,0.6]), [1])) # add a bit height to the robot
                repro_uv = repro_uv[:2] / repro_uv[2]

                if np.linalg.norm(repro_uv - np.array([d.x, d.y])) < 120.0:
                    rospy.loginfo(f'[FILTER] Detected Lidar light at in camera {cam_id} at {d.x, d.y}, reprojection error = {np.linalg.norm(repro_uv - np.array([d.x, d.y]))}')
                    continue
                else:
                    filtered_points.append(d)

        # find the closest detection to the current detection (filter out potential human noise)
        for d in filtered_points:
            det = (t, cam_id, d.x, d.y)
            l_prior, repr_error = self.lfg.compute_position_prior(det)
            if l_prior is not None and min_error > repr_error:
                closet_detection = d
                min_error = repr_error

        # if all way off, choose the last one only for the purpose of updating time and uv_prev in lfg 
        if closet_detection is not None:
            closet_detection = det
   


        # if unstable solution, unpredictable bounce might occurs, reset the graph
        # self.inference = self.lfg.update(closet_detection)
        try:
            self.inference = self.lfg.update(closet_detection)
        except:
            rospy.loginfo('[RESET] Unstable solution, unpredictable bounce might occurs')
            self.reset()
            return
        
        # if velocity in xy plane angle, it is a bounce
        last_k_vels = self.lfg.get_last_k_velocity(20)

        if last_k_vels is not None:
            # xy angle change  
            vx = last_k_vels[:, 0]
            vy = last_k_vels[:, 1]
        
            angles = np.arctan2(vy, vx)
            delta_angles = np.diff(angles)
            delta_angles = (delta_angles + np.pi) % (2 * np.pi) - np.pi # [-pi, pi]
            if np.abs(delta_angles).max() > np.pi/2:
                rospy.loginfo(f'[RESET] large velocity change in xy-plane. Angle = {np.abs(delta_angles).max()/np.pi*180:.1f} > 90 degree')
                self.reset()
                return 


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
            self.bounce_publisher.publish(Bool(True)) if len(self.bounce_idx)>0 else self.bounce_publisher.publish(Bool(False))
            self.publish_court_markers()
                

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

    def callback_odom(self, data):
        
        self.robot_location = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        # print(f'Robot location: {self.robot_location}')
        

    def publish_court_markers(self):
        '''
        use visualization_msgs/Marker to publish the court lines and net

        subscribe to '/tennis_court_markers' in frame 'world'
        '''
        def add_line(marker, x1, y1, z1, x2, y2, z2):
            """Helper to add a line segment between two 3D points to the marker."""
            p1 = Point()
            p1.x, p1.y, p1.z = x1, y1, z1
            p2 = Point()
            p2.x, p2.y, p2.z = x2, y2, z2
            
            # Each pair of points in a LINE_LIST marker is one line segment
            marker.points.append(p1)
            marker.points.append(p2)

        def create_net_marker():
            court_length = 23.77
            net_height = 0.914 
            net_x = court_length / 2
            court_width_doubles = 10.97

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "tennis_court"
            marker.id = 100  # Unique ID for net
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = net_x
            marker.pose.position.y = 0
            marker.pose.position.z = net_height / 2  # Center the cube vertically
            marker.scale.x = 0.05           # Thickness of the net
            marker.scale.y = court_width_doubles  # Full width of doubles court
            marker.scale.z = net_height     # Height of the net
            marker.color.a = 0.5            # Semi-transparent
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            return marker
        
        # Create and configure the Marker
        court = Marker()
        court.header.frame_id = "world"       # change as needed (map, odom, etc.)
        court.header.stamp = rospy.Time.now()
        court.ns = "tennis_court"
        court.id = 0
        court.type = Marker.LINE_LIST
        court.action = Marker.ADD
        
        # Set line thickness
        court.scale.x = 0.05  # meters
        
        # White color
        court.color.r = 1.0
        court.color.g = 1.0
        court.color.b = 1.0
        court.color.a = 1.0  # fully opaque
        
        z = 0.0  # all lines on the ground plane
        
        # Court dimensions (meters):
        #   - Baseline to baseline: 23.77
        #   - Net at x=11.885
        #   - Doubles width: 10.97 (y from -5.485 to +5.485)
        #   - Singles width: 8.23  (y from -4.115 to +4.115)
        #   - Service line: 6.40 from net => near side at x=5.485, far side at x=18.285
        #   - Center service line: y=0 between each service line and the net
        #   - Center mark on near baseline: short segment at x=0 from y=-0.05 to +0.05
        
        # 1) Near Baseline (doubles)
        add_line(court, 0.0, -5.485, z, 0.0,  5.485, z)
        
        # 2) Far Baseline (doubles)
        add_line(court, 23.77, -5.485, z, 23.77,  5.485, z)
        
        # 3) Left doubles sideline
        add_line(court, 0.0, -5.485, z, 23.77, -5.485, z)
        
        # 4) Right doubles sideline
        add_line(court, 0.0,  5.485, z, 23.77,  5.485, z)
        
        # 5) Left singles sideline
        add_line(court, 0.0, -4.115, z, 23.77, -4.115, z)
        
        # 6) Right singles sideline
        add_line(court, 0.0,  4.115, z, 23.77,  4.115, z)
        
        # 7) Net (x=11.885)
        add_line(court, 11.885, -5.485, z, 11.885, 5.485, z)
        
        # 8) Near service line (x=5.485)
        add_line(court, 5.485, -4.115, z, 5.485,  4.115, z)
        
        # 9) Far service line (x=18.285)
        add_line(court, 18.285, -4.115, z, 18.285,  4.115, z)
        
        # 10) Center service line (near side)
        add_line(court, 5.485, 0.0, z, 11.885, 0.0, z)
        
        # 11) Center service line (far side)
        add_line(court, 11.885, 0.0, z, 18.285, 0.0, z)
        
        # 12) Center mark on the near baseline
        add_line(court, 0.0, -0.05, z, 0.0, 0.05, z)
        
        # Continuously publish the marker
        court.header.stamp = rospy.Time.now()
        self.courtline_publisher.publish(court)
        self.courtline_publisher.publish(create_net_marker())


def listener():
    rospy.init_node('LFG_Node', anonymous=True)

    lfg_node = LFG_Node(verbose=True)

    rospy.Subscriber("/camera_1/detector_1/detections", Detections, lambda data: enqueue_callback(lfg_node.callback1,data),queue_size=2)
    rospy.Subscriber("/camera_2/detector_2/detections", Detections, lambda data: enqueue_callback(lfg_node.callback2,data),queue_size=2)
    rospy.Subscriber("/camera_3/detector_3/detections", Detections, lambda data: enqueue_callback(lfg_node.callback3,data),queue_size=2)
    # rospy.Subscriber("/camera_4/detector_4/detections", Detections, lambda data: enqueue_callback(lfg_node.callback4,data),queue_size=2)
    # rospy.Subscriber("/camera_5/detector_5/detections", Detections, lambda data: enqueue_callback(lfg_node.callback5,data),queue_size=2)
    # rospy.Subscriber("/camera_6/detector_6/detections", Detections, lambda data: enqueue_callback(lfg_node.callback6,data),queue_size=2)
    rospy.Subscriber("/wcodometry_global", Odometry, lambda data: enqueue_callback(lfg_node.callback_odom,data),queue_size=2)

    process_callback_queue()


if __name__ == '__main__':
    listener()

