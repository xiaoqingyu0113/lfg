import numpy as np
import rosbag
import matplotlib.pyplot as plt
from pycamera import triangulate, CameraParam, set_axes_equal
import yaml
import gtsam

def msg_parser(msg):
    t = msg.header.stamp.to_sec()
    points = []
    for p in msg.points:
        points.append([p.x, p.y])
    return t, points

def view_points():
    detections = {f'camera_{i}': [] for i in range(1,7)}
    bag = rosbag.Bag('/home/qingyu/Downloads/20241209_tennis/data1_2024-12-09-17-32-32.bag')

    # read all detections
    for topic, msg, t in bag.read_messages():

        camera_id = topic.split('/')[1]
        for p in msg.points:
            detections[camera_id].append([msg.header.stamp.to_sec(), p.x, p.y])
    bag.close()

    # convert to numpy array for ploting
    for camera_id, points in detections.items():
        detections[camera_id] = np.array(points)

    # plot out the detections
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    for i, (camera_id, points) in enumerate(detections.items()):
        ax = axs[i//2, i%2]
        ax.scatter(points[:,1], points[:,2], s=1)
        ax.set_title(camera_id)
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 1024)
        ax.invert_yaxis()
    plt.show()

def read_cam_calibration(filename):
    with open(filename,'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    K = np.array(data['camera_matrix']['data']).reshape(3,3)
    R = np.array(data['R_cam_world']).reshape(3,3)
    t = np.array(data['t_world_cam'])
    cam_param = CameraParam(K, R, -R@t)

    #  ------------ test,  pose in gtsam should be camera pose in the world frame ------------
    # K_gtsam = gtsam.Cal3_S2(K[0,0], K[1,1], K[2,2], K[0,2], K[1,2])
    # R_gtsam = gtsam.Rot3(R.T) 
    # t_gtsam = gtsam.Point3(t[0],t[1],t[2])
    # pose1 = gtsam.Pose3(R_gtsam, t_gtsam)
    # camera1_gtsam = gtsam.PinholeCameraCal3_S2(pose1, K_gtsam)

    # p0 =  np.array([0,0,0])
    # print(camera1_gtsam.project(p0))
    # print(cam_param.proj2img(p0))

    return cam_param


def view_triangulation():
    detections = {f'camera_{i}': [] for i in range(1,7)}
    bag = rosbag.Bag('/home/qingyu/Downloads/20241209_tennis/data1_2024-12-09-17-32-32.bag')
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    date = 'Jul18'
    cam_params = [read_cam_calibration(f'conf/camera/{cname}_calibration_{date}.yaml') for cname in camera_names]
    cam_params_dict =  {'camera_'+str(i+1):cam_params[i] for i in range(6)}

 
    # read all detections
    prev_dets = None
    prev_camid = None
    points3d = []
    for topic, msg, t in bag.read_messages():
        curr_camid = topic.split('/')[1]
        curr_dets = [np.array([p.x, p.y]) for p in msg.points]
        # triangulate
        if prev_dets is not None and prev_camid != curr_camid:
            # pairwise triangulation
            pts3d, bp_errors = [], []
            for prev_det, curr_det in zip(prev_dets, curr_dets):
                p3d = triangulate(prev_det, curr_det, cam_params_dict[prev_camid], cam_params_dict[curr_camid])
                pts3d.append(p3d)
                bp_errors.append(np.linalg.norm(cam_params_dict[prev_camid].proj2img(p3d) - prev_det) + np.linalg.norm(cam_params_dict[curr_camid].proj2img(p3d) - curr_det))
            
            # filter pts3d by reprojection error
            bp_errors = np.array(bp_errors)
            pts3d = np.array(pts3d)
            mask = bp_errors < 10
            pts3d = pts3d[mask]

            # check if ball is in the court
            if len(pts3d) > 0:
                mask = np.logical_and(pts3d[:,0] > 0, pts3d[:,0] < 23.77)
                mask = np.logical_and(mask, np.logical_and(pts3d[:,1] > -5.5, pts3d[:,1] < 5.5))
                pts3d = pts3d[mask]

            if len(pts3d) > 1 and len(points3d) > 0:
                # choose closest point to the latest point3d
                latest_point3d = points3d[-1]
                dists = np.linalg.norm(pts3d - latest_point3d, axis=1)
                idx = np.argmin(dists)
                points3d.append(pts3d[idx])

            elif len(pts3d) == 1:
                pts3d = pts3d[0]
                points3d.append(pts3d)
            

        prev_dets = curr_dets
        prev_camid = curr_camid

    bag.close()
    
    # plot out the detections
    points3d = np.array(points3d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()


def main():
    # view_points()
    view_triangulation()



if __name__ == '__main__':
    main()