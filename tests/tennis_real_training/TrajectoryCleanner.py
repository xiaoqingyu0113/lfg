import numpy as np
import yaml
from pycamera import triangulate, CameraParam, set_axes_equal
import matplotlib.pyplot as plt
import mplcursors

import json

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

def detections2points3d(detections, detection_filename):
    DEBUG = True # if True, detection_filename should be provided

    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    date = 'Dec13'
    cam_params = [read_cam_calibration(f'conf/camera/{cname}_calibration_{date}.yaml') for cname in camera_names]
    cam_params_dict =  {'camera_'+str(i+1):cam_params[i] for i in range(6)}


    prev_time = None
    prev_uv = None
    prev_camera_id = None
    prev_traj_idx = None

    points3d = []
    
    for det in detections:
        traj_idx, data_idx, timestamp, camera_id, u, v = det
        # triangulate the 3d point
        if prev_time is not None \
            and prev_camera_id != camera_id \
            and prev_camera_id != 'camera_1' \
            and camera_id != 'camera_1' \
            and prev_camera_id != 'camera_6' \
            and camera_id != 'camera_6' \
            and traj_idx == prev_traj_idx \
            and timestamp - prev_time < 0.010:

            prev_camparam = cam_params_dict[prev_camera_id]
            camparam = cam_params_dict[camera_id]

            p = triangulate(np.array(prev_uv), np.array([u, v]), prev_camparam, camparam)
            repro_error = np.linalg.norm(camparam.proj2img(p) - np.array([u, v]))

            loc_error = np.linalg.norm(p - np.array(points3d[-1][2:5])) if len(points3d) > 0  else 0

            loc_error = np.inf if prev_traj_idx != traj_idx else loc_error

            if repro_error < 120:
                points3d.append([traj_idx, timestamp, p[0], p[1], p[2], 0, 0, 0, 0, 1, 0]) # placeholder for v and w

            if DEBUG:
                pass

        prev_time = timestamp
        prev_uv = [u, v]
        prev_camera_id = camera_id
        prev_traj_idx = traj_idx
    
    return np.array(points3d)


def generate_3d_dataset(detection_filename):
    with open(detection_filename, 'r') as f:
        detections = json.load(f)
    
    max_traj_check = []
    for camera_id, points in detections.items():
        points = np.array(points)
        max_traj_check.append(np.max(points[:, 0]).astype(int))

    print(f'max_traj_check = {max_traj_check}')
    assert len(set(max_traj_check)) == 1, f'max_traj_check = {max_traj_check}'

    max_traj_idx = max_traj_check[0]

    flattend_detections = []
    start_time = detections['camera_1'][0][2]
    for camera_id, points in detections.items():
       for p in points:
            p[3] = camera_id
            p[2] -= start_time
            flattend_detections.append(p)

    flattend_detections.sort(key=lambda x: x[2])
    points = detections2points3d(flattend_detections, detection_filename)

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    current_traj_idx = 0
    mask = points[:, 0] == current_traj_idx
    traj_points = points[mask, 1:5]
    t, x,y, z = traj_points[:,0], traj_points[:,1], traj_points[:,2], traj_points[:,3]
    lineplot = ax.plot(x,y,z)[0]
    scatter = ax.scatter(x,y,z, s=3, c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    cursor = mplcursors.cursor(scatter)

    selected_idx = None
    @cursor.connect("add")
    def on_add(sel):
        nonlocal selected_idx
        # Retrieve the index of the selected point
        selected_idx = sel.index
    

    def on_key(event):
        nonlocal t, x,y,z,selected_idx, scatter, lineplot, current_traj_idx, detection_filename

        if 'd' == event.key:
            t = np.delete(t, selected_idx)
            x = np.delete(x, selected_idx)
            y = np.delete(y, selected_idx)
            z = np.delete(z, selected_idx)


            lineplot.set_data(x, y)
            lineplot.set_3d_properties(z)
            scatter._offsets3d = (x, y, z)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # set_axes_equal(ax)
            fig.canvas.draw()
        if 'S' == event.key:
            # save the points [trajectory_idx, timestamp, x, y, z, 0,0,0,1,0,0]
            detection_filename_ = detection_filename.split('/')[-1].split('.')[0]
            ppp = np.column_stack((np.ones_like(t)*current_traj_idx, t, x, y, z, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.ones_like(t), np.zeros_like(t), np.zeros_like(t)))
            np.savetxt(f'data/real/tennis_no_1_6/{detection_filename_}_{current_traj_idx:02d}.txt', ppp, fmt='%f')
            print(f'Saved to data/real/tennis_no_1_6/{detection_filename_}_{current_traj_idx:02d}.txt')
            
        if 'right' == event.key:
            current_traj_idx = min(current_traj_idx + 1, max_traj_idx)
            mask = points[:, 0] == current_traj_idx
            traj_points = points[mask, 1:5]
            t, x,y, z = traj_points[:,0], traj_points[:,1], traj_points[:,2], traj_points[:,3]
            lineplot.set_data(x, y)
            lineplot.set_3d_properties(z)
            scatter._offsets3d = (x, y, z)
            ax.set_title(f'Trajectory {current_traj_idx}')
            fig.canvas.draw()
        if 'left' == event.key:
            current_traj_idx = max(0, current_traj_idx - 1)
            mask = points[:, 0] == current_traj_idx
            traj_points = points[mask, 1:5]
            t, x,y, z = traj_points[:,0], traj_points[:,1], traj_points[:,2], traj_points[:,3]
            lineplot.set_data(x, y)
            lineplot.set_3d_properties(z)
            scatter._offsets3d = (x, y, z)
            ax.set_title(f'Trajectory {current_traj_idx}')
            fig.canvas.draw()

    cursor.connect("add", on_add)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def load_trajectory():
    folder=  'data/real/tennis_triangulated'
    import glob
    traj_files = glob.glob(folder + '/*.txt')

    lowest_z = []
    for traj_file in traj_files:
        points = np.loadtxt(traj_file)
        print(points.shape)
        lowest_z.append(np.min(points[:, 4]))
    
    print(f'lowest_z = {lowest_z}')
    print(f"mean = {np.mean(lowest_z)}")

import glob
detection_file = glob.glob('data/real/detections_tennis/data1*.json')[0]

generate_3d_dataset(detection_file)
# load_trajectory()
