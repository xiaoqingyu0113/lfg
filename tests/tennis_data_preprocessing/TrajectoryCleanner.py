import numpy as np
import yaml
from pycamera import triangulate, CameraParam, set_axes_equal
from draw_util import draw_util
import matplotlib.pyplot as plt
import mplcursors
import glob
from pathlib import Path
import json

OUTPUT_FOLDER = 'data/real/tennis_triangulated_spin'

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

def detections2points3d(detections,tid_offset=0):
    DEBUG = True # if True, detection_filename should be provided

    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    date = 'Dec13'
    cam_params = [read_cam_calibration(f'conf/camera/{cname}_calibration_{date}_pose_kpts.yaml') for cname in camera_names]
    cam_params_dict =  {'camera_'+str(i+1):cam_params[i] for i in range(6)}


    prev_time = None
    prev_uv = None
    prev_camera_id = None
    prev_traj_idx = 0

    points3d = []
    
    for det in detections:
        traj_idx, data_idx, timestamp, camera_id, u, v = det
        # triangulate the 3d point
        if prev_time is not None \
            and prev_camera_id != camera_id \
            and traj_idx == prev_traj_idx \
            and timestamp - prev_time < 0.010:

            prev_camparam = cam_params_dict[prev_camera_id]
            camparam = cam_params_dict[camera_id]

            p = triangulate(np.array(prev_uv), np.array([u, v]), prev_camparam, camparam)
            # repro_error = np.linalg.norm(camparam.proj2img(p) - np.array([u, v]))
            # loc_error = np.linalg.norm(p - np.array(points3d[-1][2:5])) if len(points3d) > 0  else 0
            # loc_error = np.inf if prev_traj_idx != traj_idx else loc_error

            # if repro_error < 120:
            points3d.append([traj_idx+tid_offset, timestamp, p[0], p[1], p[2], 0, 0, 0, 0, 1, 0]) # placeholder for v and w

            if DEBUG:
                pass
        
        if traj_idx == prev_traj_idx +1:
            points3d.append([prev_traj_idx+tid_offset, timestamp, np.nan, np.nan, np.nan, 0, 0, 0, 0, 1, 0]) # placeholder for v and w

        prev_time = timestamp
        prev_uv = [u, v]
        prev_camera_id = camera_id
        prev_traj_idx = traj_idx
    
    return np.array(points3d)

def generate_3d_dataset_without_plt_process(detection_folder):
    detection_files = glob.glob(detection_folder + '/*.json')
    print(f'Found {len(detection_files)} detection files in {detection_folder}')

    tid_offset = 0

    for det_count, detection_file in enumerate(detection_files):
        print(f'Processing ({det_count+1}/{len(detection_files)}) {detection_file}')
        # load the detection file
        with open(detection_file, 'r') as f:
            detections = json.load(f)
        

        # flatten the detections
        flattend_detections = []
        start_time = detections['camera_1'][0][2] # zero the initial time
        for camera_id, points in detections.items():
           for p in points:
                p[3] = camera_id
                p[2] -= start_time # set time w.r.t the first detection
                flattend_detections.append(p)

        flattend_detections.sort(key=lambda x: (x[0], x[2]))
        points = detections2points3d(flattend_detections , tid_offset)

   
        tid_offset = int(points[-1, 0]) + 1
        print(f"tid_offset = {tid_offset}")

        # save the points [trajectory_idx, timestamp, x, y, z, 0,0,0,1,0,0]
        # the last 6 values are placeholders for v and w
        detection_filename_ = detection_file.split('/')[-1].split('.')[0]
        ppp = np.column_stack((np.ones_like(points[:, 1])*points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4], np.zeros_like(points[:, 1]), np.zeros_like(points[:, 1]), np.zeros_like(points[:, 1]), np.ones_like(points[:, 1]), np.zeros_like(points[:, 1]), np.zeros_like(points[:, 1])))
        np.savetxt(f'{OUTPUT_FOLDER}/{detection_filename_}.txt', ppp, fmt='%f')



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
            np.savetxt(f'data/real/tennis_kpts/{detection_filename_}_{current_traj_idx:02d}.txt', ppp, fmt='%f')
            print(f'Saved to data/real/tennis_kpts/{detection_filename_}_{current_traj_idx:02d}.txt')
            
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


def check_lowest_z0():
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
    print(f"min mean = {np.min(lowest_z)}")
    print(f"max mean = {np.max(lowest_z)}")


def view_trajectory_from_file(traj_file):
    points = np.loadtxt(traj_file)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 2], points[:, 3], points[:, 4],linewidth=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    draw_util.draw_tennis_court_outline(ax)
    set_axes_equal(ax)
    plt.show()
    plt.close()     


def show_det_gif(det_file):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import FFMpegWriter

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    with open(det_file, 'r') as f:
        detections = json.load(f)

    points_cam = [np.array(v) for v in detections.values() if len(v) > 0]
    for idx, pc in enumerate(points_cam):
        pc[:, 3] = idx + 1  # set camera id
    points_cam = np.vstack(points_cam)
    points_cam = points_cam[np.lexsort((points_cam[:, 0], points_cam[:, 2]))]  # sort by trajectory index and timestamp


    bg_image_files = [
        'conf/camera/22495525_calibration_Dec13_pose_kpts.jpg',
        'conf/camera/22495526_calibration_Dec13_pose_kpts.jpg',
        'conf/camera/22495527_calibration_Dec13_pose_kpts.jpg'
    ]

    bg_images = [plt.imread(bg_image_file) for bg_image_file in bg_image_files]

    max_length = len(points_cam)
    print(f'max_length = {max_length}')

    skip = 5
    def update(frame):
        data = points_cam[frame*skip]
        tid, _, curr_t, camera_idx, u,v, = data
        camera_idx = int(camera_idx)
        ax = axes[camera_idx-1]
        ax.clear()
        ax.imshow(bg_images[camera_idx-1])
        ax.scatter(u, v, s=10, c='r', label='Current Point')

        point_so_far = points_cam[:frame*skip]
        ax.plot(point_so_far[point_so_far[:, 3] == camera_idx, 4], 
                point_so_far[point_so_far[:, 3] == camera_idx, 5], 
                linewidth=1.0)

        ax.set_title(f'Cam{camera_idx} | Traj{tid} | Time {curr_t - points_cam[0,2]:.3f}s')

        ax.set_axis_off()
        print(f"Camera {idx+1}: Frame {frame*skip}/{max_length-1}")
 
    ani = FuncAnimation(fig, update, frames=max_length//skip)
    ani.save('detections.mp4', writer=FFMpegWriter(fps=20))
    plt.close(fig)

if __name__ == '__main__':    
    # generate_3d_dataset_without_plt_process('data/real/detections_tennis_spin')

    # dir = Path('data/real/tennis_triangulated_spin')
    # txtfiles = dir.glob('*.txt')    
    # for txtfile in txtfiles:
    #     print(txtfile.name)
    #     view_trajectory_from_file(txtfile)
    # view_trajectory_from_file("data/real/tennis_triangulated_spin/spin_n1_vel_15_bag2.txt")    

    show_det_gif(Path('data/real/detections_tennis_spin/spin_n2_vel_15_bag2.json'))