import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import json
from pathlib import Path
import rosbag

def show_det_mp4_by_timestamp(det_file):
    

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
        print(f"Camera {camera_idx}: Frame {frame*skip}/{max_length-1}")
 
    ani = FuncAnimation(fig, update, frames=max_length//skip)
    ani.save('detections.mp4', writer=FFMpegWriter(fps=20))
    plt.close(fig)

def show_bag_mp4_by_sequence(bagpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bag = rosbag.Bag(bagpath)

    flatten_detections = []
    start_t = None
    for topic, msg, t in bag.read_messages():
        if start_t is None:
            start_t = msg.header.stamp.to_sec()

        camera_id = topic.split('/')[1]
        camera_idx = int(camera_id.split('_')[1])
        for p in msg.points:
            flatten_detections.append([camera_idx, p.x, p.y, msg.header.stamp.to_sec() - start_t])
    flatten_detections = np.array(flatten_detections)  
    bag.close()

    bg_image_files = [
        'conf/camera/22495525_calibration_Dec13_pose_kpts.jpg',
        'conf/camera/22495526_calibration_Dec13_pose_kpts.jpg',
        'conf/camera/22495527_calibration_Dec13_pose_kpts.jpg'
    ]
    bg_images = [plt.imread(bg_image_file) for bg_image_file in bg_image_files]

    max_length = len(flatten_detections)
    skip = 5
    def update(frame):
        camera_idx, u, v, curr_t = flatten_detections[frame*skip]
        camera_idx = int(camera_idx)
        # camera_idx = camera_id - 1
        ax = axes[camera_idx-1]
        ax.clear()
        ax.imshow(bg_images[camera_idx-1])
        ax.scatter(u, v, s=10, c='r', label='Current Point')

        point_so_far = flatten_detections[:frame*skip]
        ax.plot(point_so_far[point_so_far[:, 0] == camera_idx, 1], 
                point_so_far[point_so_far[:, 0] == camera_idx, 2], 
                linewidth=1.0)

        ax.set_title(f'Cam{camera_idx}  | Time {curr_t :.3f}s')

        ax.set_axis_off()
        print(f"Camera {camera_idx}: Frame {frame*skip}/{max_length-1}")
 
    ani = FuncAnimation(fig, update, frames=max_length//skip)
    ani.save('bagfile.mp4', writer=FFMpegWriter(fps=20))
    plt.close(fig)

if __name__ == "__main__":
    # show_det_mp4
    # show_det_mp4_by_timestamp(Path('data/real/detections_tennis_spin/spin_n2_vel_25_bag1.json'))
    show_bag_mp4_by_sequence(Path("/home/core-robotics/bag_files/20250403_sensitivity/spin_n2_vel_25_bag1.bag"))