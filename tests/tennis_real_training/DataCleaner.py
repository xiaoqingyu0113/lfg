import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import glob
import json
import matplotlib as mlp
import rosbag
from pycamera import triangulate, CameraParam, set_axes_equal
import yaml


def read_from_bag(bag_file):
    bag = rosbag.Bag(bag_file)
    detections = {f'camera_{i}': [] for i in range(1, 7)}
    for topic, msg, t in bag.read_messages():
        camera_id = topic.split('/')[1]
        for p in msg.points:
            detections[camera_id].append([-1, 0, msg.header.stamp.to_sec(),0, p.x, p.y]) # [traj_idx, data_idx, timestamp, camera_id, u, v, w0]
    bag.close()

    for camera_id, points in detections.items():
        detections[camera_id] = np.array(points)
    return detections

class DataCleaner:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Cleaner")
        self.master.geometry("1280x1024")
        self.master.configure(background="white")
        self.master.resizable(False, False)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)


        # press 'o' to open a bag file inside folder ~/Downloads/20241209_tennis
        self.master.bind('o', self.open_bag)
        # press arrow to change the current index
        self.master.bind('<Left>', self.prev_point)
        self.master.bind('<Right>', self.next_point)
        self.master.bind('<Up>', self.increase_traj)
        self.master.bind('<Down>', self.decrease_traj)
        
        # double press arrow to change large step (control + arrow)
        self.master.bind('<Control-Left>', self.more_prev_point)
        self.master.bind('<Control-Right>', self.more_next_point)

        # save the detection dataset
        self.master.bind('s', self.save_detections)
        self.master.bind('d', self.delete_point)





        # create dropdown menu for selecting camera, int from camera_1 to camera_6
        self.camera_id = tk.StringVar()
        options = [f'camera_{i}' for i in range(1, 7)]
        self.camera_id.set(options[0])
        # add callback function option menu
        self.camera_menu = tk.OptionMenu(self.master, self.camera_id, *options, command=self.draw_detections)
        
        
        self.camera_menu.pack()

        self.states = {
            "file": None,
            "detections": None,
            'current_traj': 0,
            'current_point_idx': 0,
            'figure': None,
            'axis': None,
        }

        self.canvas = tk.Canvas(self.master, width=896, height=717)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-3>", self.on_rclick)
        self.canvas.bind("<Button-2>", self.on_mclick)
        self.canvas.pack()

        # draw the detections on canvas
        self.draw_detections(event=None)

        # press 'q' to quit

    def on_closing(self):
        # Close the Matplotlib figure
        if self.states['figure'] is not None:
            plt.close(self.states['figure'])

        # Destroy the Tkinter window
        self.master.destroy()

    def on_click(self, event):
        x = (event.x -100)/(721-100) * 1280
        y = (event.y - 72)/(535-72) * 1024
        print(f'clicked at {x}, {y}')
        self.states['on_click'] = [x,y]

    def on_drag(self, event):
        x = (event.x -100)/(721-100) * 1280
        y = (event.y - 72)/(535-72) * 1024
        x1, y1 = self.states['on_click']
        event_x1 = x1 * (721-100) / 1280 + 100
        event_y1 = y1 * (535-72) / 1024 + 72

        # draw rectangle
        self.canvas.delete('rect')
        self.canvas.create_rectangle(event_x1, event_y1, event.x, event.y, outline='green', tag='rect')

    def on_mclick(self, event):
        x = (event.x -100)/(721-100) * 1280
        y = (event.y - 72)/(535-72) * 1024

        # find the closest point to the click in this trajectory, set 'current_point_idx' to that index
        detections = self.states['detections'][self.camera_id.get()]
        curr_traj_idx = np.abs(detections[:, 0] - self.states['current_traj']) < 0.1
        curr_traj_points = detections[curr_traj_idx]
        dists = np.linalg.norm(curr_traj_points[:, 4:6] - np.array([x, y]), axis=1)
        idx = np.argmin(dists)
        self.states['current_point_idx'] = np.where(curr_traj_idx)[0][idx]

        self.draw_detections(event=None)

    def on_release(self, event):
        x2 = (event.x -100)/(721-100) * 1280
        y2 = (event.y - 72)/(535-72) * 1024

        x1, y1 = self.states['on_click']

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        curr_traj = self.states['current_traj']
        # select point in the rectangle for current trajectory
        detections = self.states['detections'][self.camera_id.get()]
        mask = np.logical_and(detections[:, 0] == curr_traj, np.logical_and(detections[:, 4] > x1, detections[:, 4] < x2))
        mask = np.logical_and(mask, np.logical_and(detections[:, 5] > y1, detections[:, 5] < y2))
        
        selected_points = detections[mask]
        self.states['selected_points'] = selected_points

        self.draw_detections(event=None)

    def on_rclick(self, event):
        self.states['selected_points'] = []
        self.draw_detections(event=None)
            


    def save_detections(self, event):
        # save the detections to a json file
        if self.states['detections'] is None:
            return
        file = self.states['file']
        detections = self.states['detections'].copy()
        # convert the numpy array to list
        for camera_id, points in detections.items():
            detections[camera_id] = points.tolist()

        with open(f'data/real/detections_tennis/{file}.json', 'w') as f:
            json.dump(detections, f, indent=4)

        print(f'saved to {file}.json')

        

    
    def open_bag(self, event):
        file = filedialog.askopenfilename(initialdir="~/Downloads/20241209_tennis")
        if file and file.endswith('.bag'):
            self.states['detections'] = read_from_bag(file)
            # draw the detections on canvas
            self.draw_detections(event=None)
            # name of the bag file
            self.states['file'] = file.split('/')[-1].split('.')[0]

        elif file and file.endswith('.json'):
            with open(file, 'r') as f:
                self.states['detections'] = json.load(f)
            # convert the list to numpy array
            for camera_id, points in self.states['detections'].items():
                self.states['detections'][camera_id] = np.array(points)
            # draw the detections on canvas
            self.draw_detections(event=None)
            # name of the bag file
            self.states['file'] = file.split('/')[-1].split('.')[0]

    def prev_point(self, event):
        if self.states['detections'] is None:
            return
        curr = self.states['current_point_idx']
        next_idx = max(0, self.states['current_point_idx'] - 1)
        self.states['current_point_idx'] = next_idx
        self.states['detections'][self.camera_id.get()][curr, 0] = self.states['current_traj']
        self.draw_detections(event=None)
    
    def next_point(self, event):
        if self.states['detections'] is None:
            return
        curr = self.states['current_point_idx']
        next_idx = min(len(self.states['detections'][self.camera_id.get()]) - 1, self.states['current_point_idx'] + 1)
        self.states['current_point_idx'] = next_idx
        self.states['detections'][self.camera_id.get()][curr,0] = self.states['current_traj']
        self.draw_detections(event=None)

    def more_prev_point(self, event):
        if self.states['detections'] is None:
            return
        curr = self.states['current_point_idx']
        next_idx = max(0, self.states['current_point_idx'] - 20)
        self.states['current_point_idx'] = next_idx
        self.states['detections'][self.camera_id.get()][next_idx+1: curr+1, 0] = self.states['current_traj']
        self.draw_detections(event=None)
       

    def more_next_point(self, event):
        if self.states['detections'] is None:
            return
        curr = self.states['current_point_idx']
        next_idx = min(len(self.states['detections'][self.camera_id.get()]) - 1, self.states['current_point_idx'] + 20)
        self.states['current_point_idx'] = next_idx
        self.states['detections'][self.camera_id.get()][curr:next_idx,0] = self.states['current_traj']
        self.draw_detections(event=None)
        
    def increase_traj(self, event):
        if self.states['detections'] is None:
            return
        self.states['current_traj'] += 1
        self.draw_detections(event=None)
    
    def decrease_traj(self, event):
        if self.states['detections'] is None:
            return
        self.states['current_traj'] = max(0, self.states['current_traj'] - 1)
        self.draw_detections(event=None)

    def delete_point(self, event):
        # delete selected point
        if 'selected_points' in self.states and len(self.states['selected_points']) > 0:
            selected_points = self.states['selected_points']
            detections = self.states['detections'][self.camera_id.get()]
            for point in selected_points:
                idx = np.where(np.all(detections == point, axis=1))[0]
                detections = np.delete(detections, idx, axis=0)
            self.states['detections'][self.camera_id.get()] = detections
            self.states['selected_points'] = []
            curr = self.states['current_point_idx']
            self.states['current_point_idx'] = min(len(self.states['detections'][self.camera_id.get()]) - 1, curr)
            self.draw_detections(event=None)
        
        # delete current point
        else:
            if self.states['detections'] is None:
                return
            curr = self.states['current_point_idx']
            detections = self.states['detections'][self.camera_id.get()]
            self.states['detections'][self.camera_id.get()] = np.delete(detections, curr, axis=0)
            self.states['current_point_idx'] = min(len(self.states['detections'][self.camera_id.get()]) - 1, curr)
            self.draw_detections(event=None)

    def draw_detections(self, event):
        
        if event is not None and 'camera' in event:
            self.states['current_point_idx'] = 0
            self.states['current_traj'] = 0

        
        if self.states['detections'] is None:
            return
        camera_id = self.camera_id.get()
        points = self.states['detections'][camera_id]

        curr_traj_idx = np.abs(points[:, 0] - self.states['current_traj']) < 0.1
        curr_traj_points= points[curr_traj_idx]
        
        


        # plot using matplotlib
        if self.states['figure'] is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            self.states['figure'] = fig
            self.states['axis'] = ax
        else:
            fig = self.states['figure']
            ax = self.states['axis']
            ax.clear()


        
        ax.scatter(curr_traj_points[:, 4], curr_traj_points[:, 5], s=1, c='b')
        ax.plot(curr_traj_points[:, 4], curr_traj_points[:, 5], c='b', alpha=0.5)
        ax.scatter(points[self.states['current_point_idx'], 4], points[self.states['current_point_idx'], 5], s=3, c='r')
        ax.text(points[self.states['current_point_idx'], 4], points[self.states['current_point_idx'], 5], f'traj={self.states["current_traj"]}', fontsize=12, color='r')

        if 'selected_points' in self.states and len(self.states['selected_points']) > 0:
            selected_points = self.states['selected_points']
            ax.scatter(selected_points[:, 4], selected_points[:, 5], s=3, c='g')

        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 1024)
        ax.invert_yaxis()
        ax.set_title(f'current traj: {self.states["current_traj"]}')
        # ax.legend()

        # convert the plot to image
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img_tk = ImageTk.PhotoImage(img)

        # display the image on canvas
        self.canvas.create_image(0, 0, image=img_tk, anchor='nw')
        self.canvas.image = img_tk

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
    date = 'Jul18'
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
            and prev_camera_id != 'camera_4' \
            and camera_id != 'camera_4' \
            and prev_camera_id != 'camera_3' \
            and camera_id != 'camera_3' \
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


                


def generate_3d_dataset():
    detection_filename = 'data/real/detections_tennis/data7_2024-12-09-18-21-24.json'
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
    for i in range(30):
        mask = points[:, 0] == i
        ax.plot(points[mask, 2], points[mask, 3], points[mask, 4])
    # ax.scatter(points[:,2], points[:,3], points[:,4], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    plt.show()

   
def start_app():
    root = tk.Tk()
    app = DataCleaner(root)
    root.mainloop()


if __name__ == "__main__":
    # start_app()

    generate_3d_dataset()