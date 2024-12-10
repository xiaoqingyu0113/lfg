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

    def save_detections(self, event):
        # save the detections to a json file
        if self.states['detections'] is None:
            return
        file = self.states['file']
        detections = self.states['detections']
        # convert the numpy array to list
        for camera_id, points in detections.items():
            detections[camera_id] = points.tolist()

        with open(f'data/real/detections_tennis/{file}.json', 'w') as f:
            json.dump(detections, f, indent=4)

        print(f'saved to {file}.json')

        

    
    def open_bag(self, event):
        file = filedialog.askopenfilename(initialdir="/home/qingyu/Downloads/20241209_tennis")
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
        if self.states['detections'] is None:
            return
        curr = self.states['current_point_idx']
        detections = self.states['detections'][self.camera_id.get()]
        self.states['detections'][self.camera_id.get()] = np.delete(detections, curr, axis=0)
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

        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 1024)
        ax.invert_yaxis()
        ax.set_title(f'current traj: {self.states["current_traj"]}')
        ax.legend()

        # convert the plot to image
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img_tk = ImageTk.PhotoImage(img)

        # display the image on canvas
        self.canvas.create_image(0, 0, image=img_tk, anchor='nw')
        self.canvas.image = img_tk




if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleaner(root)
    root.mainloop()