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

from pycamera import triangulate, CameraParam, set_axes_equal
from draw_util import draw_util

class ImageAndPlotViewer:
    def __init__(self, root):
        # Initialize plot attributes
        self.canvas = None
        self.fig = None
        self.detections = None
        self.is_clean = False
        self.clicked = None
        self.curr_filename = None
        self.pos = None
        self.tids = None
        self.times = None
        self.prev_opera = []

        self.json_files = glob.glob("data/real/detections/*.json")
        self.json_files = sorted(self.json_files)
        self.original_json_files = [json_file for json_file in self.json_files if 'clean' not in json_file]
        self.clean_json_files = [json_file for json_file in self.json_files if 'clean' in json_file]
        self.file_length = len(self.original_json_files)
        self.file_index = 0




        self.root = root
        self.root.title("Image and 3D Plot Viewer")

        # Make sure the application closes properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main frame to hold everything
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # buttons and image display
        button_frame = tk.Frame(main_frame)
        self.image_size = (853, 682)
        self.canvas_image = tk.Canvas(main_frame, width=self.image_size[0], height=self.image_size[1], bd=0, highlightthickness=0)
        self.plot_frame = tk.Frame(main_frame)
        # self.canvas_image.config(width=self.image_size[0], height=self.image_size[1])
        self._add_button_panels(button_frame)

        # pack
        button_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas_image.pack(side=tk.LEFT, fill=tk.BOTH)

        # Right side: 3D plot display
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind the canvas to capture mouse clicks
        self.canvas_image.bind("<Button-1>", self.on_click)
        self.canvas_image.bind("<Key>", self.on_key)

        

        # Generate and display the 3D plot
        self.create_3d_plot()
    
    def _reset_buffer(self):
        self.clicked = None
        self.prev_opera = []
        self.pos = None
        self.tids = None

    def _add_button_panels(self, button_frame):
        # Add a dropdown menu to select the camera
        self.selected_camera = tk.StringVar()
        self.selected_camera.set("camera_1")
        self.on_select_camera()
        self.selected_camera.trace_add("write", self.on_select_camera)
        btn_select_camera = tk.OptionMenu(button_frame, self.selected_camera, "camera_1", "camera_2", "camera_3")
        btn_select_camera.pack(pady=10)

        # add a entry to select json file
        self.navigator_frame = tk.Frame(button_frame)
        self.navigator_frame.pack(pady=10)
        
        self.left_button = tk.Button(self.navigator_frame, text="<", command=self.go_left)
        self.left_button.pack(side=tk.LEFT)

        self.input_json = tk.Entry(self.navigator_frame, width=8)
        self.input_json.pack(side=tk.LEFT)
        self.input_json.bind("<Return>", self.on_enter)
        # default to the first file
        self.input_json.insert(0, self.original_json_files[0].split('/')[-1].split('.')[0])
        self.on_enter(None)
        
        self.right_button = tk.Button(self.navigator_frame, text=">", command=self.go_right)
        self.right_button.pack(side=tk.LEFT)

        

        self.compute_triangulation_button = tk.Button(button_frame, text="Compute Triangulation", command=self.create_3d_plot)
        self.compute_triangulation_button.pack(pady=10)

        # add a button to save the json file
        self.save_button = tk.Button(button_frame, text="Save", command=self.on_save)
        self.save_button.pack(pady=10)


    def on_select_camera(self, *args):
        self._reset_buffer()
        print(f"Selected camera: {self.selected_camera.get()}")
        img = Image.open(f"data/real/bg/{self.selected_camera.get()}.jpg")
        if self.detections is not None:
            self.draw_detections_on_image(img, self.detections[self.selected_camera.get()])
        self.display_image(img)

    def display_image(self, img):

        # border to indicate clean or not
        bd = 10
        if self.is_clean:
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]
        image_array = np.array(img)
        image_array[:bd, :] = color
        image_array[-bd:, :] = color
        image_array[:, :bd] = color
        image_array[:, -bd:] = color
        img = Image.fromarray(image_array)

        # draw progress
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=32)

        draw.text((20,20), f"{self.file_index+1}/{self.file_length}", fill=(255,255,255), font=font)

        # resize and display
        img.thumbnail(self.image_size)  # Resize image to fit in the canvas
        self.image_display = ImageTk.PhotoImage(img)
        self.canvas_image.create_image(0, 0, image=self.image_display, anchor=tk.NW)
    
    def set_curr_filename(self, filename):
        curr_entry = self.input_json.get()

        if '!' == curr_entry[0]:
            curr_entry = curr_entry[:-1]
        elif '_clean' in curr_entry:
            curr_entry = curr_entry[:-6]
        # refresh the json files
        self.json_files = glob.glob("data/real/detections/*.json")
        self.json_files = sorted(self.json_files)
        self.original_json_files = [json_file for json_file in self.json_files if 'clean' not in json_file]
        self.clean_json_files = [json_file for json_file in self.json_files if 'clean' in json_file]
        self.file_index = self.original_json_files.index(f"data/real/detections/{curr_entry}.json")

        if '!' == curr_entry[-1]:
            self.curr_filename = f"data/real/detections/{curr_entry[:-1]}.json"
            self.is_clean = False
            print(f"Force to Selected: {self.curr_filename}")
            return

        if any(curr_entry in file_name for file_name in self.clean_json_files):
            if 'clean' not in curr_entry:
                self.curr_filename = f"data/real/detections/{curr_entry}_clean.json"
                self.is_clean = True
            else:
                self.curr_filename = f"data/real/detections/{curr_entry}.json"
                self.is_clean =True
            # self.curr_filename = f"data/real/detections/{curr_entry}_clean.json"
            # self.is_clean = True
        elif any(curr_entry in file_name for file_name in self.original_json_files):
            self.curr_filename = f"data/real/detections/{curr_entry}.json"
            self.is_clean = False
        else:
            self.curr_filename = None
        print(f"Selected: {self.curr_filename}")

    def find_closest_index_and_detection(self, detections, x, y):
        curr_click = np.array([x, y])
        min_dist, min_tid, min_did = 1000, None, None
        for tid, traj in enumerate(detections):
            for did, det in enumerate(traj):
                dist = np.linalg.norm(curr_click - np.array(det[:2]))
                if dist < min_dist:
                    min_dist = dist
                    min_tid = tid
                    min_did = did
        return min_tid, min_did
    
    def draw_detections_on_image(self, img, detections):
        draw = ImageDraw.Draw(img)
        colormap = mlp.colormaps['tab20c']
        
        for tra_idx, detection in enumerate(detections):
            for idx in range(0, len(detection)-1):
                x1, y1 = [int(detection[idx][0]), int(detection[idx][1])]
                x2, y2 = [int(detection[idx+1][0]), int(detection[idx+1][1])]
                # circle
                radius = 3
                color =colormap(tra_idx/10.0)[:3]
                color = tuple([int(c*255) for c in color])
                draw.ellipse([x1-radius, y1-radius, x1+radius, y1+radius], fill=color)
                # line
                draw.line([x1, y1, x2, y2], fill=color)

    def go_left(self):
        if self.curr_filename is not None:
            file_label = self.curr_filename.split('/')[-1].split('.')[0]
            if '_clean' in file_label:
                file_label = file_label[:-6]
            original_files = self.original_json_files

            file_idx = original_files.index(f"data/real/detections/{file_label}.json")
            if file_idx > 0:
                self.input_json.delete(0, tk.END)
                self.input_json.insert(0, original_files[file_idx-1].split('/')[-1].split('.')[0])
                self.on_enter(None)

    def go_right(self):
        if self.curr_filename is not None:
            file_label = self.curr_filename.split('/')[-1].split('.')[0]
            if '_clean' in file_label:
                file_label = file_label[:-6]
            original_files = self.original_json_files

            file_idx = original_files.index(f"data/real/detections/{file_label}.json")
            if file_idx < len(original_files)-1:
                self.input_json.delete(0, tk.END)
                self.input_json.insert(0, original_files[file_idx+1].split('/')[-1].split('.')[0])
                self.on_enter(None)

    def on_click(self, event):
        self.canvas_image.focus_set()
        camera = self.selected_camera.get()
        img = Image.open(f"data/real/bg/{self.selected_camera.get()}.jpg")
        W, H = img.size
        x, y = event.x * W/self.image_size[0], event.y * H/self.image_size[1]
        print(f"Clicked at: {x}, {y}")

        if self.detections is not None:
            min_tid, min_did = self.find_closest_index_and_detection(self.detections[camera], x, y)
            self.clicked = [min_tid, min_did]
            draw = ImageDraw.Draw(img)

            # draw all detections
            self.draw_detections_on_image(img, self.detections[camera])

            colormap = mlp.colormaps['tab20c']
            # highlight the trajectory
            for idx in range(0, len(self.detections[camera][min_tid])-1):
                color = colormap(min_tid/10.0)[:3]
                x1, y1 = [int(self.detections[camera][min_tid][idx][0]), int(self.detections[camera][min_tid][idx][1])]
                x2, y2 = [int(self.detections[camera][min_tid][idx+1][0]), int(self.detections[camera][min_tid][idx+1][1])]
                # draw.line([x1, y1, x2, y2], fill= tuple([int(c*255*1.2) if c*1.2 < 1.0 else 255 for c in color ]))
                # draw.ellipse([x1-3, y1-3, x1+3, y1+3], fill= tuple([int(c*255*1.2) if c*1.2 < 1.0 else 255 for c in color ]))
                draw.line([x1, y1, x2, y2], fill= (255,155,155))
                draw.ellipse([x1-3, y1-3, x1+3, y1+3], fill=(255, 155, 155))

            # draw the clicked detection
            xc, yc = self.detections[camera][min_tid][min_did][:2]
            draw.ellipse([xc-3, yc-3, xc+3, yc+3], fill=(255, 0, 0))

        self.display_image(img)
    
    def delete_detection(self):
        tid, did = self.clicked
        camera = self.selected_camera.get()
        if len(self.detections[camera][tid]) > 0:
            det = self.detections[camera][tid].pop(did)
            self.prev_opera.append(['d', [tid, did, det]])
            print(f"Deleted detection: {det}")

    def delete_to_end(self):
        camera = self.selected_camera.get()
        tid, did = self.clicked
        if len(self.detections[camera][tid]) > 0:
            self.prev_opera.append(['e', [tid, did, self.detections[camera][tid][did:]]])
            self.detections[camera][tid] = self.detections[camera][tid][:did]

    def undo_operation(self):
        if len(self.prev_opera) > 0:
            key, (tid, did, data) = self.prev_opera.pop()
            camera = self.selected_camera.get()
            if key == 'e':
                self.detections[camera][tid].extend(data)
                print('undo delete to end')
            elif key == 'd':
                self.detections[camera][tid].insert(did, data)
                print('undo delete')
            elif key == 'x':
                # insert the whole trajectory to tid index
                self.detections[camera].insert(tid, data)
                print('undo delete whole trajectory')
    
    def delete_whole_trajectory(self):
        tid, did = self.clicked
        camera = self.selected_camera.get()
        if len(self.detections[camera][tid]) > 0:
            self.prev_opera.append(['x', [tid, did, self.detections[camera][tid]]])
            self.detections[camera].pop(tid)

    def on_key(self, event):
        key = event.keysym
        if key == 'd':
            self.delete_detection()
        if key == 'e':
            self.delete_to_end()
        if key == 'u':
            self.undo_operation()
        if key == 'x':
            self.delete_whole_trajectory()
        img = Image.open(f"data/real/bg/{self.selected_camera.get()}.jpg")
        self.draw_detections_on_image(img, self.detections[self.selected_camera.get()])
        self.display_image(img)

    def on_enter(self, event):
        self._reset_buffer()
        self.set_curr_filename(self.input_json.get())
        with open(self.curr_filename, 'r') as f:
            self.detections = json.load(f)
        
        # Display the detections
        camera = self.selected_camera.get()
        img = Image.open(f"data/real/bg/{camera}.jpg")
        draw = ImageDraw.Draw(img)
        W,H = img.size
        
        self.draw_detections_on_image(img, self.detections[camera])
        self.display_image(img)
        self.create_3d_plot()

    def on_save(self):
        # save clean json file
        if 'clean' not in self.curr_filename:
            save_filename = self.curr_filename.split('.')[0] + '_clean.json'
        else:
            save_filename = self.curr_filename

        with open(save_filename, 'w') as f:
            json.dump(self.detections, f)
        print(f"Saved to {save_filename}")

        # save detection file
        pos_to_save = []
        spin_text = save_filename.split('/')[-1].split('.')[0][:-6]
        vx, vy, vz = 0, 0, 0 # dummy placeholder
        wx = int(spin_text[1]) if spin_text[0] == 'p' else -int(spin_text[1])
        wy = int(spin_text[4]) if spin_text[3] == 'p' else -int(spin_text[4])
        wz = 0
        for tid, t, p in zip(self.tids, self.times, self.pos):
            pos_to_save.append([tid, t, p[0], p[1], p[2], vx, vy, vz, wx, wy, wz])
            
        # save to csv
        np.savetxt(f"data/real/triangulated/{spin_text}.csv", pos_to_save, delimiter=',', fmt='%d, %f, %f, %f, %f, %d, %d, %d, %d, %d, %d')
        print(f"Saved to data/real/triangulated/{spin_text}.csv")

   
    
    def triangulate_detections(self):
        if self.detections is not None:
            # flatten the detections
            flattened_detections = []
            for detections in self.detections.values():
                for tid, traj in enumerate(detections):
                    for did, det in enumerate(traj):
                        det.append(tid)
                        det.append(did)
                        flattened_detections.append(det)

            flattened_detections = sorted(flattened_detections, key=lambda x: (x[2], x[3]))
            

            cam_ids = {'camera_1': '22276213', 'camera_2': '22276209', 'camera_3': '22276216'} # dont change order
            camera_param_dict = {camera_id: CameraParam.from_yaml(f'conf/camera/{camera_id}_calibration.yaml') for camera_id in cam_ids.values()}

            pos = []
            tids = []
            times = []
            for det_id in range(len(flattened_detections)-1):
                left_det = flattened_detections[det_id]
                right_det = flattened_detections[det_id+1]
                left_cam, right_cam = left_det[4], right_det[4]
                left_tid, right_tid = left_det[5], right_det[5]
                if left_cam != right_cam:
                    left_param, right_param = camera_param_dict[cam_ids[left_cam]], camera_param_dict[cam_ids[right_cam]]
                    p = triangulate(left_det[:2], right_det[:2], left_param, right_param)
                    #backproject to check
                    uv_left_bp = left_param.proj2img(p)
                    uv_right_bp = right_param.proj2img(p)

                    if np.linalg.norm(uv_left_bp - np.array(left_det[:2])) < 10 and np.linalg.norm(uv_right_bp - np.array(right_det[:2])) < 10:
                        if left_tid == right_tid:
                            pos.append(p)
                            tids.append(left_tid)
                            times.append((left_det[2] + right_det[2] + 1e-9*(left_det[3] + right_det[3]))/2.0)

            self.pos = np.array(pos)
            self.tids = np.array(tids)
            self.times = np.array(times) - times[0] # relative time, otherwise number exceeds
        else:
            self.pos = None
            self.tids = None
            self.times = None
    
        
    
    def create_3d_plot(self):

        self.triangulate_detections()

        # Close the previous plot if it exists
        # if self.fig:
        #     plt.close(self.fig)
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        if self.pos is not None:
            self.ax.clear()
            for tid in np.unique(self.tids):
                colormap = mlp.colormaps['tab20c']
                color =colormap(tid/10.0)[:3]
                mask = self.tids == tid
                self.ax.plot(self.pos[mask, 0], self.pos[mask, 1], self.pos[mask, 2], color=color, label=f'tid={tid}')
            set_axes_equal(self.ax)
            draw_util.draw_pinpong_table_outline(self.ax)
            self.ax.legend()
            self.ax.set_title('Triangulated 3D plot')
            # set view angle
            self.ax.view_init(elev=30, azim=165)

            
        # Embed the plot in the Tkinter window
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_closing(self):
        # Close the Matplotlib figure
        if self.fig:
            plt.close(self.fig)

        # Destroy the Tkinter window
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    viewer = ImageAndPlotViewer(root)
    root.mainloop()
