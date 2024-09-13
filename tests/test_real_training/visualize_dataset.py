from tests.test_real_training.train_real_traj import RealTrajectoryDataset
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pycamera import triangulate, CameraParam, set_axes_equal

class DatasetViewer:
    def __init__(self, master):
        self.dataset = RealTrajectoryDataset('data/real/triangulated')


        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.master.title("Dataset Viewer")
        self.master.geometry("800x600")

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(main_frame)
        self.max_input = tk.Entry(input_frame)
        self.max_input.pack(side=tk.TOP)
        self.max_input.insert(0, "10")
        self.max_input.bind("<Return>", self.on_enter)
        
        self.min_input = tk.Entry(input_frame)
        self.min_input.pack(side=tk.TOP)
        self.min_input.insert(0, "0")
        self.min_input.bind("<Return>", self.on_enter)

        self.num_points = tk.Entry(input_frame)
        self.num_points.pack(side=tk.TOP)
        self.num_points.insert(0, "60")
        self.num_points.bind("<Return>", self.on_enter)


        input_frame.pack(side=tk.LEFT)

        view_frame = tk.Frame(main_frame)
        view_frame.pack(side=tk.LEFT)
        
        
        self.fig = plt.figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=view_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        


    def on_enter(self, event):
        max_idx = int(self.max_input.get())
        min_idx = int(self.min_input.get())
        N = int(self.num_points.get())

        self.update_plot(max_idx, min_idx, N)

    def update_plot(self, max_idx, min_idx, N):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d') 
        for i in range(min_idx, max_idx):
            trajectory = self.dataset[i].cpu().numpy()
            ax.plot(trajectory[:,2], trajectory[:,3], trajectory[:,4])
            ax.scatter(trajectory[:N,2], trajectory[:N,3], trajectory[:N,4])

        set_axes_equal(ax)
        self.canvas.draw()


    def  on_closing(self):
        # Close the Matplotlib figure
        if self.fig:
            plt.close(self.fig)

        # Destroy the Tkinter window
        self.master.destroy()
        

if __name__ == "__main__":
    root = tk.Tk()
    viewer = DatasetViewer(root)
    root.mainloop()