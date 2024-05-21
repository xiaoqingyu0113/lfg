import time
import contextlib
import numpy as np
from tensorboardX import SummaryWriter
from matplotlib.figure import Figure
import torch
from typing import Dict, List, Tuple
from pycamera import CameraParam, triangulate

# define a function to augment the trajectory data from torch.tensor [seq_len, 4]
# to [batch, seq_len, 4], where 4 means [t,x,y,z]. 
# Rotate along z axis randomly at the first point
# the rotation origin is the first point
# the rotation angle need to be recorded for the future use
def augment_trajectory(data:torch.Tensor, batch:int, max_angle:float, device:str='cpu'):
    '''
        data: [seq_len, 4]
        batch: int
        max_angle: float
    '''
    seq_len = data.shape[0]
    t = data[:,0]
    x = data[:,1] - data[0,1]
    y = data[:,2] - data[0,2]
    z  = data[:,3]
  
  
    t = t.unsqueeze(0).repeat(batch,1)
    x = x.unsqueeze(0).repeat(batch,1)
    y = y.unsqueeze(0).repeat(batch,1)
    z = z.unsqueeze(0).repeat(batch,1)
    # rotate along z axis
    angle = torch.rand(batch, device=device) * max_angle
    angle = angle.unsqueeze(1).repeat(1,seq_len)
    x_rot = x * torch.cos(angle) - y * torch.sin(angle) + data[0,1]
    y_rot = x * torch.sin(angle) + y * torch.cos(angle) + data[0,2]
    return torch.stack((t,x_rot,y_rot,z), dim=2), angle

def unaugment_trajectory(data:torch.Tensor, angle:torch.Tensor):
    '''
        data: [batch, seq_len, 3]
        angle: [batch, seq_len]
    '''
    batch = data.shape[0]
    seq_len = data.shape[1]
    x = data[:,:,0] - data[:,0:1,0]
    y = data[:,:,1] - data[:,0:1,1]
    z = data[:,:,2] 
    x_rot = x * torch.cos(angle) + y * torch.sin(angle) + data[:,0:1,0]
    y_rot = -x * torch.sin(angle) + y * torch.cos(angle) +  data[:,0:1,1]
    return torch.stack((x_rot,y_rot,z), dim=2)


@contextlib.contextmanager
def timeit(name:str):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f} seconds")

def plot_to_tensorboard(writer:SummaryWriter, tag:str, figure:Figure, global_step:int):
    """
    Converts a matplotlib figure to a TensorBoard image and logs it.

    Parameters:
    - writer: The TensorBoard SummaryWriter instance.
    - tag: The tag associated with the image.
    - figure: The matplotlib figure to log.
    - global_step: The global step value to associate with this image.
    """
    # Draw the figure canvas
    figure.canvas.draw()

    # Convert the figure canvas to an RGB image
    width, height = figure.canvas.get_width_height()
    img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    # Convert RGB to BGR format (which OpenCV uses)
    img = img[:, :, [2, 1, 0]]

    # Add an image channel dimension (C, H, W)
    img = np.moveaxis(img, -1, 0)

    # Convert to float and scale to [0, 1]
    img = img / 255.0

    # Log the image to TensorBoard
    writer.add_image(tag, img, global_step)


def get_uv_from_3d(y, cam_id_list, camera_param_dict):
    '''
        Get the uv coordinates from 3D positions
        return seq, 2
    '''
    uv_list = []
    for yi, cm in zip(y, cam_id_list):
        uv = camera_param_dict[cm].proj2img(yi)
        uv_list.append(uv)
        
    return torch.stack(uv_list, dim=0)


def compute_stamped_triangulations(data:np.ndarray, camera_param_dict:Dict[str, CameraParam]):
    '''
        Compute the 3D positions of the stamped points

        data: [seq_len, 4]

        output: [seq_len-1, 4]
    '''
    positions = []
    stamp = []
    for data_left, data_right in zip(data[0:-1], data[1:]):
        uv_left = data_left[4:6]
        uv_right = data_right[4:6]
        camid_left = str(int(data_left[3]))
        camid_right = str(int(data_right[3]))
        if camid_left != camid_right:
            p = triangulate(uv_left, uv_right, camera_param_dict[camid_left], camera_param_dict[camid_right])
            positions.append(p)
            stamp.append(data_right[2])
    positions = np.array(positions)
    stamp = np.array(stamp)
    return np.hstack((stamp.reshape(-1, 1), positions))