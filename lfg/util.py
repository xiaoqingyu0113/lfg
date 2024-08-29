import time
import contextlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure
import torch
from typing import Dict, List, Tuple
import csv
from pathlib import Path
import tensorflow as tf
from pycamera import CameraParam, triangulate


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



def load_csv(file_path:str) -> np.ndarray:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data, dtype=float)



def find_writer_last_step(event_file_path):
        last_step = -1
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file_path):
                if e.step > last_step:
                    last_step = e.step
        except Exception as e:
            print(f"Failed to read event file {event_file_path}: {str(e)}")
        
        return last_step

def get_summary_writer_path(config):
    logdir = Path(config.model.logdir) 

    if not logdir.exists():
        logdir.mkdir(parents=True)
        ret_path =logdir / 'run00'
    else:
        # get the largest number of run in the logdir using pathlib
        paths = list(logdir.glob('*run*'))
        indices = [int(str(p).split('run')[-1]) for p in paths]

        if len(indices) == 0:
            max_run_num = 0
        else:
            max_run_num = max(indices)

        if 'run' in config.model.continue_training:
            ret_path = logdir / config.model.continue_training
        elif 'last' in config.model.continue_training:
            ret_path =logdir / f'run{max_run_num:02d}' 
        elif 'new' in config.model.continue_training:
            ret_path =logdir / f'run{1+max_run_num:02d}'
        else:
            raise  ValueError(f'Invalid config.model.continue_training option. Given {config.model.continue_training}, expected one of [run[number], last, new]')

        # if config.model.continue_training:
        #     ret_path =logdir / f'run{max_run_num:02d}'    
        # else:
        #     ret_path =logdir / f'run{1+max_run_num:02d}'

    return ret_path

def get_summary_writer(config) -> Tuple[SummaryWriter, int]:
    '''
        Get the summary writer
    '''
    initial_step = 0
    run_path = get_summary_writer_path(config)
    tb_writer = SummaryWriter(log_dir=run_path)

    # update initial step if continue training
    if 'new' not in config.model.continue_training:
        loss_dir = run_path/'loss_training'
        loss_dir = list(loss_dir.glob('events.out.tfevents.*'))
        initial_step = max([find_writer_last_step(str(rd)) for rd in loss_dir])

    return tb_writer, initial_step