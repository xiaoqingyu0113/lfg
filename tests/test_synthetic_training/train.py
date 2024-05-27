# Description: Train the RNN model on the synthetic data
from pathlib import Path
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import csv
from mamba_ssm import Mamba

from draw_util import draw_util
import matplotlib.pyplot as plt

from common import load_csv
from pycamera import triangulate, CameraParam

from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import hydra
from functools import partial
from synthetic.data_generator import generate_data

# CONFIG = OmegaConf.load('../../conf/config.yaml')
# MODEL_CONFIG = CONFIG.model[CONFIG.model_name]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from lfg.util import compute_stamped_triangulations
from lfg.util import get_uv_from_3d

# torch.autograd.set_detect_anomaly(True)


def get_camera_param_dict(config) -> Dict[str, CameraParam]:
    '''
        Get camera parameters from config.yaml
    '''
    camera_param_dict = {camera_id: CameraParam.from_yaml(Path(config.camera.folder) / f'{camera_id}_calibration.yaml') for camera_id in config.camera.cam_ids}
    return camera_param_dict

def test_camera_torch(config):
    camera_param_dict = get_camera_param_dict(config)
    p = np.random.rand(3)*5.0
    p = p.tolist()
    cm = list(camera_param_dict.values())[0]
    print('numpy result',cm.proj2img(np.array(p)))
    cm.to_torch()
    print('torch result', cm.proj2img(torch.tensor(p, dtype=torch.float32)))


class MyDataset(Dataset):
    '''
        Dataset, use numpy array for the convinience of triangulation
    '''
    def __init__(self, csv_file):
        self.data = load_csv(csv_file)
    
    def __len__(self):
        return int(self.data[-1, 0]) + 1 
    
    def __getitem__(self, idx):
        return self.data[idx == self.data[:, 0].astype(int),:]


    
def view_triangulations(config):
    '''
        Visualize the 3D positions
    '''
    camera_param_dict = {camera_id: CameraParam.from_yaml(Path(config.camera.folder) / f'{camera_id}_calibration.yaml') for camera_id in config.camera.cam_ids}
    dataset = MyDataset(Path(config.dataset.folder) / config.dataset.camera_data)

    data = dataset[0]
    stamped_positions = compute_stamped_triangulations(data, camera_param_dict)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(stamped_positions[:,1], stamped_positions[:,2], stamped_positions[:,3], label='3D positions')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    for cm in camera_param_dict.values():
        cm.draw(ax,scale=0.20)
    draw_util.set_axes_equal(ax)
    draw_util.set_axes_pane_white(ax)
    draw_util.draw_pinpong_table_outline(ax)
    plt.show()



def get_data_loaders(config) -> Tuple[DataLoader, DataLoader]:
    '''
        Get the data loaders for training and validation
    '''
    dataset = MyDataset(Path(config.dataset.folder) / config.dataset.camera_data)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, \
                                    [int(len(dataset)*config.model.training_data_split), len(dataset) - int(len(dataset)*config.model.training_data_split)])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

def get_summary_writer(config) -> SummaryWriter:
    '''
        Get the summary writer
    '''
    def find_last_step(event_file_path):
        last_step = -1
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file_path):
                if e.step > last_step:
                    last_step = e.step
        except Exception as e:
            print(f"Failed to read event file {event_file_path}: {str(e)}")
        
        return last_step
    
    logdir = Path(config.model.logdir) / config.dataset.name
    initial_step = 0
    if not logdir.exists():
        logdir.mkdir(parents=True)
        tb_writer = SummaryWriter(log_dir=logdir / 'run00')
    else:
        # get the largest number of run in the logdir using pathlib
        paths = list(logdir.glob('*run*'))
        indices = [int(str(p).split('run')[-1]) for p in paths]

        if len(indices) == 0:
            max_run_num = 0
        else:
            max_run_num = max(indices)
        if config.model.continue_training:
            tb_writer = SummaryWriter(log_dir=logdir / f'run{max_run_num:02d}')
            rundir = logdir/f'run{max_run_num:02d}'/'loss_training'
            rundir = list(rundir.glob('events.out.tfevents.*'))
            initial_step = max([find_last_step(str(rd)) for rd in rundir])
        else:
            tb_writer = SummaryWriter(log_dir=logdir / f'run{1+max_run_num:02d}')
    return tb_writer, initial_step


def get_model(config) -> nn.Module:
    '''
        Get the model
    '''

    model_config = config.model
    if model_config.model_name == 'gru':
        from lfg.model import GRUCellModel, gru_autoregr
        model = GRUCellModel(model_config.input_size, model_config.hidden_size, model_config.output_size)
        model.to(DEVICE)
        model_autoregr = gru_autoregr
    elif model_config.model_name == 'mamba_ssm':
        from lfg.model import MambaModel, mamba_autoregr
        model = MambaModel(model_config.d_model, model_config.d_state, model_config.d_conv, model_config.expand, model_config.output_size).to(DEVICE)
        for param in model.parameters():
            param.data.zero_()
        model_autoregr = mamba_autoregr
    elif model_config.model_name == 'spatial':
        from lfg.model import SpatialModel, spatial_autoregr
        model = SpatialModel(model_config.input_size, model_config.hidden_size, model_config.output_size).to(DEVICE)
        model_autoregr = spatial_autoregr
    elif model_config.model_name == 'gru_his':
        from lfg.model import GRUHisModel, gruhis_autoregr
        model = GRUHisModel(model_config.input_size, model_config.history, model_config.hidden_size, model_config.output_size).to(DEVICE)
        model_autoregr = gruhis_autoregr
    elif model_config.model_name == 'mskip':
        from lfg.model import MSkipModel, mskip_autoregr
        model = MSkipModel(model_config.input_size, model_config.c_size, model_config.hidden_size, model_config.output_size).to(DEVICE)
        model_autoregr = mskip_autoregr
    elif model_config.model_name == 'physics':
        from lfg.model import PhysicsModel, physics_autoregr
        model = PhysicsModel(model_config.his_len, model_config.hidden_size).to(DEVICE)
        model_autoregr = physics_autoregr
    elif model_config.model_name == 'physics_kf':
        from lfg.model import PhysicsKFModel, physicskf_autoregr
        model = PhysicsKFModel().to(DEVICE)
        model_autoregr = physicskf_autoregr
    elif model_config.model_name == 'physics_optim':
        from lfg.model import CombinedModel, physics_optim_autoregr
        model = CombinedModel().to(DEVICE)
        model_autoregr = physics_optim_autoregr
    return model, model_autoregr



def train_loss(model, model_pass, data, camera_param_dict, criterion):
    data = data[0] # ignore the batch size

    y, positions = model_pass(model, data, camera_param_dict)


    loss = criterion(y, positions)
   
   
    return loss
    # return loss


        

def test_autoregr_loss(config, model, model_autoregr, test_loader, camera_param_dict, criterion):
    '''
        Validate the GRU model
    '''
    with torch.no_grad():
        test_loss = 0.0
        for data in test_loader:
            N_data = int(config.model.use_data * data.shape[1])
            data = data[:, :N_data, :]
            loss = train_loss(model, model_autoregr, data, camera_param_dict, criterion)
            test_loss += loss.item()

        
    return test_loss / len(test_loader)

def update_spin_info(spin, spin_info):
    for i, key in enumerate(spin_info.keys()):
        if spin[i] > spin_info[key]['max']:
            spin_info[key]['max'] = spin[i]
        if spin[i] < spin_info[key]['min']:
            spin_info[key]['min'] = spin[i]
    return spin_info

def train_loop(config):
    '''
        Train the LSTM model
    '''
    # Set seed
    torch.manual_seed(config.model.seed)
    np.random.seed(config.model.seed)

    # Tensorboard writer
    tb_writer, step_count = get_summary_writer(config)
    

    # camera files 
    camera_param_dict = get_camera_param_dict(config)
    
    # Load data
    train_loader, test_loader = get_data_loaders(config)

    # training loop
    model, model_autoregr = get_model(config)
 

    if config.model.continue_training:
        model_path = Path(tb_writer.get_logdir())/f'model_{config.model.model_name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"model loaded from {model_path}")

    model_autoregr = partial(model_autoregr, config=config)
  

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr_init, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.model.lr_step_size, gamma=config.model.lr_gamma)

    
    spin_info = { 'w0_x':{
        'max': -torch.inf,
        'min': torch.inf
        },
        'w0_y':{
            'max': -torch.inf,
            'min': torch.inf
        },
        'w0_z':{
            'max': -torch.inf,
            'min': torch.inf
        }
    }

    print('------------------- Training task-------------------')
    print(OmegaConf.to_yaml(config.task))
    print('-------------------- Model configuration-------------------')
    print(OmegaConf.to_yaml(config.model))
    print('-------------------- dataset configuration-------------------')
    print(OmegaConf.to_yaml(config.dataset))

    print(f'initial step: {step_count}')

    for epoch in range(config.model.num_epochs):
        total_loss = torch.tensor(0.0).to(DEVICE)
        optimizer.zero_grad()
        batch_traj = config.model.batch_trajectory
        
        for i, data in enumerate(train_loader):
            # data (batch_size, seq_len, input_size)

            spin = data[0,0,6:9].float() # get spin
            update_spin_info(spin, spin_info)

            # fig = visualize_predictions(config, data = next(iter(test_loader))[0] , model = model,  model_autoregr = model_autoregr)
            # tb_writer.add_figure('predictions vs. actuals',
            #                 fig,
            #                 global_step=step_count)
            # raise
            loss = train_loss(model, model_autoregr, data, camera_param_dict, criterion)
            total_loss += loss
            step_count += 1 # for logs

            # batch backward and optimize
            if (i+1) % batch_traj == 0 or i+1 == len(train_loader):
                total_loss /= batch_traj
                # Print loss
                tb_writer.add_scalars(main_tag='loss', tag_scalar_dict={'training':total_loss.item()}, global_step=step_count)
                print(f'Epoch [{epoch+1}/{config.model.num_epochs}], Step [{i+1}/{len(train_loader)}], total_loss: {total_loss.item()}')

                # Backward and optimize
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss = torch.tensor(0.0).to(DEVICE)
                optimizer.zero_grad()
                # fig = visualize_predictions(config, data = next(iter(test_loader))[0] , model = model,  model_autoregr = model_autoregr)
                # tb_writer.add_figure('predictions vs. actuals',
                #                 fig,
                #                 global_step=step_count)
        scheduler.step()

        # Test the model every 1 epochs
        if (epoch+1) % 1 == 0:
            auto_regr_loss = test_autoregr_loss(config, model, model_autoregr, test_loader, camera_param_dict, criterion)
            print(f' \t Autoregressive Loss: {auto_regr_loss}')

            tb_writer.add_scalars(main_tag='loss', tag_scalar_dict={'auto_regr':auto_regr_loss}, global_step=step_count)
            tb_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step_count)
            tb_writer.add_scalars(main_tag='spin_info', tag_scalar_dict={'max_w0_x':spin_info['w0_x']['max'], 'min_w0_x':spin_info['w0_x']['min'],
                                                                        'max_w0_y':spin_info['w0_y']['max'], 'min_w0_y':spin_info['w0_y']['min'],
                                                                        'max_w0_z':spin_info['w0_z']['max'], 'min_w0_z':spin_info['w0_z']['min']}, 
                                                                        global_step=step_count)
        
            fig = visualize_predictions(config, data = next(iter(test_loader))[0] , model = model,  model_autoregr = model_autoregr)
            tb_writer.add_figure('predictions vs. actuals',
                            fig,
                            global_step=step_count)

        # Save model checkpoint
        tb_writer.flush()
        model_save_path = Path(tb_writer.get_logdir())/f'model_{config.model.model_name}.pth'
        torch.save(model.state_dict(), model_save_path )
        print(f"model saved at {model_save_path}")

    # Close the writer
    tb_writer.close()


def visualize_predictions(config, data = None, model = None,  model_autoregr = None):
    # Load data
    if data is None:
        train_loader, test_loader = get_data_loaders(config)
        data = next(iter(test_loader))[0] # ignore the batch size

    # camera dict
    camera_param_dict = get_camera_param_dict(config)
    
    # training loop
    if model is None:
        model, model_autoregr = get_model(config)
        state_dict = torch.load( Path(config.model.model_path) / f'model_{config.model.model_name}.pth')
        model.load_state_dict(state_dict)
        model_autoregr = partial(model_autoregr, config=config)

    model.eval()

    # predict
    with torch.no_grad():
        y, positions = model_autoregr(model, data, camera_param_dict)

    y = y.cpu().numpy() 
    positions = positions.cpu().numpy()

    # visualize
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(projection='3d')
    ax.plot(positions[:,0], positions[:,1], positions[:,2], label='gt')
    ax.plot(y[:,0], y[:,1], y[:,2], 'r', label='pred')
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    draw_util.set_axes_equal(ax)
    draw_util.set_axes_pane_white(ax)
    return fig



@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):

    if cfg.task.generate_data:
        generate_data(cfg)

    if cfg.task.train:
        train_loop(cfg)
    
    # test_camera_torch(cfg)
    # view_triangulations(cfg)

if __name__ == '__main__':
    main()




