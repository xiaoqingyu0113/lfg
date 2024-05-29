from pathlib import Path
import tensorflow as tf
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import csv

from draw_util import draw_util
import matplotlib.pyplot as plt


from pycamera import triangulate, CameraParam

from typing import Dict, List, Tuple

from omegaconf import OmegaConf
import hydra

from functools import partial
from synthetic.data_generator import generate_data

import theseus as th
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from lfg.util import compute_stamped_triangulations
from lfg.util import get_uv_from_3d

torch.autograd.set_detect_anomaly(True)

def load_csv(file_path:str) -> np.ndarray:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data, dtype=float)


def get_camera_param_dict(config) -> Dict[str, CameraParam]:
    camera_param_dict = {camera_id: CameraParam.from_yaml(Path(config.camera.folder) / f'{camera_id}_calibration.yaml') for camera_id in config.camera.cam_ids}
    return camera_param_dict


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = load_csv(csv_file)
    
    def __len__(self):
        return int(self.data[-1, 0]) + 1 
    
    def __getitem__(self, idx):
        return self.data[idx == self.data[:, 0].astype(int),:]

def air_model(params, x0, dt):
    '''
    x0 : (batch*seq, 6)
    '''
    v = x0[:,:3]
    w = x0[:,3:6]
    cr = torch.linalg.cross(w, v)
    vn = torch.linalg.norm(v, dim=1, keepdim=True)
    wn = torch.linalg.norm(w, dim=1, keepdim=True)
    feat = torch.cat([v, w, cr, vn, wn], dim=1)

    acc = params.reshape(-1,3,11) @ feat.unsqueeze(-1)

    dt = dt.unsqueeze(-1)
    dp = v * dt
    dv = acc.reshape(-1,3) * dt
    dw = torch.zeros_like(dp,device=DEVICE)

    return torch.cat([dp, dv, dw], dim=1)


def bounce_model(params, vw):
    v = vw[:,0:3]
    w = vw[:,3:6]
    feat = torch.cat([v, w], dim=1)
    vw_new = params.reshape(-1,6,6) @ feat.unsqueeze(-1)
    return vw_new



def est_state_error(optim_vars, aux_vars):
    p_var, state_var = optim_vars
    dt, params = aux_vars
    state = state_var.tensor
    
    batch, state_vec = state.shape
    state = state.reshape(-1, 6)
    dx = air_model(params.tensor, state[:-batch,:], dt.tensor.reshape(-1)).reshape(-1,9)

    dp = dx[:,0:3].reshape(batch,-1,3)
    dv = dx[:,3:6].reshape(batch,-1,3)
    dw = dx[:,6:9].reshape(batch,-1,3)



    dp_curr = p_var.tensor.reshape(batch, -1, 3)

    state = state.reshape(batch, -1, 6)
    dv_curr = state[:,:,0:3].reshape(batch, -1, 3)
    dw_curr = state[:,:,0:3].reshape(batch, -1, 3)


    error_p = dp - torch.diff(dp_curr,dim=1)
    error_v = dv  - torch.diff(dv_curr,dim=1)
    error_w = dw  - torch.diff(dw_curr,dim=1)

    return torch.cat([error_p.reshape(batch,-1), error_v.reshape(batch,-1), error_w.reshape(batch,-1)], dim=1)

def est_pos_prior_error(optim_vars, aux_vars):
    p_var,  = optim_vars
    p_data, = aux_vars
    return p_var.tensor - p_data.tensor


    
def estimate_initial_state(params, stamped_positions, x0_prior=torch.zeros(1,9, device=DEVICE)):
    N_est = stamped_positions.shape[1]
    t = stamped_positions[:,:,0]
    dt_data = th.Variable(torch.diff(t, dim=1), name='dt_data')
    p_data = th.Variable(stamped_positions[:,:,1:].reshape(-1,3*N_est), name='p_data')

    objective = th.Objective()
    # prior
    p_var = th.Vector(3*N_est, name='p_var')
    objective.add(th.AutoDiffCostFunction([p_var], est_pos_prior_error,3*N_est, 
                                          aux_vars=[p_data], name='errfn_prior'))

    # model
    state_var = th.Vector(6*N_est, name='state_var')
    objective.add(th.AutoDiffCostFunction([p_var, state_var], est_state_error,9*(N_est-1),
                                          aux_vars = [dt_data, params], name='errfn_state'))
    

    # solve
    optimizer = th.LevenbergMarquardt(objective)
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(DEVICE)



    theseus_inputs  = {'dt_data': dt_data.tensor.clone(), 
                       'p_data': p_data.tensor.clone(),
                       'p_var': p_data.tensor.clone(),
                       'state_var': x0_prior[:,3:].repeat(1,N_est)}

    
    with torch.no_grad():
        updated_inputs, info = theseus_optim.forward(theseus_inputs)

    return updated_inputs, info

def autoregre_error(optim_vars, aux_vars):
    params,  = optim_vars
    dt_data, p_data, x0 = aux_vars

    batch = p_data.tensor.shape[0]

    state0 = x0.tensor[:,3:]
    dt_data = dt_data.tensor.reshape(-1)

    
    p0 = x0.tensor[:,0:3]
    pN= [p0]
    for dt in dt_data:
        dx = air_model(params.tensor, state0, dt).reshape(-1,9)
        state0 = dx[:,3:] + state0
        pN.append(pN[-1] + dx[:,0:3])

    pN = torch.cat(pN[1:], dim=0)


    # error = pN.reshape(batch,-1) - p_data.tensor[:,3:] # exclude the first position
    error = pN - p_data.tensor.reshape(batch,-1,3)[:,1:,:]
    return torch.linalg.norm(error,dim=2).mean().reshape(batch,-1)




    
def optimize_params(params, x0, stamped_positions):

    N = stamped_positions.shape[1]
    dt_data = th.Variable(torch.diff(stamped_positions[:,:,0], dim=1), name='dt_data')
    p_data = th.Variable(stamped_positions[:,:,1:].reshape(-1,3*N), name='p_data')

    objective = th.Objective()
    objective.add(th.AutoDiffCostFunction([params], 
                                          autoregre_error,
                                            1, 
                                            aux_vars=[dt_data, p_data, x0], 
                                            name='errfn_autoregre'))

    optimizer = th.LevenbergMarquardt(objective, max_iterations=2,step_size=1)
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(DEVICE)

    theseus_inputs  = {'dt_data': dt_data.tensor.clone(), 
                       'p_data': p_data.tensor.clone(),
                       'params': params.tensor.clone(),
                       'x0': x0.tensor.clone()}
    
    with torch.no_grad():
        updated_inputs, info = theseus_optim.forward(theseus_inputs,optimizer_kwargs={'verbose':True})
        loss = objective.error_metric(updated_inputs)


    return updated_inputs, info, loss

def get_data_loaders(config) -> Tuple[DataLoader, DataLoader]:
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


def print_config(config):
    print('------------------- Training task-------------------')
    print(OmegaConf.to_yaml(config.task))
    print('-------------------- Model configuration-------------------')
    print(OmegaConf.to_yaml(config.model))
    print('-------------------- dataset configuration-------------------')
    print(OmegaConf.to_yaml(config.dataset))

def train_loop(config):

    print_config(config)

    # Set seed
    torch.manual_seed(config.model.seed)
    np.random.seed(config.model.seed)

    # Tensorboard writer
    # tb_writer, step_count = get_summary_writer(config)
    
    # camera files 
    camera_param_dict = get_camera_param_dict(config)
    
    # Load data
    train_loader, test_loader = get_data_loaders(config)

    params = th.Vector(tensor=torch.randn(1,3*11, device=DEVICE)*0.001, name='params')

    # training loop
    
    # print(f'initial step: {step_count}')

    for epoch in range(config.model.num_epochs):        
        for i, data in enumerate(train_loader):
            # data (batch_size, seq_len, input_size)
            N_est = 50
            # optim for estimation
            spin = data[:,0,6:9].float().to(DEVICE) # get spin
            stamped_positions = compute_stamped_triangulations(data[0].numpy(), camera_param_dict)
            stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)
            stamped_positions = stamped_positions.unsqueeze(0)

            x0_prior = th.Variable(torch.cat([stamped_positions[:,0,1:],
                                              torch.zeros(1,3,device=DEVICE), 
                                              spin], dim=1),
                                              name='x0_prior')
            solution, info = estimate_initial_state(params, stamped_positions[:,:N_est,:],
                                                                  x0_prior=x0_prior)
            p_update = solution['p_var']
            state_update = solution['state_var'] 
            p0 = p_update.reshape(1, -1, 3)[:,-1,:]
            state0 = state_update.reshape(1, -1, 6)[:,-1,:]
            


            # optim for params
            x0 = th.Vector(tensor=torch.cat([p0, state0], dim=1), name='x0')
            solution, info, loss = optimize_params(params, x0, stamped_positions[:,N_est-1:,:])
            params = th.Vector(tensor = solution['params'] , name='params')

            print(loss)
         
         

    # Close the writer
    # tb_writer.close()





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




