import os
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'


from pathlib import Path
import csv
# import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


from draw_util import draw_util
import matplotlib.pyplot as plt

from lfg.util import get_summary_writer

from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import hydra
from functools import partial



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def print_cfg(cfg):
    checklist = ['model', 'estimator', 'dataset', 'task']

    txt = ''
    for k in checklist:
        txt += '\n'
        txt += '#'*15 + f' {k} config ' + '#'*15 + '\n'
        cf_k = cfg[k]
        for kk, vv in cf_k.items():
            txt += f'{kk}: {vv}\n'
        txt += '#'*15 + f' {k} ' + '(END)' + '#'*15 + '\n'
    print(txt)


class RealTrajectoryDataset(Dataset):
    def __init__(self, csv_folder, interpolate=300):
        self.interpolate = interpolate
        data_np = self.read_all_csv_from_folder(Path(csv_folder))
        self.data = torch.from_numpy(data_np).to(DEVICE).float()
 

    def read_from_cvs(self, cvs_file):
        with open(cvs_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        return np.array(data, dtype=float)
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        '''
        preprocess the data
        '''
        data = self.make_idx_continuous(data)
        data = self.interpolate_data(data, self.interpolate)

        return data
    
    def make_idx_continuous(self, data: np.ndarray) -> np.ndarray:
        '''
        make the index continuous.
          e.g. [0, 1, 2, 4, 5] -> [0, 1, 2, 3, 4]
          e.g. [1, 2, 3, 4] -> [0, 1, 2, 3]
        '''
        unique_idx = np.unique(data[:, 0])
        for cont_idx, uni_idx in enumerate(unique_idx):
            data[data[:, 0] == uni_idx, 0] = cont_idx
        return data
    
    def interpolate_data(self, data_tmp: np.ndarray, interpolate: int) -> np.ndarray:
        '''
        interpolate data to have equal length for batch learning
        '''
        data = []
        for i in range(int(data_tmp[-1][0])+1):
            mask = data_tmp[:, 0].astype(int) == i
            data_tmp_i = data_tmp[mask, :]
            interp_f = interp1d(data_tmp_i[:, 1], data_tmp_i, axis=0)
            tmax = data_tmp_i[-1, 1]
            tmin = data_tmp_i[0, 1]
            t_avg_spacing = (tmax - tmin) / interpolate
            t_random_noise = np.random.uniform(-t_avg_spacing/2.5, t_avg_spacing/2.5, interpolate)
            t = np.linspace(tmin, tmax, interpolate) + t_random_noise
            t[0] = tmin
            t[-1] = tmax
            data_i = interp_f(t)
            data.append(data_i)
        return np.array(data)
        
        
    def read_all_csv_from_folder(self, folder) -> np.ndarray:
        cvs_files = list(folder.glob('*.csv'))
        for f in cvs_files:
            data_tmp = self.preprocess_data(self.read_from_cvs(f))
            if 'data' in locals():
                data_tmp[:,:, 0] += data[-1,-1,0] + 1
                data = np.vstack((data, data_tmp))
            else:
                data = data_tmp
        return data.astype(np.float32)
    
    def show_info(self):
        data_sizes = []
        for i in range(len(self)):
            data_got = self.data[i]
            data_sizes.append(data_got.shape[0])

    def random_plot(self, ax, indices):
        for i in indices:
            data_got = self.data[i].cpu().numpy()
            ax.plot(data_got[:, 2], data_got[:, 3], data_got[:, 4])
        draw_util.set_axes_equal(ax, zoomin=2.0)
       
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       return self.data[idx]
    
    @classmethod
    def get_dataloaders(cls, config):
        dataset = cls(Path(config.dataset.folder), config.dataset.interpolate)
        total_data_size = len(dataset)
        split_ratio = config.model.training_data_split
        train_data_size = int(total_data_size * split_ratio)
        test_data_size = total_data_size - train_data_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.dataset.batch_size, shuffle=False)
        return train_loader, test_loader


def get_model(cfg):
    '''
    model, estimator, autoregression function
    '''
    from lfg import model_traj
    from lfg import estimator

    Model = getattr(model_traj, cfg.model.name)
    model = Model(*cfg.model.args, **cfg.model.kwargs)
    model = model.to(DEVICE)

    if cfg.estimator.name == 'GT':
        est = None
    elif cfg.estimator.name == 'SlideWindowEstimator':
        Estimator = getattr(estimator, cfg.estimator.name)
        est = Estimator(*cfg.estimator.args, **cfg.estimator.kwargs)
        est = est.to(DEVICE)
    else:
        Estimator = getattr(estimator, cfg.estimator.name)
        est = Estimator(model, **cfg.estimator.kwargs)
        est = est.to(DEVICE)

    autoregr = getattr(model_traj, f'autoregr_{cfg.model.name}')
    return model, est, autoregr

def compute_loss(model, est, autoregr, data, criterion,cfg):
    pN_est = autoregr(data, model, est, cfg)

    # failed to estimate due to estimator
    if pN_est is None:
        return None
    

    pN_gt = data[:, :,2:5]

    loss = 0.0

    if 'pos' in cfg.model.loss_type:
        loss += criterion(pN_est, pN_gt)

    if 'vel' in cfg.model.loss_type:
        d_pN_est = torch.diff(pN_est, dim=1)
        d_pN_gt = torch.diff(pN_gt, dim=1)
        loss += criterion(d_pN_est, d_pN_gt)*100.0
    
    if 'acc' in cfg.model.loss_type :
        dd_pN_est = torch.diff(torch.diff(pN_est, dim=1), dim=1)
        dd_pN_gt = torch.diff(torch.diff(pN_gt, dim=1), dim=1)
        loss += criterion(dd_pN_est, dd_pN_gt)*100.0

    return loss


def compute_valid_loss(model, est, autoregr, test_loader, criterion, cfg):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num = 0
    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            loss = compute_loss(model, est, autoregr, data, criterion, cfg)
            if loss is None:
                continue
            num +=data.shape[0]
            total_loss += loss.item()* data.shape[0]
    model.train()  # Set the model back to training mode
    return total_loss / num

def visualize_traj(model, est, autoregr, data, cfg):
    model.eval()
    single_data = data[:3,:,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        pN_est = autoregr(single_data, model, est, cfg)
        pN_gt = single_data[:, :, 2:5]
        pN_gt = pN_gt.cpu().numpy()
        pN_est = pN_est.cpu().numpy()

        for batch in range(single_data.shape[0]):
            ax.plot(pN_gt[batch, :, 0], pN_gt[batch, :, 1], pN_gt[batch, :, 2], label=f'gt_{batch}')
            ax.plot(pN_est[batch, :, 0], pN_est[batch, :, 1], pN_est[batch, :, 2], label=f'est_{batch}')

        ax.legend()
        draw_util.set_axes_equal(ax, zoomin=2.0)

    model.train()
    return fig





class ParamConstraint:
    def small_param_thresh(model):
        threshold = 1e-20  # Define a threshold
        for name, param in model.named_parameters():
            with torch.no_grad():
                param.data = torch.where(param.data < -threshold,param.data, torch.tensor(-threshold, dtype=param.data.dtype))
                param.data = torch.where(param.data > threshold, param.data, torch.tensor(threshold, dtype=param.data.dtype))

    def positive_param(model):
        for name, param in model.named_parameters():
            if 'model' not in name:
                with torch.no_grad():
                    param.data = torch.where(param.data < 0.0, torch.tensor(0.0, dtype=param.data.dtype), param.data)


def train_loop(cfg):

    print_cfg(cfg)
    
    # get dataloaders
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #train_loader.dataset.dataset.random_plot(ax, range(20))
    #plt.show()

    #raise
    

    # get summary writer
    tb_writer, initial_step = get_summary_writer(cfg)
    tb_writer.add_text('config', f'```yaml\n{OmegaConf.to_yaml(cfg)}\n```',global_step=initial_step)

    # get model
    model, est, autoregr = get_model(cfg)
    if cfg.model.warm_start:
        model.aero_layer.load_state_dict(torch.load('data/archive/aero_model.pth',map_location=DEVICE))
        model.bc_layer.load_state_dict(torch.load('data/archive/bounce_model.pth',map_location=DEVICE))

    if cfg.estimator.name != 'GT':
        est.model = model

    if cfg.model.continue_training:
        model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"model loaded from {model_path}")

        if cfg.estimator.name != 'GT':
            est_path = Path(tb_writer.get_logdir())/f'est_{cfg.estimator.name}.pth'
            est.load_state_dict(torch.load(est_path))
            est.model = model
            print(f"est loaded from {est_path}")

    # set params in both model and est
    opt_params = list(model.parameters()) + [p for n, p in est.named_parameters() if 'model' not in n] if cfg.estimator.name != 'GT' \
                else model.parameters()
    optimizer = torch.optim.Adam(opt_params, lr=cfg.model.lr_init, weight_decay=1e-3, eps=1e-5)
    criterion = nn.L1Loss()

    best_valid_loss = torch.inf
    for epoch in range(cfg.model.num_epochs):
        for i, data in enumerate(train_loader):
            est_size = 0 if cfg.estimator.name == 'GT' else cfg.estimator.kwargs.size
            N_seq = max(int(data.shape[1] * cfg.model.seq_ratio), est_size)
            data = data[:, :N_seq, :] 

            optimizer.zero_grad()
            loss = compute_loss(model,est, autoregr, data, criterion, cfg)
            # fail to estimate due to estimator
            if loss is None:
                continue
            loss.backward()
                       

            # for name, param in model.named_parameters():
            #     if 'aero' in name and 'dec.4' in name:
            #         print(name, param)


            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, norm_type = 2.0, error_if_nonfinite=True)
            optimizer.step()
            # Constraint.small_param_thresh(model)

            # est params should be positive
            if cfg.estimator.name != 'GT':
                ParamConstraint.positive_param(est)

            # write to tensorboard     
            tb_writer.add_scalars('loss', {'training': loss.item()}, initial_step)
            initial_step += 1
            if i % 1 == 0:
                print(f'epoch: {epoch} iter: {i} training loss: {loss.item()}')


        if epoch % cfg.model.valid_interval == 0:
            valid_loss = compute_valid_loss(model,est, autoregr, test_loader, criterion, cfg)
            tb_writer.add_scalars('loss', {'validation': valid_loss}, initial_step)
            tb_writer.add_figure('plot_train', visualize_traj(model, est, autoregr, data, cfg), initial_step)
            tb_writer.add_figure('plot_validate', visualize_traj(model, est, autoregr, next(iter(test_loader)), cfg), initial_step)
            print(f'epoch: {epoch} validation loss: {valid_loss}')


            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"model saved to {model_path}")

                if cfg.estimator.name != 'GT':  
                    est_path = Path(tb_writer.get_logdir())/f'est_{cfg.estimator.name}.pth'
                    torch.save(est.state_dict(), est_path)
                    print(f"est saved to {est_path}")
    

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    train_loop(cfg) 

if __name__ == '__main__':
    main()
