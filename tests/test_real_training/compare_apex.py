import sys
sys.path.append('tests/test_real_training')
import os
import json

from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnn import MNN, autoregr_MNN
from lfg.model_traj.phytune import PhyTune, autoregr_PhyTune
from lfg.model_traj.mlp import MLP, autoregr_MLP
from lfg.model_traj.puremlp import PureMLP, autoregr_PureMLP
from lfg.model_traj.skip import Skip, autoregr_Skip
from lfg.model_traj.lstm import LSTM, autoregr_LSTM


from lfg.estimator import OptimLayer
from draw_util import draw_util



def find_bounce_points(data: np.ndarray, z0:float= 0.010):
    '''
    input: data: (batch, seq_len, ...)
    output:  (batch, )
    '''
    window_size = 13
    region_size = window_size // 2
    th = 0.030

    batch, seq_len, _ = data.shape
    bounce_indices = [[] for _ in range(batch)]

    for b, datum in enumerate(data):
        for i in range(region_size,  seq_len - region_size):
            min_z = np.min(datum[i-region_size:i+region_size, 4])
            if np.isclose(datum[i, 4], min_z) and min_z < z0 + th:
                bounce_indices[b].append(i)


    return bounce_indices



def peek(cfg):
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)

    # load mnn
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run28/model_MNN.pth'))
    mnn.eval()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=20)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run28/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn

    # load phytune
    phy = PhyTune()
    phy.load_state_dict(torch.load('logdir/traj_train/PhyTune/pos/real/OptimLayer/run19/model_PhyTune.pth'))
    phy.eval()
    phy.to('cuda')

    phy_est = OptimLayer(phy, size=80, allow_grad=False, damping=0.1, max_iterations=20)
    phy_est.load_state_dict(torch.load('logdir/traj_train/PhyTune/pos/real/OptimLayer/run19/est_OptimLayer.pth'))
    phy_est.eval()
    phy_est.to('cuda')
    phy_est.model = phy

    iter_loader = iter(test_loader)
    data = next(iter_loader)
    data = next(iter_loader)
    data = next(iter_loader)

    s_idx = 3
    e_idx = 4
    data = data[s_idx:e_idx] # choose some samples
    with torch.no_grad():
        pN_mnn = autoregr_MNN(data, mnn, mnn_est, cfg)
        pN_phy = autoregr_PhyTune(data, phy, phy_est, cfg)
    # pN_mnn = pN_mnn.cpu().numpy()
    # pN_phy = pN_phy.cpu().numpy()

    data = data.cpu().numpy()
    bounce_indices = find_bounce_points(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        if 'pN_mnn' in locals():
            ax.plot(pN_mnn[i, :, 0], pN_mnn[i, :, 1], pN_mnn[i, :, 2], label='MNN (ours)')
        if 'pN_phy' in locals():
            ax.plot(pN_phy[i, :, 0], pN_phy[i, :, 1], pN_phy[i, :, 2], label='PhyTune')
        ax.plot(data[i, :, 2], data[i, :, 3], data[i, :, 4], label='GT')
        for idx in bounce_indices[i]:
            ax.scatter(data[i, idx, 2], data[i, idx, 3], data[i, idx, 4], c='r', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    draw_util.set_axes_equal(ax)
    draw_util.draw_pinpong_table_outline(ax)


    plt.show()

def loop_apex_loss(cfg):
    logpaths = {'LSTM+Aug.': 'logdir/traj_train/LSTM/pos/real/OptimLayer/run09',
                'Diffusion+Aug.': 'logdir/traj_train/Diffusion',
                'A-Tune+Aug.': 'logdir/traj_train/PhyTune/pos/real/OptimLayer/run19',
                'MLP+Aug.': 'logdir/traj_train/PureMLP/pos/real/OptimLayer/run03',
                'MLP+GS (ours)':'logdir/traj_train/MLP/pos/real/OptimLayer/run02',
                'Skip+GS. (ours)': 'logdir/traj_train/Skip/pos/real/OptimLayer/run00',
                'MNN+GS (ours)': 'logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44',
                } 
    np.random.seed(42)
    torch.manual_seed(42)
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
    
    loop_name = list(logpaths.keys())
    looped_apex_loss = {}
    looped_bounce_loss = {}
    looped_between_loss = {}

    for model_name in loop_name:
        # diffusion benchmark is not included
        if "diffusion" in model_name.lower():
            continue
        logdir = logpaths[model_name]
        pth_files = [f for f in os.listdir(logdir) if f.endswith('.pth')]
        for pth in pth_files:
            NN_name = pth.split('_')[1].split('.')[0]
            if 'model' in pth:
                model = globals()[NN_name]()
                model.load_state_dict(torch.load(os.path.join(logdir, pth)))
                model.cuda().eval()
                auto_regr = globals()[f'autoregr_{NN_name}']
        for pth in pth_files:
            if 'est' in pth:
                estimator = OptimLayer(model, size=80, allow_grad=False, damping=0.1, max_iterations=30)
                estimator.load_state_dict(torch.load(os.path.join(logdir, pth)))
                estimator.cuda().eval()
                estimator.model = model
        
        apexloss, bounce_loss, between_loss = apex_loss(model, estimator, auto_regr, test_loader)
        looped_apex_loss[model_name] = apexloss
        looped_bounce_loss[model_name] = bounce_loss
        looped_between_loss[model_name] = between_loss

        #save apex in json
        def json_serializer(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(f'data/real/compare/looped_apex_loss.json', 'w') as f:
            json.dump(looped_apex_loss, f, indent=4, default=json_serializer)
        
        with open(f'data/real/compare/looped_bounce_loss.json', 'w') as f:
            json.dump(looped_bounce_loss, f, indent=4, default=json_serializer)

        with open(f'data/real/compare/looped_between_loss.json', 'w') as f:
            json.dump(looped_between_loss, f, indent=4, default=json_serializer)


def apex_loss(model, estimator, auto_regr, loader):

    
    # numpy rmse loss where x, y are of shape ( 3)
    loss_fn = lambda x, y: np.sqrt(np.mean((x - y)**2))
    # numpy rmse loss where x, y are of shape (seq, 3)
    # seq_loss_fn = lambda x, y: np.sqrt(np.mean((x - y)**2, axis=1))

    apex_loss = {'Before#1':[], 'Between#1#2':[], 'Between#2#3':[]}
    between_loss =  {'Before#1':[], 'Between#1#2':[], 'Between#2#3':[]}
    bounce_loss = {1:[], 2:[], 3:[]}

    for data in loader:
        print('Processing new batch ...')
        with torch.no_grad():
            pN_MNN = auto_regr(data, model, estimator, None)
        data= data.cpu().numpy()
        pN_MNN = pN_MNN.cpu().numpy()
        bounce_indices = find_bounce_points(data)

        for batch in range(data.shape[0]):
            if len(bounce_indices[batch]) >0 :
                apex_idx = bounce_indices[batch][0]//2
                apex_time = data[batch, apex_idx, 1] - data[batch, 0, 1]
                aloss = loss_fn(data[batch, apex_idx, 2:5], pN_MNN[batch, apex_idx, :3])
                apex_loss['Before#1'].append([apex_time, aloss])

                bounce_id = bounce_indices[batch][0]
                bt_loss = loss_fn(data[batch, :bounce_id, 2:5], pN_MNN[batch, :bounce_id, :3])
                between_loss['Before#1'].append([apex_time, bt_loss])
            if len(bounce_indices[batch]) >1 :
                apex_idx = (bounce_indices[batch][0] + bounce_indices[batch][1])//2
                apex_time = data[batch, apex_idx, 1] - data[batch, 0, 1]
                aloss = loss_fn(data[batch, apex_idx, 2:5], pN_MNN[batch, apex_idx, :3])
                apex_loss['Between#1#2'].append([apex_time, aloss])

                bounce_id = bounce_indices[batch][1]
                bt_loss = loss_fn(data[batch, bounce_indices[batch][0]:bounce_id, 2:5], pN_MNN[batch, bounce_indices[batch][0]:bounce_id, :3])
                between_loss['Between#1#2'].append([apex_time, bt_loss])

            if len(bounce_indices[batch]) >2 :
                apex_idx = (bounce_indices[batch][1] + bounce_indices[batch][2])//2
                apex_time = data[batch, apex_idx, 1] - data[batch, 0, 1]
                aloss = loss_fn(data[batch, apex_idx, 2:5], pN_MNN[batch, apex_idx, :3])
                apex_loss['Between#2#3'].append([apex_time, aloss])

                bounce_id = bounce_indices[batch][2]
                bt_loss = loss_fn(data[batch, bounce_indices[batch][1]:bounce_id, 2:5], pN_MNN[batch, bounce_indices[batch][1]:bounce_id, :3])
                between_loss['Between#2#3'].append([apex_time, bt_loss])

            for number_bounce, bounce_id in enumerate(bounce_indices[batch]):
                if number_bounce < 3:
                    bloss = loss_fn(data[batch, bounce_id, 2:5], pN_MNN[batch, bounce_id, :3])
                    bounce_loss[number_bounce+1].append(bloss)

                    
                
    return apex_loss, bounce_loss, between_loss

def print_apex_loss(apex_loss):     
    with open('data/real/compare/looped_apex_loss.json', 'r') as f:
        looped_apex_loss = json.load(f)
    with open('data/real/compare/looped_bounce_loss.json', 'r') as f:
        looped_bounce_loss = json.load(f)
    with open('data/real/compare/looped_between_loss.json', 'r') as f:
        looped_between_loss = json.load(f)
    # fig, axes = plt.subplots(1, 3, figsize=(9, 3)) 
    for model_name, apex_loss in looped_apex_loss.items():
        for idx, (k, v) in enumerate(apex_loss.items()):
            # ax = axes[idx]
            values = np.array(v)
            mean_loss = np.mean(values[:,1])
            std_loss = np.std(values[:,1])
            print(f'Apex loss - {model_name}, {k} : mean: {mean_loss}, std: {std_loss}')

    for model_name, bloss in looped_bounce_loss.items():
        for idx, (k, v) in enumerate(bloss.items()):
            # ax = axes[idx]
            values = np.array(v)
            mean_loss = np.mean(values)
            std_loss = np.std(values)
            print(f'Bounce loss - {model_name}, {k} : mean: {mean_loss}, std: {std_loss}')

    for model_name, bloss in looped_between_loss.items():
        for idx, (k, v) in enumerate(bloss.items()):
            # ax = axes[idx]
            values = np.array(v)
            print(values.shape)
            mean_loss = np.mean(values[:,1])
            std_loss = np.std(values[:,1])
            print(f'Between loss - {model_name}, {k} : mean: {mean_loss}, std: {std_loss}')

    plt.show()

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    # apex_loss(cfg) 
    loop_apex_loss(cfg)
    print_apex_loss(cfg)
    # peek(cfg)


if __name__ == "__main__":
    main()