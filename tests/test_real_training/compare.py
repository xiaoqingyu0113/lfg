import sys
sys.path.append('tests/test_real_training')

from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnn import MNN, autoregr_MNN
from lfg.model_traj.phytune import PhyTune, autoregr_PhyTune
from lfg.estimator import OptimLayer
from draw_util import draw_util

np.random.seed(42)
torch.manual_seed(42)

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
    # with torch.no_grad():
    #     pN_mnn = autoregr_MNN(data, mnn, mnn_est, cfg)
    #     pN_phy = autoregr_PhyTune(data, phy, phy_est, cfg)
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

def show_roto_translational(cfg):
    # train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for th in np.linspace(0, 2*np.pi, 20):
        p0 = torch.tensor([[[0.5, -3.0, 0.100]]], dtype=torch.float32).to('cuda')
        v0 = torch.tensor([[[2.0, 0.0, 1.0]]], dtype=torch.float32).to('cuda')
        w0 = torch.tensor([[[5.0, 3.0, 0.0]]], dtype=torch.float32).to('cuda')
        dt = torch.tensor([[[0.01]]], dtype=torch.float32).to('cuda')
        # rotate v0, w0 based on th
        R = torch.tensor([[np.cos(th), -np.sin(th), 0],
                            [np.sin(th), np.cos(th), 0],
                            [0, 0, 1]], dtype=torch.float32).to('cuda')
        v0 = torch.matmul(v0, R)
        w0 = torch.matmul(w0, R)
        with torch.no_grad():
            pN_est = [p0]
            for i in range(1, 50):
                b0 = p0[:,:,2:3]
                p0 = p0 + v0* dt
                v0, w0 = mnn(b0, v0, w0, dt)
                pN_est.append(p0)
            pN_est = torch.cat(pN_est, dim=1)
        pN_est = pN_est.cpu().numpy()
        ax.plot(pN_est[0, :, 0], pN_est[0, :, 1], pN_est[0, :, 2], label='MNN (ours)')

    for th in np.linspace(0, 2*np.pi, 20):
        p0 = torch.tensor([[[-0.5, -3.0, 0.100]]], dtype=torch.float32).to('cuda')
        v0 = torch.tensor([[[2.0, 0.0, 1.0]]], dtype=torch.float32).to('cuda')
        w0 = torch.tensor([[[5.0, 3.0, 0.0]]], dtype=torch.float32).to('cuda')
        dt = torch.tensor([[[0.01]]], dtype=torch.float32).to('cuda')
        # rotate v0, w0 based on th
        R = torch.tensor([[np.cos(th), -np.sin(th), 0],
                            [np.sin(th), np.cos(th), 0],
                            [0, 0, 1]], dtype=torch.float32).to('cuda')
        v0 = torch.matmul(v0, R)
        w0 = torch.matmul(w0, R)
        with torch.no_grad():
            pN_est = [p0]
            for i in range(1, 50):
                b0 = p0[:,:,2:3]
                p0 = p0 + v0* dt
                v0, w0 = mnn(b0, v0, w0, dt)
                pN_est.append(p0)
            pN_est = torch.cat(pN_est, dim=1)
        pN_est = pN_est.cpu().numpy()
        ax.plot(pN_est[0, :, 0], pN_est[0, :, 1], pN_est[0, :, 2], label='MNN (ours)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    draw_util.set_axes_equal(ax)
    plt.show()


def apex_loss(cfg):
    model_name = 'PhyTune'
    loader_name = 'test'

    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)

    loader = train_loader if loader_name == 'train' else test_loader

    if model_name == 'MNN':
        model = MNN(z0=0.010)
        model.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run28/model_MNN.pth'))
        model.eval()
        model.to('cuda')

        estimator = OptimLayer(model, size=80, allow_grad=False, damping=0.1, max_iterations=20)
        estimator.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run28/est_OptimLayer.pth'))
        estimator.eval()
        estimator.to('cuda')

    elif model_name == 'PhyTune':
        model = PhyTune()
        model.load_state_dict(torch.load('logdir/traj_train/PhyTune/pos/real/OptimLayer/run19/model_PhyTune.pth'))
        model.eval()
        model.to('cuda')

        estimator = OptimLayer(model, size=80, allow_grad=False, damping=0.1, max_iterations=20)
        estimator.load_state_dict(torch.load('logdir/traj_train/PhyTune/pos/real/OptimLayer/run19/est_OptimLayer.pth'))
        estimator.eval()
        estimator.to('cuda')
                                         
    

    model = model
    estimator.model = model
    
    # numpy rmse loss where x, y are of shape ( 3)
    loss_fn = lambda x, y: np.sqrt(np.mean((x - y)**2))
    apex_loss = {'Before#1':[], 'Between#1#2':[], 'Between#2#3':[]}
    for data in loader:
        print('Processing new batch ...')
        with torch.no_grad():
            pN_MNN = autoregr_MNN(data, model, estimator, None)
        data= data.cpu().numpy()
        pN_MNN = pN_MNN.cpu().numpy()
        bounce_indices = find_bounce_points(data)

        for batch in range(data.shape[0]):
            if len(bounce_indices[batch]) >0 :
                apex_idx = bounce_indices[batch][0]//2
                apex_time = data[batch, apex_idx, 1] - data[batch, 0, 1]
                aloss = loss_fn(data[batch, apex_idx, 2:5], pN_MNN[batch, apex_idx, :3])
                apex_loss['Before#1'].append([apex_time, aloss])
            if len(bounce_indices[batch]) >1 :
                apex_idx = (bounce_indices[batch][0] + bounce_indices[batch][1])//2
                apex_time = data[batch, apex_idx, 1] - data[batch, 0, 1]
                aloss = loss_fn(data[batch, apex_idx, 2:5], pN_MNN[batch, apex_idx, :3])
                apex_loss['Between#1#2'].append([apex_time, aloss])
            if len(bounce_indices[batch]) >2 :
                apex_idx = (bounce_indices[batch][1] + bounce_indices[batch][2])//2
                apex_time = data[batch, apex_idx, 1] - data[batch, 0, 1]
                aloss = loss_fn(data[batch, apex_idx, 2:5], pN_MNN[batch, apex_idx, :3])
                apex_loss['Between#2#3'].append([apex_time, aloss])

        
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    for k, v in apex_loss.items():
        ax.clear()
        values = np.array(v)
        mean_loss = np.mean(values[:,1])
        std_loss = np.std(values[:,1])
        print(f'{model_name}, {k} : mean: {mean_loss}, std: {std_loss}')

        ax.scatter(values[:, 0], values[:, 1], label=k)
        ax.plot([0, values[:,0].max()], [mean_loss, mean_loss], 'r-', label='Mean')
        ax.plot([0, values[:,0].max()], [mean_loss + std_loss, mean_loss+std_loss], 'r--', label='std+')
        ax.plot([0, values[:,0].max()], [mean_loss - std_loss, mean_loss-std_loss], 'r--', label='std-')

        ax.set_title(f'{model_name}_{loader_name}_Apex Loss')
        ax.set_xlabel(f'Time to Apex {k} (sec)')
        ax.set_ylabel('RMSE Loss (m)')
        if k == 'Before#1':
            ax.set_ylim([0, 0.05])
        elif k == 'Between#1#2':
            ax.set_ylim([0, 0.3])
        elif k == 'Between#2#3':
            ax.set_ylim([0, 0.5])

        fig.savefig(f'{model_name}_{loader_name}_{k}.png')
    

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    apex_loss(cfg) 
    # peek(cfg)
    # show_roto_translational(cfg)


if __name__ == "__main__":
    main()