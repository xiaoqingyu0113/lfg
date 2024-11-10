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
from lfg.model_traj.mlp import MLP, autoregr_MLP
from lfg.model_traj.puremlp import PureMLP
from lfg.model_traj.phytune import PhyTune, autoregr_PhyTune
from lfg.estimator import OptimLayer
from draw_util import draw_util
import time
np.random.seed(42)
torch.manual_seed(42)

def show_roto_translational():
    # train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)

    # load mnn
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run45/model_MNN.pth'))
    mnn.eval()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=20)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run45/est_OptimLayer.pth'))
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
            t_start = time.time()
            for i in range(1, 150):
                b0 = p0[:,:,2:3]
                p0 = p0 + v0* dt
                v0, w0 = mnn(b0, v0, w0, dt)
                pN_est.append(p0)
            pN_est = torch.cat(pN_est, dim=1)
            print('Time taken:', time.time() - t_start)
        pN_est = pN_est.cpu().numpy()
        ax.plot(pN_est[0, :, 0], pN_est[0, :, 1], pN_est[0, :, 2], label='MNN (ours)', linewidth=2)

    for th in np.linspace(0, 2*np.pi, 20):
        p0 = torch.tensor([[[-2.5, -3.0, 0.100]]], dtype=torch.float32).to('cuda')
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
            for i in range(1, 150):
                b0 = p0[:,:,2:3]
                p0 = p0 + v0* dt
                v0, w0 = mnn(b0, v0, w0, dt)
                pN_est.append(p0)
            pN_est = torch.cat(pN_est, dim=1)
        pN_est = pN_est.cpu().numpy()

        ax.plot(pN_est[0, :, 0], pN_est[0, :, 1], pN_est[0, :, 2], label='MNN (ours)', linewidth=2)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color (RGBA format)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    draw_util.set_axes_equal(ax)

    # increase tick size and axis label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.zaxis.label.set_size(15)
    
    plt.show()


def compute_trajectory(model, p0, v0, w0, dt, th):
    
    # rotate v0, w0 based on th
    R = torch.tensor([[np.cos(th), -np.sin(th), 0],
                        [np.sin(th), np.cos(th), 0],
                        [0, 0, 1]], dtype=torch.float32).to('cuda')
    v0 = torch.matmul(v0, R)
    w0 = torch.matmul(w0, R)
    t_start = time.time()   
    with torch.no_grad():
        pN_est = [p0]
        for i in range(1, 100):
            b0 = p0[:,:,2:3]
            p0 = p0 + v0* dt
            v0, w0 = model(b0, v0, w0, dt)
            pN_est.append(p0)
        pN_est = torch.cat(pN_est, dim=1)
        print('Time taken:', time.time() - t_start)
    return pN_est, R

def compute_roto_trans_error():

    model_name = 'MLP+Aug'

    if model_name == 'MLP+GS':
        model = MLP()
        model.load_state_dict(torch.load('logdir/traj_train/MLP/pos/real/OptimLayer/run02/model_MLP.pth'))
    elif model_name == 'MLP+Aug':
        model = PureMLP()
        model.load_state_dict(torch.load('logdir/traj_train/PureMLP/pos/real/OptimLayer/run03/model_PureMLP.pth'))
    model.eval()
    model.to('cuda')
    

    p0 = torch.tensor([[[0.5, -3.0, 0.100]]], dtype=torch.float32).to('cuda')
    v0 = torch.tensor([[[2.0, 0.0, 1.0]]], dtype=torch.float32).to('cuda')
    w0 = torch.tensor([[[-5.0, 3.0, 0.0]]], dtype=torch.float32).to('cuda')
    dt = torch.tensor([[[0.01]]], dtype=torch.float32).to('cuda')

    loss = nn.MSELoss()

    model.compile()
    pN_est, R = compute_trajectory(model, p0, v0, w0, dt, 0)

    total_loss = []
    fig = plt.figure()  
    ax = fig.add_subplot(projection='3d')   
    for th in np.linspace(0, 2*np.pi, 30):
        pN_est_2, R = compute_trajectory(model, p0, v0, w0, dt, th)
        pN_est_2_np = pN_est_2.cpu().numpy()

        # rotate pN_est_2 back to original frame
        pN_est_2 = torch.matmul(pN_est_2 - pN_est_2[:,0:1,:], R.transpose(0, 1)) + pN_est_2[:,0:1,:]

        # compute error
        err = loss(pN_est, pN_est_2)

        total_loss.append(err.item())

        print(model_name, np.sqrt(np.mean(total_loss)), np.std(total_loss))
        
        pN_est_np= pN_est.cpu().numpy()
        

        # scatter size is 1, label is 'o'
        ax.scatter(pN_est_np[0, :, 0], pN_est_np[0, :, 1], pN_est_np[0, :, 2], label='o', s=2)
        ax.scatter(pN_est_2_np[0, :, 0], pN_est_2_np[0, :, 1], pN_est_2_np[0, :, 2], label='.', s=1)

    draw_util.set_axes_equal(ax)
    plt.show()

if __name__ == '__main__':
    # show_roto_translational()
    compute_roto_trans_error()