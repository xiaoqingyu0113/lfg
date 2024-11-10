import sys

from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnn import MNN, autoregr_MNN, AeroModel, BounceModel
from lfg.model_traj.puremlp import PureMLP, autoregr_PureMLP
from lfg.model_traj.phytune import PhyTune, autoregr_PhyTune
from lfg.model_traj.mlp import MLP, autoregr_MLP
from lfg.model_traj.skip import Skip, autoregr_Skip
from lfg.model_traj.lstm import LSTM, autoregr_LSTM


from lfg.estimator import OptimLayer
from draw_util import draw_util


# same seed in training


def compute_valid_loss(cfg):

    logpaths = {'LSTM+Aug.': 'logdir/traj_train/LSTM/pos/real/OptimLayer/run09',
                'Diffusion+Aug.': 'logdir/traj_train/Diffusion',
                'PhyTune+Aug.': 'logdir/traj_train/PhyTune/pos/real/OptimLayer/run19',
                'MLP+Aug.': 'logdir/traj_train/PureMLP/pos/real/OptimLayer/run03',
                'MLP+GS (ours)':'logdir/traj_train/MLP/pos/real/OptimLayer/run02',
                'Skip+GS. (ours)': 'logdir/traj_train/Skip/pos/real/OptimLayer/run00',
                'MNN+GS (ours)': 'logdir/traj_train/MNN/pos/real/OptimLayer/run44',
                } 
    
    # load mnn
    
   
    mnn = PhyTune()
    autoregr = autoregr_PhyTune
    mnn.load_state_dict(torch.load('logdir/traj_train/PhyTune/pos/real/OptimLayer/run19/model_PhyTune.pth'))

    
    mnn.eval()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/PhyTune/pos/real/OptimLayer/run19/est_OptimLayer.pth'))
    # mnn_est.load_state_dict(torch.load('logdir/traj_train/MLP/pos/real/OptimLayer/run02/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn
    
    # dataloader
    np.random.seed(42)
    torch.manual_seed(42)
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
    loss_fn = nn.L1Loss(reduction='none')

    total_loss = []
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            pN_est = autoregr(data, mnn, mnn_est, cfg)
        pN_gt = data[:, :,2:5]
        loss = loss_fn(pN_est, pN_gt)
        total_loss.append(loss)
    total_loss = torch.cat(total_loss, dim=0)

    print(f'Validation Loss: {total_loss.mean().item():.4f}')
    #std
    print(f'Validation Loss std: {total_loss.std().item():.4f}')


def draw_validation_loss_bar(cfg):
    loss = {'LSTM+Aug.': {'mean': 0.1948, 'std': 0.2218},
            'Diffusion+Aug.': {'mean': 0.0710, 'std': 0.0875},
            'A-Tune+Aug.': {'mean': 0.0513, 'std': 0.0664},
            'MLP+Aug.': {'mean': 0.0467, 'std': 0.0617},
            'MLP+GS': {'mean': 0.0322, 'std': 0.0496},
            'Skip+GS.': {'mean': 0.0304, 'std': 0.0445},
            'MNN+GS (ours)': {'mean': 0.0253, 'std': 0.030},
            }
    models = list(loss.keys())
    means = [loss[model]['mean'] for model in models]
    stds = [loss[model]['std'] for model in models]

    # Setting up the visual style using seaborn
    

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    # barplot = ax.barplot(x=models, y=means, palette=sns.color_palette("coolwarm", n_colors=len(models)))
    barplot = plt.bar(models, means, yerr=stds, capsize=5, edgecolor='black')
    #minor grid
    plt.grid(axis='y', linestyle='--', alpha=0.9)
    # minor ticks
    plt.minorticks_on()
    # minor grid
    plt.grid(axis='y', which='minor', linestyle='--', alpha=0.5)

    # error bars
    # plt.errorbar(models, means, yerr=stds, fmt='none', c='black', capsize=5)

    # Adding titles and labels
    # plt.title('Validation Loss by Model (m)', fontsize=18)
    plt.ylabel('RMSE (m)', fontsize=22)
    plt.xlabel('Model', fontsize=22)
    plt.xticks(rotation=45)  # Rotate model names for better readability

    # increase the font of the ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # Annotating the bars with the loss values
    # for i, v in enumerate(means):
    #     barplot.text(i, v + 0.005, f'{v:.4f}', color='black', ha='center', fontsize=16)

    # Display the plot
    plt.tight_layout()
    plt.show()

def validate_3d_plot(cfg):
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/model_MNN.pth'))
    mnn.eval()
    model = AeroModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in MNN: {total_params}')
    model = BounceModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in MNN: {total_params}')
    raise

    mnn.compile()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn
    
    # dataloader
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)

    data = next(iter(test_loader))
    data = next(iter(test_loader))

    
    pN_gt = data[2:3, :,2:5]
    
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _ in range(20):
    # ax.scatter(pN_gt[:, 40:, 0].cpu().numpy(), pN_gt[:, 40:, 1].cpu().numpy(), pN_gt[:, 40:, 2].cpu().numpy(), color='orange', s=3)
    # ax.scatter(pN_gt[:, :40, 0].cpu().numpy(), pN_gt[:, :40, 1].cpu().numpy(), pN_gt[:, :40, 2].cpu().numpy(), color='green', s=3)
        with torch.no_grad():
            pN_est = autoregr_MNN(data, mnn, mnn_est, cfg)
    ax.scatter(pN_est[2:3, :, 0].cpu().numpy(), pN_est[2:3, :, 1].cpu().numpy(), pN_est[2:3, :, 2].cpu().numpy(), color='orange', s=3)
    draw_util.set_axes_equal(ax)
    draw_util.draw_pinpong_table_outline(ax)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color (RGBA format)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
    # no ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=15, azim=41)
    plt.show()

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    # compute_valid_loss(cfg)
    draw_validation_loss_bar(cfg)
    # validate_3d_plot(cfg)


if __name__ == '__main__':
    main()