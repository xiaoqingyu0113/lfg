import sys

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
from lfg.estimator import OptimLayer
from draw_util import draw_util
import seaborn as sns

# same seed in training
np.random.seed(42)
torch.manual_seed(42)

def compute_valid_loss(cfg):
    # load mnn
    
   
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/model_MNN.pth'))
    
    # mnn = MLP()
    # mnn.load_state_dict(torch.load('logdir/traj_train/MLP/pos/real/OptimLayer/run02/model_MLP.pth'))
    
    mnn.eval()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/est_OptimLayer.pth'))
    # mnn_est.load_state_dict(torch.load('logdir/traj_train/MLP/pos/real/OptimLayer/run02/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn
    
    # dataloader
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
    loss_fn = nn.L1Loss()

    total_loss = 0
    loss_num = 0
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            pN_est = autoregr_MNN(data, mnn, mnn_est, cfg)
        pN_gt = data[:, :,2:5]
        loss = loss_fn(pN_est, pN_gt)
        total_loss += loss.item()* data.shape[0]
        loss_num += data.shape[0]
    print(total_loss/loss_num)


def draw_validation_loss_bar(cfg):
    loss = {'LSTM+Aug.': {'mean': 0.1948, 'std': None},
            'Diffusion+Aug.': {'mean': 0.0710, 'std': None},
            'A-Tune+Aug.': {'mean': 0.0513, 'std': None},
            'MLP+Aug.': {'mean': 0.0467, 'std': None},
            'MLP+GS (ours)': {'mean': 0.0322, 'std': None},
            'Skip+GS. (ours)': {'mean': 0.0304, 'std': None},
            'MNN+GS (ours)': {'mean': 0.0253, 'std': None},
            }
    models = list(loss.keys())
    means = [loss[model]['mean'] for model in models]

    # Setting up the visual style using seaborn
    sns.set(style="whitegrid", palette="muted")

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=models, y=means, palette=sns.color_palette("coolwarm", n_colors=len(models)))

    # Adding titles and labels
    # plt.title('Validation Loss by Model (m)', fontsize=18)
    plt.ylabel('Mean Loss (m)', fontsize=18)
    plt.xlabel('Model', fontsize=18)
    plt.xticks(rotation=45)  # Rotate model names for better readability

    # increase the font of the ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Annotating the bars with the loss values
    for i, v in enumerate(means):
        barplot.text(i, v + 0.005, f'{v:.4f}', color='black', ha='center', fontsize=16)

    # Display the plot
    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    # compute_valid_loss(cfg)
    draw_validation_loss_bar(cfg)


if __name__ == '__main__':
    main()