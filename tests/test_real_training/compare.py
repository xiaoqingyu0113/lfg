import sys
sys.path.append('tests/test_real_training')

from omegaconf import OmegaConf
import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np


from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnn import MNN, autoregr_MNN
from lfg.model_traj.phytune import PhyTune, autoregr_PhyTune
from lfg.estimator import OptimLayer
from draw_util import draw_util



def run(cfg):
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
    model = MNN(z0=0.010)
    model.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run28/model_MNN.pth'))
    model.eval()
    model.to('cuda')

    estimator = OptimLayer(model, size=80, allow_grad=False, damping=0.1, max_iterations=2)
    estimator.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run28/est_OptimLayer.pth'))
    estimator.eval()
    estimator.to('cuda')

    model = model
    estimator.model = model
    # estimator = torch.compile(estimator)

    data = next(iter(test_loader))

    data = data[:8] # first 8 samples
    with torch.no_grad():
        pN_MNN = autoregr_MNN(data, model, estimator, cfg)

    # show in figure
    pN_MNN = pN_MNN.cpu().numpy()
    data = data.cpu().numpy()

    # compute rmse for each sample
    rmse = np.sqrt(np.mean((pN_MNN - data[:,:,2:5])**2, axis=(1,2)))
    print(f"RMSE: {rmse}")

    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(pN_MNN.shape[0]):
        ax.plot(pN_MNN[i, :, 0], pN_MNN[i, :, 1], pN_MNN[i, :, 2], label='MNN')
        ax.plot(data[i, :, 2], data[i, :, 3], data[i, :, 4], label='GT')
    ax.legend()
    draw_util.set_axes_equal(ax)
    plt.show()

    
    
    

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    run(cfg) 


if __name__ == "__main__":
    main()