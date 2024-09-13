
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from train_real_traj import RealTrajectoryDataset
import hydra

from lfg.model_traj.mnn import MNN, autoregr_MNN
from lfg.model_traj.phytune import PhyTune, autoregr_PhyTune
from lfg.model_traj.mlp import MLP, autoregr_MLP

from lfg.estimator import OptimLayer
from pycamera import CameraParam


def load_model(model_name):
    logpaths = {'LSTM+Aug.': 'logdir/traj_train/LSTM/pos/real/OptimLayer/run09',
                    'Diffusion+Aug.': 'logdir/traj_train/Diffusion',
                    'A-Tune+Aug.': 'logdir/traj_train/PhyTune/pos/real/OptimLayer/run19',
                    'MLP+Aug.': 'logdir/traj_train/PureMLP/pos/real/OptimLayer/run03',
                    'MLP+GS (ours)':'logdir/traj_train/MLP/pos/real/OptimLayer/run02',
                    'Skip+GS. (ours)': 'logdir/traj_train/Skip/pos/real/OptimLayer/run00',
                    'MNN+GS (ours)': 'logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44',
                    } 
    

    # diffusion benchmark is not included
    if "diffusion" in model_name.lower():
        raise NotImplementedError("Diffusion model is not implemented yet.")
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
    return model, estimator

def read_bg_image(i):
    return plt.imread(f'data/real/bg/camera_{i}.jpg')


def read_camera_param(i):
    cam_ids = {'camera_1': '22276213', 'camera_2': '22276209', 'camera_3': '22276216'} # dont change order
    serial = cam_ids[f'camera_{i}']
    camera_param = CameraParam.from_yaml(f'conf/camera/{serial}_calibration.yaml')
    return camera_param

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def run(cfg):
    np.random.seed(42)
    torch.manual_seed(42)
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
    model, estimator = load_model('MNN+GS (ours)')
    bg_image= read_bg_image(1)
    camera_param = read_camera_param(1)


    # Prepare the figure
    fig = plt.figure(figsize=(10, 8.44))
    ax = fig.add_subplot(111)

    data = next(iter(test_loader))
    with torch.no_grad():
        pN_est = autoregr_MNN(data, model, estimator, cfg)
    pN_est = pN_est.cpu().numpy()
    pN_gt = data[:, :, 2:5].cpu().numpy()
    choices = [1, 2,  30] # pick some good ones for presentation purposes only
    for i in choices:
        pN_gt_i = pN_gt[i]
        pN_est_i = pN_est[i]
        topspin = int(data[i,0,8].cpu().item())
        sidespin = int(data[i,0,9].cpu().item())
        gt_image = camera_param.proj2img(pN_gt_i)
        est_image = camera_param.proj2img(pN_est_i)
        ax.scatter(gt_image[:, 0], gt_image[:, 1], label=f'GT TS={topspin} SS={sidespin}', s=1)
        ax.plot(est_image[:, 0], est_image[:, 1], label='Prediction', linewidth=2)


    data = next(iter(test_loader))
    with torch.no_grad():
        pN_est = autoregr_MNN(data, model, estimator, cfg)
    pN_est = pN_est.cpu().numpy()
    pN_gt = data[:, :, 2:5].cpu().numpy()
    choices =  [2, 3, 5,  9, 21] 
    for i in choices:
        pN_gt_i = pN_gt[i]
        pN_est_i = pN_est[i]
        topspin = int(data[i,0,8].cpu().item())
        sidespin = int(data[i,0,9].cpu().item())
        gt_image = camera_param.proj2img(pN_gt_i)
        est_image = camera_param.proj2img(pN_est_i)
        ax.scatter(gt_image[:, 0], gt_image[:, 1], label=f'GT TS={topspin} SS={sidespin}', s=1)
        ax.plot(est_image[:, 0], est_image[:, 1], label='Prediction', linewidth=2)

    ax.imshow(bg_image)
    # legend lower left
    ax.legend(loc='lower left')
    ax.legend(fontsize=15)

    




    # layout tight, don't show x and y ticks, increase the font in the legend
    plt.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])

    
    plt.show()


if __name__ == '__main__':
    run()  # Call the run function to execute the code