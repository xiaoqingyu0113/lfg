import numpy as np
import torch
import torch.nn as nn
import gtsam
from gtsam.symbol_shorthand import L,V,W,X

from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnn import MNN, autoregr_MNN, AeroModel, BounceModel
from lfg.estimator import OptimLayer
from lfg.derive import PriorFactor3, PositionFactor, VWFactor

from omegaconf import OmegaConf
import hydra
import time

DTYPE = np.float64

def load_mnn(cfg):
    mnn = MNN()
    autoregr = autoregr_MNN
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/model_MNN.pth'))
    mnn.eval()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn
    return mnn, mnn_est, autoregr_MNN



def ivp_gtsam(data):
    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()
    parameters = gtsam.ISAM2Params()
    isam2 = gtsam.ISAM2(parameters)
    initial_estimate = gtsam.Values()
    optim_estimate = None
    start_time = None
    minimum_graph_size = 10 # minimum number of factors to optimize



    # Create noise models

    pPriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.010, 0.010, 0.010], dtype=DTYPE))
    vwNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1], dtype=DTYPE))
    wPriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1], dtype=DTYPE))

    N_seq, N_data = data.shape

    for i in range(N_seq):
        t, p, v, w = data[i, 1], data[i, 2:5], data[i, 5:8], data[i, 8:11]
        
        graph.push_back(PriorFactor3(pPriorNoise, L(i), p))
        initial_estimate.insert(L(i), p)
        
        # add w prior only at the beginning
        if i == 0:
            graph.push_back(PriorFactor3(wPriorNoise, W(i), w))
            initial_estimate.insert(W(i), w)
            initial_estimate.insert(V(i), 1e-3*np.random.rand(3).astype(DTYPE))
        
        # add forward dynamics factor for i > 0
        else:
            t_prev = data[i-1, 1]
            graph.push_back(PositionFactor(pPriorNoise, L(i-1), V(i-1), L(i), t_prev, t))
            graph.push_back(VWFactor(vwNoise, V(i-1), W(i-1), V(i), W(i), t_prev, t))

            if optim_estimate is None:
                initial_estimate.insert(V(i), 1e-3*np.random.rand(3).astype(DTYPE))
                initial_estimate.insert(W(i), 1e-3*np.random.rand(3).astype(DTYPE))
            else:
                v_prev = optim_estimate.atVector(V(i-1))
                w_prev = optim_estimate.atVector(W(i-1))
                initial_estimate.insert(V(i), v_prev)
                initial_estimate.insert(W(i), w_prev)

        if i > minimum_graph_size:
            if not start_time:
                start_time = time.time()
            isam2.update(graph, initial_estimate)
            optim_estimate = isam2.calculateEstimate()
            graph.resize(0)
            initial_estimate.clear()

        if i >= 80:
            print("inference time", (time.time() - start_time)/(80-minimum_graph_size))
            return optim_estimate

    



class TestCases:

    def __init__(self,cfg):
        self.cfg = cfg

    @staticmethod
    def test_mnn(cfg):
        mnn, mnn_est, autoregr = load_mnn(cfg)
        # dataloader
        np.random.seed(42)
        torch.manual_seed(42)
        train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
        
        data = next(iter(train_loader))[0:1, ...]

        est_size= 80
        w0 = data[0:1, 0:1, 8:11]
        with torch.no_grad():
            p0, v0, w0 = mnn_est(data[:,:est_size, 1:5], w0=w0)

        print(f"p0_est: {p0}")
        print(f"p0_data: {data[0, 0, 2:5]}")

        print("v0_est:", v0)
        print("v0_data:", data[0, 0, 5:8])

        print("w0_est:", w0)
        print("w0_data:", data[0, 0, 8:11])

    @staticmethod
    def test_gtsam(cfg):
        mnn, mnn_est, autoregr = load_mnn(cfg)
        # dataloader
        np.random.seed(42)
        torch.manual_seed(42)
        train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
        
        data = next(iter(train_loader))[0:1, ...]
        data = data.to('cpu').numpy()[0]
        
        res = ivp_gtsam(data)

        print(res.atVector(L(0)))
        print(data[0, 2:5])

        print(res.atVector(V(0)))
        print(data[0, 5:8])






@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    # TestCases.test_mnn(cfg)
    TestCases.test_gtsam(cfg)

if __name__ == '__main__':
    main()


