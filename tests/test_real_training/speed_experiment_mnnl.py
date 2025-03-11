import sys
import os
# os.environ["NUMBA_DEBUG"] = "1"
import time
import numba.typed
from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
import torch.autograd.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch_tensorrt


from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnnl import MNNL, autoregr_MNNL, AeroModel, BounceModel



from lfg.estimator import OptimLayer
from draw_util import draw_util
import numba
import logging

from lfg.derive_mnnl import gs, dgs, aero_forward, aero_jacobian, _aero_forward

DTYPE = np.float64
DTYPE_TORCH = torch.float64

    


def get_original_model(compile=True):
    mnn = MNNL(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNNL/pos/real_tennis/OptimLayer/run00/model_MNNL.pth'))
    mnn.eval()
    if compile:
        mnn.compile()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNNL/pos/real_tennis/OptimLayer/run00/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn
    return mnn, mnn_est, autoregr_MNNL




def get_np_params():
    mnn = MNNL(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNNL/pos/real_tennis/OptimLayer/run00/model_MNNL.pth'))
    
    aero_model = mnn.aero_layer
    bounce_model = mnn.bc_layer

    aero_params = {}
    bounce_params = {}
    for name, param in aero_model.named_parameters():
        aero_params[name] = param.cpu().detach().numpy().astype(DTYPE)
        print(name, aero_params[name].shape)
    
    for name, param in bounce_model.named_parameters():
        bounce_params[name] = param.cpu().detach().numpy().astype(DTYPE)
        print(name, bounce_params[name].shape)

    return aero_params, bounce_params


def gs_torch(v,w):
    v_orthonormal = v / (torch.norm(v) + 1e-6)
    proj = torch.dot(w, v_orthonormal) * v_orthonormal
    w_orthogonal = w - proj
    w_orthonormal = w_orthogonal / (torch.norm(w_orthogonal) + 1e-6)
    u_orthonormal = torch.linalg.cross(v_orthonormal, w_orthonormal)

    R = torch.stack((v_orthonormal, w_orthonormal, u_orthonormal), dim=-1)

    v_local = R.T@v
    w_local = R.T@ w

    return R, v_local, w_local





global_aero_params, _ = get_np_params()
global_recode_weight = global_aero_params['recode.weight']
global_recode_bias = global_aero_params['recode.bias']
global_layer1_weight = global_aero_params['layer1.0.weight']
global_layer1_bias = global_aero_params['layer1.0.bias']
global_layer2_weight = global_aero_params['layer2.0.weight']
global_layer2_bias = global_aero_params['layer2.0.bias']

global_dec_0_weight = global_aero_params['dec.0.weight']
global_dec_0_bias = global_aero_params['dec.0.bias']
global_bias = global_aero_params['bias']

    


def forward_twin(v,w):
    return _forward_twin(global_recode_weight, global_recode_bias,
                            global_layer1_weight, global_layer1_bias,
                            global_layer2_weight, global_layer2_bias,
                            global_dec_0_weight, global_dec_0_bias,
                            global_bias, v, w)

def _forward_twin(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, dec_0_weight, dec_0_bias, bias, v, w):
    
    ## set up the parameters
    recode_weight = torch.tensor(recode_weight)
    recode_bias = torch.tensor(recode_bias)
    layer1_weight = torch.tensor(layer1_weight)
    layer1_bias = torch.tensor(layer1_bias)

    layer2_weight = torch.tensor(layer2_weight)
    layer2_bias = torch.tensor(layer2_bias)
    dec_0_bias = torch.tensor(dec_0_bias)
    dec_0_weight = torch.tensor(dec_0_weight)
    bias = torch.tensor(bias)

    # forward
    w = w @ recode_weight.T + recode_bias

    R, v_local, w_local = gs_torch(v, w)     


    # R, v,w local passed the test
    # ---------------------------------
    relu = nn.LeakyReLU()
    feat = torch.stack([v_local[0], w_local[0], w_local[1]])  # Use stack instead of creating a new tensor
    h1 =  feat @ layer1_weight.T + layer1_bias
    h1m = relu(h1)
    # print('twin h1m', h1m)   

    
    # h1m passed the test
    # ---------------------------------

    h2 = h1m @ layer2_weight.T + layer2_bias
    h2m = relu(h2)
    h2mul = h2m * h1m + h1m



    # h2mul passed the test
    # ---------------------------------
    y1 = h2mul @ dec_0_weight.T + dec_0_bias

    y3 = R@y1+ bias.reshape(-1)
                           
    return y3




@numba.jit(nopython=True)
def phy(v,w):
    return np.cross(w,v)*0.1 + np.array([0.0, 0.0, -9.81])  - 0.5 * v * np.linalg.norm(v)


class TestCases:
    def __init__(self):
        pass
    @staticmethod
    def test_aero_forward():
        num_loop = 1

        mnn, mnn_est, autoregr_MNN = get_original_model(compile=False)
        aero_params, _ = get_np_params()

        v = torch.rand(1,3).cuda()*30
        w = torch.rand(1,3).cuda()*10

        torch_time = time.time()
        for _ in range(num_loop):
            y_torch = mnn.aero_layer(v, w)
        print('Torch time:', time.time() - torch_time)
        y_torch = y_torch.cpu().detach().numpy()

        y_twin = forward_twin(v.reshape(-1).cpu().double(),w.reshape(-1).cpu().double())

        v = v.reshape(-1).cpu().detach().numpy().astype(DTYPE)
        w =  w.reshape(-1).cpu().detach().numpy().astype(DTYPE)

        y_global = aero_forward(v,w)
        global_jitted_time = time.time()
        for _ in range(num_loop):
            y_global = aero_forward(v,w)
        print('Global jitted time:', time.time() - global_jitted_time)


        print('Twin:', y_twin)
        print('Torch:', y_torch)
        print('Global Jitted:', y_global)

    @staticmethod
    def test_jacobian():
        mnn, mnn_est, autoregr_MNN = get_original_model(compile=False)

        aero_model = mnn.aero_layer
        
        v = torch.rand(3).cuda()*10
        w = torch.rand(3).cuda()*10
        input_tensor = (v, w)
        forward_func = lambda x, y: aero_model(x[None, :], y[None, :])

        # Compute the Jacobian
        jacobian_torch = F.jacobian(forward_func, input_tensor)
        torch_time = -time.time()
        for _ in range(100):
            jacobian_torch = F.jacobian(forward_func, input_tensor)
        torch_time += time.time()
        print('Torch time:', torch_time/100)

        v_np = v.cpu().numpy().astype(DTYPE)
        w_np = w.cpu().numpy().astype(DTYPE)
        comp_time = -time.time()
        jacobian_np = aero_jacobian(v_np, w_np)
        comp_time += time.time()
        print('Compiled model loading time:', comp_time)
        np_time = -time.time()
        for _ in range(100):
            jacobian_np = aero_jacobian(v_np, w_np)
        np_time += time.time()
        print('Compiled model time:', np_time/100)

        ## accuracy test
        v = v.cpu()
        w = w.cpu()
        v_np = v.cpu().numpy().astype(DTYPE)
        w_np = w.cpu().numpy().astype(DTYPE)
        jacobian_twin = F.jacobian(lambda x: forward_twin(x[:3], x[3:]), torch.cat([v, w], dim=0).to(DTYPE_TORCH),)

        # jacobian_np = jacobian_jitted(v_np, w_np)
        jacobian_np = aero_jacobian(v_np, w_np)
        jacobian_twin = jacobian_twin.cpu().detach().numpy()
        error = np.abs(jacobian_twin - jacobian_np)
        idx= np.argmax(error)
        r_idx, c_idx = np.unravel_index(idx, error.shape)  # Convert to 2D index
        max_error = error.max()
        print('Torch:')
        print(jacobian_torch)
        print('Twin:')
        print(jacobian_twin)
        print('Numpy:')
        print(jacobian_np)
        print(jacobian_twin[r_idx, c_idx], jacobian_np[r_idx, c_idx], f'\nError: at ({r_idx}, {c_idx}) = {max_error}',)

    @staticmethod
    def test_gs_gradient():

     
        def _gs_torch(x):
            v= x[:3]
            w = x[3:]
            v_orthonormal = v / (torch.norm(v) + 1e-6)
            proj = torch.dot(w, v_orthonormal) * v_orthonormal
            w_orthogonal = w - proj

            w_orthogonal2 = w_orthogonal / (torch.norm(w_orthogonal) + 1e-6)
            u_orthonormal = torch.cross(v_orthonormal, w_orthogonal2)

            R = torch.stack((v_orthonormal, w_orthogonal2, u_orthonormal), dim=-1)

        
            v_local = R.T@v
            w_local = R.T@ w
            R, v_local, w_local = gs_torch(x[:3], x[3:])
            return torch.cat([R.reshape(-1), v_local, w_local], dim=0)
        
        
        
        x_cuda = torch.rand(6).cuda()   
        dgs_torch = F.jacobian(_gs_torch, x_cuda)
        torch_time = -time.time()
        for  _ in range(100):
            dgs_torch = F.jacobian(_gs_torch, x_cuda)
        torch_time += time.time()

        print('Torch time:', torch_time)

        x = x_cuda.cpu().detach().numpy().astype(DTYPE)
        dgs_np = dgs(x)
        np_time = -time.time()
        for  i in range(100):
            dgs_np = dgs(x)
        np_time += time.time()
        print('compiled version time:', np_time)


        ## accuracy test
        dgs_torch = F.jacobian(_gs_torch, torch.tensor(x))
        dgs_torch = dgs_torch.cpu().numpy()

        dgs_np = dgs(x)

        error = np.abs(dgs_torch - dgs_np)
        idx= np.argmax(error)
        r_idx, c_idx = np.unravel_index(idx, error.shape)  # Convert to 2D index
        max_error = error.max()

        print('Torch:\n', dgs_torch[9:,:])
        print('Numpy:\n', dgs_np[9:,:])
        print(f'Max Error is {max_error} at ({r_idx}, {c_idx}), where the values are {dgs_torch[r_idx, c_idx], dgs_np[r_idx, c_idx]}')






@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    # TestCases.test_aero_forward()
    TestCases.test_jacobian()
    # TestCases.test_gs_gradient()

    


if __name__ == '__main__':
    main()