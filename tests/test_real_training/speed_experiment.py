import sys
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
from lfg.model_traj.mnn import MNN, autoregr_MNN, AeroModel, BounceModel



from lfg.estimator import OptimLayer
from draw_util import draw_util
import numba
import logging


def matrix_speed():
    def numpy_matmul(A,x):
        return np.matmul(A, x)
    
    def torch_matmul(A,x):
        return torch.matmul(A, x)
    N  = 6
    A_np, x_np = np.random.rand(N, N), np.random.rand(N, 1)
    A_torch, x_torch = torch.rand(N,N).cuda(), torch.rand(N,1).cuda()

    start = time.time()
    for _ in range(1000):
        numpy_matmul(A_np, x_np)
    print('Numpy time:', (time.time() - start)/1000)

    start = time.time()
    for _ in range(1000):
        torch_matmul(A_torch, x_torch)
    print('Torch time:', (time.time() - start)/1000)


    


def get_original_model(compile=True):
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/model_MNN.pth'))
    mnn.eval()
    if compile:
        mnn.compile()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/est_OptimLayer.pth'))
    mnn_est.eval()
    mnn_est.to('cuda')
    mnn_est.model = mnn
    return mnn, mnn_est, autoregr_MNN



def get_tensorRT_model():
    mnn, mnn_est, autoregr_MNN = get_original_model(compile=False)
    aero_model = mnn.aero_layer
    bounce_model = mnn.bc_layer
    aero_model = torch.compile(aero_model, backend='tensorrt')
    bounce_model = torch.compile(bounce_model, backend='tensorrt')
    mnn.aero_layer = aero_model
    mnn.bc_layer = bounce_model
    return mnn, mnn_est, autoregr_MNN

def get_np_params():
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/model_MNN.pth'))
    
    aero_model = mnn.aero_layer
    bounce_model = mnn.bc_layer

    aero_params = {}
    bounce_params = {}
    for name, param in aero_model.named_parameters():
        aero_params[name] = param.cpu().detach().numpy().astype(np.float32)
        print(name, aero_params[name].shape)
    
    for name, param in bounce_model.named_parameters():
        bounce_params[name] = param.cpu().detach().numpy().astype(np.float32)
        print(name, bounce_params[name].shape)

    return aero_params, bounce_params

@numba.jit(nopython=True)
def gs(v,w):
    v_orthonormal = v / (np.linalg.norm(v) + np.float32(1e-6))
    proj = np.dot(w, v_orthonormal) * v_orthonormal
    w_orthogonal = w - proj
    w_orthonormal = w_orthogonal / (np.linalg.norm(w_orthogonal) + np.float32(1e-6))
    u_orthonormal = np.cross(v_orthonormal, w_orthonormal)

    R = np.stack((v_orthonormal, w_orthonormal, u_orthonormal), axis=-1)

    v_local = R.T@v
    w_local = R.T@ w

    return R, v_local, w_local

@numba.jit(nopython=True )
def dgs(x):
    v = x[:3]
    w = x[3:]

    v_orthonormal = v / np.sqrt(np.dot(v,v)) + np.float32(1e-6)

    # @numba.jit(nopython=True )
    def J1(v):
        g = np.dot(v,v) + np.float32(1e-6)
        sqrt_g = np.sqrt(g) 
        J = -np.outer(v,v)
        J = J / (g*sqrt_g)
        J += np.eye(3,dtype=np.float32)/sqrt_g
        return J
    
    J_vo_v =  J1(v)
    J_vo_w = np.zeros((3,3), dtype=np.float32)

    
    proj = np.dot(w, v_orthonormal) * v_orthonormal
    
    # @numba.jit(nopython=True )
    def J2(v,w):
        Jv = np.eye(3, dtype=np.float32)*np.dot(v,w) + np.outer(v,w)
        Jw = np.outer(v,v)  
        return Jv, Jw
    J_proj_vo, J_proj_w = J2(v_orthonormal, w)
    J_proj_v = J_proj_vo @ J_vo_v
    
    # passed J_proj_v and J_proj_w
    #---------------------------- 

    w_orthogonal = w - proj
    J_wo_w = np.eye(3, dtype=np.float32) - J_proj_w
    J_wo_v = -J_proj_v

    w_orthonormal2 = w_orthogonal / (np.linalg.norm(w_orthogonal) + np.float32(1e-6))
    J_wo2_wo = J1(w_orthogonal)

    J_wo2_w = J_wo2_wo @ J_wo_w
    J_wo2_v = J_wo2_wo @ J_wo_v

    u_orthonormal = np.cross(v_orthonormal, w_orthonormal2)
    # @numba.jit(nopython=True )
    def J3(v, w):
        return np.array([[np.float32(0), w[2], -w[1]],[-w[2],np.float32(0), w[0]],[w[1], -w[0], np.float32(0)]]), np.array([[np.float32(0), -v[2], v[1]],[v[2], np.float32(0), -v[0]],[-v[1], v[0], np.float32(0)]])
    
    J_uo_vo, J_uo_wo2 = J3(v_orthonormal, w_orthonormal2)
    J_uo_v = J_uo_vo @ J_vo_v + J_uo_wo2 @ J_wo2_v
    J_uo_w = J_uo_wo2 @ J_wo2_w

    

    R = np.stack((v_orthonormal, w_orthonormal2, u_orthonormal), axis=-1)

    J_r_v = np.concatenate((np.stack((J_vo_v[0,:],J_wo2_v[0,:], J_uo_v[0,:])), 
                        np.stack((J_vo_v[1,:],J_wo2_v[1,:], J_uo_v[1,:])), 
                        np.stack((J_vo_v[2,:],J_wo2_v[2,:], J_uo_v[2,:]))), axis=0)


    J_r_w = np.concatenate((np.stack((J_vo_w[0,:],J_wo2_w[0,:], J_uo_w[0,:])),
                        np.stack((J_vo_w[1,:],J_wo2_w[1,:], J_uo_w[1,:])), 
                        np.stack((J_vo_w[2,:],J_wo2_w[2,:], J_uo_w[2,:])),), axis=0)
    # @numba.jit(nopython=True )
    def J4(R_r, v):
        v0, v1, v2 = v
        o = np.float32(0)
        return np.array([[v0, o, o, v1, o, o, v2, o, o],
                        [o, v0, o, o, v1, o, o, v2, o],
                        [o, o, v0, o, o, v1, o, o, v2]]),R_r.reshape(3,3).T
    
    J_vlocal_r, J_vlocal_v = J4(R.reshape(-1), v)
    J_wlocal_r, J_wlocal_w = J4(R.reshape(-1), w)

   
    J_vlocal_v = J_vlocal_v + J_vlocal_r @ J_r_v

    J_vlocal_w =  J_vlocal_r @ J_r_w
    J_wlocal_w = J_wlocal_w + J_wlocal_r @ J_r_w
    J_wlocal_v =  J_wlocal_r @ J_r_v

    return np.concatenate((np.concatenate((J_r_v, J_r_w), axis=1), np.concatenate((J_vlocal_v, J_vlocal_w), axis=1), np.concatenate((J_wlocal_v, J_wlocal_w), axis=1),), axis=0)

def gs_torch(v,w):
    v_orthonormal = v / (torch.norm(v) + 1e-6)
    proj = torch.dot(w, v_orthonormal) * v_orthonormal
    w_orthogonal = w - proj
    w_orthonormal = w_orthogonal / (torch.norm(w_orthogonal) + 1e-6)
    u_orthonormal = torch.cross(v_orthonormal, w_orthonormal)

    R = torch.stack((v_orthonormal, w_orthonormal, u_orthonormal), dim=-1)

    v_local = R.T@v
    w_local = R.T@ w

    return R, v_local, w_local

def aero_forward(aero_params, v, w):
    w = w @ aero_params['recode.weight'].T + aero_params['recode.bias']
    R, v_local, w_local = gs(v, w)     
    feat = np.array([v_local[0], w_local[0], w_local[1]])
    h = np.maximum(0, feat @ aero_params['layer1.0.weight'].T + aero_params['layer1.0.bias'])
    h2 = np.maximum(h @ aero_params['layer2.0.weight'].T + aero_params['layer2.0.bias'], 0) * h + h
    y = np.maximum(h2 @ aero_params['dec.0.weight'].T + aero_params['dec.0.bias'], 0)
    y = y @ aero_params['dec.2.weight'].T + aero_params['dec.2.bias']
    y = R@y
    y = y + aero_params['bias'].reshape(-1) 
    return y

@numba.jit(nopython=True)
def aero_forward_jitted(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, bias, v, w):
    w = w @ recode_weight.T + recode_bias

    R, v_local, w_local = gs(v, w)     
    feat = np.array([v_local[0], w_local[0], w_local[1]])
    h = np.maximum(0, feat @ layer1_weight.T + layer1_bias)
    h2 = np.maximum(h @ layer2_weight.T + layer2_bias, 0) * h + h
    y = np.maximum(h2 @ dec_0_weight.T + dec_0_bias, 0)
    y = y @ dec_2_weight.T + dec_2_bias
    y = R@y
    y = y + bias.reshape(-1) 
    return y



global_aero_params, _ = get_np_params()
global_recode_weight = global_aero_params['recode.weight']
global_recode_bias = global_aero_params['recode.bias']
global_layer1_weight = global_aero_params['layer1.0.weight']
global_layer1_bias = global_aero_params['layer1.0.bias']
global_layer2_weight = global_aero_params['layer2.0.weight']
global_layer2_bias = global_aero_params['layer2.0.bias']
global_dec_0_weight = global_aero_params['dec.0.weight']
global_dec_0_bias = global_aero_params['dec.0.bias']
global_dec_2_weight = global_aero_params['dec.2.weight']
global_dec_2_bias = global_aero_params['dec.2.bias']
global_bias = global_aero_params['bias']


# @numba.jit(nopython=True)
# def aero_forward_global_jitted(v,w):
#     w = w @ global_recode_weight.T + global_recode_bias
#     R, v_local, w_local = gs(v, w)     
#     feat = np.array([v_local[0], w_local[0], w_local[1]])
#     h = np.maximum(0, feat @ global_layer1_weight.T + global_layer1_bias)
#     h2 = np.maximum(h @ global_layer2_weight.T + global_layer2_bias, 0) * h + h
#     y = np.maximum(h2 @ global_dec_0_weight.T + global_dec_0_bias, 0)
#     y = y @ global_dec_2_weight.T + global_dec_2_bias
#     y = R@y
#     y = y + global_bias.reshape(-1) 
#     return y
    
@numba.jit(nopython=True)
def aero_forward_global_jitted(v,w):
    return aero_forward_jitted(global_recode_weight, global_recode_bias,
                                global_layer1_weight, global_layer1_bias,
                                global_layer2_weight, global_layer2_bias,
                                global_dec_0_weight, global_dec_0_bias,
                                global_dec_2_weight, global_dec_2_bias,
                                global_bias, v, w)


from lfg.derive import dJ

@numba.jit(nopython=True )
def jacobian_jitted(v,w):
    return _jacobian_jitted(global_recode_weight, global_recode_bias,
                            global_layer1_weight, global_layer1_bias,
                            global_layer2_weight, global_layer2_bias,
                            global_dec_0_weight, global_dec_0_bias,
                            global_dec_2_weight, global_dec_2_bias,
                            global_bias, v, w)

@numba.jit(nopython=True )
def _jacobian_jitted(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, bias, v, w):
    
    w = w @ recode_weight.T + recode_bias
    Jv = np.eye(3, dtype=np.float32)
    Jw = recode_weight
    # J_vw = np.block([[Jv, np.zeros((3,3))], [np.zeros((3,3)), Jw]]) # 6x6
    # J_vw use concatenated instead of block
    J_vw = np.concatenate((np.concatenate((Jv, np.zeros((3,3),dtype=np.float32)), axis=1), np.concatenate((np.zeros((3,3), dtype=np.float32), Jw), axis=1)), axis=0) # 6x6
    
    R, v_local, w_local = gs(v, w)     

    # J_rvw = dJ(v, w) @ J_vw # 15x6
    J_rvw = dgs(np.concatenate((v, w))) @ J_vw
    
    # return J_rvw
    # -------------------------------
    # J_rw passed the test!
    # -------------------------------

    feat = np.array([v_local[0], w_local[0], w_local[1]])

    # J_feat = J_rvw[[9,12,13], :] # 3x6
    # J_feat use concatenated instead of block
    J_feat = np.stack((J_rvw[9,:], J_rvw[12,:], J_rvw[13,:])) # 3x6
    

    h1 =  feat @ layer1_weight.T + layer1_bias
   
    
    J_h1 = layer1_weight @ J_feat # 32x6

    h1m = np.maximum(np.float32(0.0), h1)
    # J_h1m =  np.diag((h1 > 0.0).astype(float)) @ J_h1# 32x6

    # don't use implicit bound function for bool
    tmp = np.zeros_like(h1, dtype=np.float32)
    tmp[h1 > 0] = np.float32(1.0)
    J_h1m = np.diag(tmp) @ J_h1 # 32x6
    
    
    # -------------------------------
    # J_h1m passed the test!
    # -------------------------------
    
    # self multiplicative layer
    h2 = h1m @ layer2_weight.T + layer2_bias
    J_h2_ = layer2_weight

    h2m = np.maximum(h2, 0)
    # J_h2m_ = np.diag((h2 > 0).astype(float)) @ J_h2_ # 32x32
    tmp = np.zeros_like(h2, dtype=np.float32)
    tmp[h2 > 0] = np.float32(1.0)
    J_h2m_ = np.diag(tmp) @ J_h2_ # 32x32
    J_h2m = J_h2m_ @ J_h1m # 32x6

    h2mul = h2m * h1m + h1m
    J_h2mul = np.diag(h1m)@J_h2m + np.diag(1 +  h2m) @ J_h1m
    # end self multiplicative layer
    # -------------------------------
    # J_h2mul passed the test!
    # -------------------------------

    # decode layer
    
    y1 = h2mul @ dec_0_weight.T + dec_0_bias
    y1m = np.maximum(y1, 0)

    tmp = np.zeros_like(y1, dtype=np.float32)
    tmp[y1 > 0] = np.float32(1.0)
    J_y1m_ = np.diag(tmp) @ dec_0_weight # 128x32
    # J_y1m_ = np.diag((y1 > 0).astype(float)) @ dec_0_weight # 128x32
    J_y1m = J_y1m_ @ J_h2mul # 128x6

    y2 = y1m @ dec_2_weight.T + dec_2_bias
    J_y2 = dec_2_weight @ J_y1m # 3x3

    # y3 = R@y2 + bias.reshape(-1)
    J_r = J_rvw[:9,:]
    J_y3_y2 = R # 3x3
    # J_y3_r = np.block([[y2, np.zeros(3,dtype=np.float32), np.zeros(3,dtype=np.float32)], [np.zeros(3,dtype=np.float32), y2, np.zeros(3,dtype=np.float32)], [np.zeros(3,dtype=np.float32), np.zeros(3,dtype=np.float32), y2]]) # 3x9
    # use concatenated instead of block

    J_y3_r = np.stack((np.concatenate((y2, np.zeros(3,dtype=np.float32), np.zeros(3,dtype=np.float32))), 
                             np.concatenate((np.zeros(3,dtype=np.float32), y2, np.zeros(3,dtype=np.float32))),
                               np.concatenate((np.zeros(3,dtype=np.float32), np.zeros(3,dtype=np.float32), y2)),)) # 3x9
  
    J_y3 = J_y3_r @ J_r + J_y3_y2 @ J_y2
     # -------------------------------
    # J_y3 passed the test!
    # -------------------------------

    return J_y3

def forward_twin(v,w):
    return _forward_twin(global_recode_weight, global_recode_bias,
                            global_layer1_weight, global_layer1_bias,
                            global_layer2_weight, global_layer2_bias,
                            global_dec_0_weight, global_dec_0_bias,
                            global_dec_2_weight, global_dec_2_bias,
                            global_bias, v, w)

def _forward_twin(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, bias, v, w):
    
    ## set up the parameters
    recode_weight = torch.tensor(recode_weight)
    recode_bias = torch.tensor(recode_bias)
    layer1_weight = torch.tensor(layer1_weight)
    layer1_bias = torch.tensor(layer1_bias)

    layer2_weight = torch.tensor(layer2_weight)
    layer2_bias = torch.tensor(layer2_bias)
    dec_0_bias = torch.tensor(dec_0_bias)
    dec_0_weight = torch.tensor(dec_0_weight)
    dec_2_bias = torch.tensor(dec_2_bias)
    dec_2_weight = torch.tensor(dec_2_weight)
    bias = torch.tensor(bias)

    # forward
    w = w @ recode_weight.T + recode_bias

    R, v_local, w_local = gs_torch(v, w)     


    # R, v,w local passed the test
    # ---------------------------------
    relu = nn.ReLU()
    feat = torch.stack([v_local[0], w_local[0], w_local[1]])  # Use stack instead of creating a new tensor
    h1 =  feat @ layer1_weight.T + layer1_bias
    h1m = relu(h1)

    # return feat
    
    # h1m passed the test
    # ---------------------------------

    h2 = h1m @ layer2_weight.T + layer2_bias
    h2m = relu(h2)
    h2mul = h2m * h1m + h1m

    # h2mul passed the test
    # ---------------------------------
    y1 = h2mul @ dec_0_weight.T + dec_0_bias
    y1m = relu(y1)
    y2 = y1m @ dec_2_weight.T + dec_2_bias
    y3 = R@y2+ bias.reshape(-1)
                           
    return y3




@numba.jit(nopython=True)
def phy(v,w):
    return np.cross(w,v)*0.1 + np.array([0.0, 0.0, -9.81])  - 0.5 * v * np.linalg.norm(v)
class TestCases:
    def __init__(self):
        pass
    @staticmethod
    def test_aero_forward():
        mnn, mnn_est, autoregr_MNN = get_original_model(compile=False)
        aero_params, _ = get_np_params()

        v = torch.rand(1,3).cuda()*10
        w = torch.rand(1,3).cuda()*10

        torch_time = time.time()
        for _ in range(100):
            y_torch = mnn.aero_layer(v, w)
        print('Torch time:', time.time() - torch_time)
        y_torch = y_torch.cpu().detach().numpy()

        y_np = aero_forward(aero_params, v.reshape(-1).cpu().detach().numpy().astype(np.float32), w.reshape(-1).cpu().detach().numpy().astype(np.float32))
        np_time = time.time()
        for _ in range(100):
            y_np = aero_forward(aero_params, v.reshape(-1).cpu().detach().numpy().astype(np.float32), w.reshape(-1).cpu().detach().numpy().astype(np.float32))
        print('Numpy time:', time.time() - np_time)


        
        y_np_jitted = aero_forward_jitted(aero_params['recode.weight'], aero_params['recode.bias'], 
                                          aero_params['layer1.0.weight'], aero_params['layer1.0.bias'], 
                                          aero_params['layer2.0.weight'], aero_params['layer2.0.bias'],
                                            aero_params['dec.0.weight'], aero_params['dec.0.bias'], 
                                            aero_params['dec.2.weight'], aero_params['dec.2.bias'],
                                            aero_params['bias'], 
                                            v.reshape(-1).cpu().detach().numpy().astype(np.float32), 
                                            w.reshape(-1).cpu().detach().numpy().astype(np.float32))
        jitted_time = time.time()
        for _ in range(100):
            y_np_jitted = aero_forward_jitted(aero_params['recode.weight'], aero_params['recode.bias'], 
                                          aero_params['layer1.0.weight'], aero_params['layer1.0.bias'], 
                                          aero_params['layer2.0.weight'], aero_params['layer2.0.bias'],
                                            aero_params['dec.0.weight'], aero_params['dec.0.bias'], 
                                            aero_params['dec.2.weight'], aero_params['dec.2.bias'],
                                            aero_params['bias'], 
                                            v.reshape(-1).cpu().detach().numpy().astype(np.float32), 
                                            w.reshape(-1).cpu().detach().numpy().astype(np.float32))
        print('Jitted time:', time.time() - jitted_time)

        v= v.cpu().detach().numpy().astype(np.float32)
        w= w.cpu().detach().numpy().astype(np.float32)
        y_phy = phy(v, w)
        phy_time = time.time()
        for _ in range(100):
            y_phy = phy(v, w)
        print('Physics time:', time.time() - phy_time)


        y_global = aero_forward_global_jitted(np.random.rand(3).astype(np.float32), np.random.rand(3).astype(np.float32))
        global_jitted_time = time.time()
        for _ in range(100):
            y_global = aero_forward_global_jitted(v.reshape(-1),w.reshape(-1))
        print('Global jitted time:', time.time() - global_jitted_time)




        print('Torch:', y_torch)
        print('Numpy:', y_np)
        print('Jitted:', y_np_jitted)
        # print('Physics:', y_phy)
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
        print('Torch time:', torch_time)
        # print(jacobian_torch)


        v_np = v.cpu().numpy()
        w_np = w.cpu().numpy()
        jacobian_np = jacobian_jitted(v_np, w_np)
        np_time = -time.time()
        for _ in range(100):
            jacobian_np = jacobian_jitted(v_np, w_np)
        np_time += time.time()
        print('Numpy time:', np_time)
        # print(jacobian_np)



        ## accuracy test
        v = v.cpu()
        w = w.cpu()
        v_np = v.cpu().numpy().astype(np.float32)
        w_np = w.cpu().numpy().astype(np.float32)
        jacobian_twin = F.jacobian(lambda x: forward_twin(x[:3], x[3:]), torch.cat([v, w], dim=0),)
        jacobian_np = jacobian_jitted(v_np, w_np)
        jacobian_twin = jacobian_twin.cpu().detach().numpy()
        error = np.abs(jacobian_twin - jacobian_np)
        idx= np.argmax(error)
        r_idx, c_idx = np.unravel_index(idx, error.shape)  # Convert to 2D index
        max_error = error.max()
        print('Torch:')
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
        
        
        def dgs(x):
            v = x[:3]
            w = x[3:]

            v_orthonormal = v / np.sqrt(np.dot(v,v)) + np.float32(1e-6)


            def J1(v):
                g = np.dot(v,v) + np.float32(1e-6)
                sqrt_g = np.sqrt(g) 
                J = -np.outer(v,v)
                J = J / (g*sqrt_g)
                J += np.eye(3)/sqrt_g
                return J
            
            J_vo_v =  J1(v)
            J_vo_w = np.zeros((3,3), dtype=np.float32)

            
            proj = np.dot(w, v_orthonormal) * v_orthonormal
            

            def J2(v,w):
                Jv = np.eye(3, dtype=np.float32)*np.dot(v,w) + np.outer(v,w)
                Jw = np.outer(v,v)  
                return Jv, Jw
            J_proj_vo, J_proj_w = J2(v_orthonormal, w)
            J_proj_v = J_proj_vo @ J_vo_v
            
            # passed J_proj_v and J_proj_w
            #---------------------------- 

            w_orthogonal = w - proj
            J_wo_w = np.eye(3, dtype=np.float32) - J_proj_w
            J_wo_v = -J_proj_v

            w_orthonormal2 = w_orthogonal / (np.linalg.norm(w_orthogonal) + np.float32(1e-6))
            J_wo2_wo = J1(w_orthogonal)

            J_wo2_w = J_wo2_wo @ J_wo_w
            J_wo2_v = J_wo2_wo @ J_wo_v

            u_orthonormal = np.cross(v_orthonormal, w_orthonormal2)

            def J3(v, w):
                return np.array([[np.float32(0), w[2], -w[1]],[-w[2],np.float32(0), w[0]],[w[1], -w[0], np.float32(0)]]), np.array([[np.float32(0), -v[2], v[1]],[v[2], np.float32(0), -v[0]],[-v[1], v[0], np.float32(0)]])
            
            J_uo_vo, J_uo_wo2 = J3(v_orthonormal, w_orthonormal2)
            J_uo_v = J_uo_vo @ J_vo_v + J_uo_wo2 @ J_wo2_v
            J_uo_w = J_uo_wo2 @ J_wo2_w

            

            R = np.stack((v_orthonormal, w_orthonormal2, u_orthonormal), axis=-1)

            J_r_v = np.concatenate((np.stack((J_vo_v[0,:],J_wo2_v[0,:], J_uo_v[0,:])), 
                                np.stack((J_vo_v[1,:],J_wo2_v[1,:], J_uo_v[1,:])), 
                                np.stack((J_vo_v[2,:],J_wo2_v[2,:], J_uo_v[2,:]))), axis=0)


            J_r_w = np.concatenate((np.stack((J_vo_w[0,:],J_wo2_w[0,:], J_uo_w[0,:])),
                                np.stack((J_vo_w[1,:],J_wo2_w[1,:], J_uo_w[1,:])), 
                                np.stack((J_vo_w[2,:],J_wo2_w[2,:], J_uo_w[2,:])),), axis=0)

            def J4(R_r, v):
                v0, v1, v2 = v
                o = np.float32(0.0)
                return np.array([[v0, o, o, v1, o, o, v2, o, o],
                                [o, v0, 0, o, v1, o, o, v2, o],
                                [o, o, v0, o, o, v1, o, o, v2]]),R_r.reshape(3,3).T
            
            J_vlocal_r, J_vlocal_v = J4(R.reshape(-1), v)
            J_wlocal_r, J_wlocal_w = J4(R.reshape(-1), w)

            J_vlocal_v = J_vlocal_v + J_vlocal_r @ J_r_v

            J_vlocal_w =  J_vlocal_r @ J_r_w
            J_wlocal_w = J_wlocal_w + J_wlocal_r @ J_r_w
            J_wlocal_v =  J_wlocal_r @ J_r_v

            return np.concatenate((np.concatenate((J_r_v, J_r_w), axis=1), np.concatenate((J_vlocal_v, J_vlocal_w), axis=1), np.concatenate((J_wlocal_v, J_wlocal_w), axis=1),), axis=0)

        x_cuda = torch.rand(6).cuda()   
        # dgs_torch = F.jacobian(_gs_torch, x_cuda)
        # torch_time = -time.time()
        # for  _ in range(100):
        #     dgs_torch = F.jacobian(_gs_torch, x_cuda)
        # torch_time += time.time()

        # print('Torch time:', torch_time)

        x = x_cuda.cpu().detach().numpy().astype(np.float32)
        # dgs_np = dgs(x)
        # np_time = -time.time()
        # for  i in range(100):
        #     dgs_np = dgs(x)
        # np_time += time.time()
        # print('Numpy time:', np_time)


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




def validate_3d_plot(cfg):
    mnn,  mnn_est, autoregr_MNN = get_tensorRT_model()

    # dataloader
    train_loader, test_loader = RealTrajectoryDataset.get_dataloaders(cfg)
    data = next(iter(test_loader))
    data = next(iter(test_loader))

    pN_gt = data[2:3, :,2:5]
    
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _ in range(10):
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
    # TestCases.test_aero_forward()
    TestCases.test_jacobian()
    # TestCases.test_gs_gradient()

    


if __name__ == '__main__':
    main()