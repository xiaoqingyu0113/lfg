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
from lfg.model_traj.mnn import MNN, autoregr_MNN, AeroModel, BounceModel



from lfg.estimator import OptimLayer
from draw_util import draw_util
import numba
import logging

from lfg.derive import bounce_forward, bounce_jacobian, gs2d, dgs2d

DTYPE = np.float64
DTYPE_TORCH = torch.float64



def get_original_model(compile=True):
    mnn = MNN(z0=0.010)
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20/model_MNN.pth'))
    mnn.eval()
    if compile:
        mnn.compile()
    mnn.to('cuda')

    mnn_est = OptimLayer(mnn, size=80, allow_grad=False, damping=0.1, max_iterations=30)
    mnn_est.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20/est_OptimLayer.pth'))
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
    mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20/model_MNN.pth'))
    
    aero_model = mnn.aero_layer
    bounce_model = mnn.bc_layer

    aero_params = {}
    bounce_params = {}

    print('----------aero model-----------------------')
    for name, param in aero_model.named_parameters():
        aero_params[name] = param.cpu().detach().numpy().astype(DTYPE)
        print(name, aero_params[name].shape)
    
    print('----------bounce model-----------------------')
    for name, param in bounce_model.named_parameters():
        bounce_params[name] = param.cpu().detach().numpy().astype(DTYPE)
        print(name, bounce_params[name].shape)

    return aero_params, bounce_params


def gs2d_torch(v2d,w2d):
    '''
    passed the test
    '''
    v_normal = v2d / (torch.linalg.norm(v2d) + 1e-8)
    R = torch.stack([
        torch.stack([v_normal[0], -v_normal[1]]), 
        torch.stack([v_normal[1], v_normal[0]])   
    ])
    v_local = R.T@v2d
    w_local = R.T@ w2d

    return R, v_local, w_local

# def gs2d(v2d, w2d):
#     '''
#     passed the test
#     '''
#     v_normal = v2d / (np.linalg.norm(v2d) + 1e-8)
#     R = np.array([
#         [v_normal[0], -v_normal[1]], 
#         [v_normal[1], v_normal[0]]   
#     ])
#     v_local = R.T@ v2d
#     w_local = R.T@ w2d

#     return R, v_local, w_local

# def dgs2d(v2d, w2d):
#     '''
#     passed accuracy test   
#     '''
#     vx, vy = v2d
#     wx, wy = w2d
#     eps=  1e-8
#     g = np.sqrt((vx**2 + vy**2 + eps))
#     tmp_y2 = vy**2 + eps
#     tmp_x2 = vx**2 + eps
#     tmp_xy = -vx*vy
#     J_vnorm_v = np.array([[tmp_y2, tmp_xy], [tmp_xy, tmp_x2]]) / g**3
#     J_R_vn = np.array([[1.0, 0.0], [0.0, -1.0],[0.0, 1.0],[1.0, 0.0]])
#     J_R_v = J_R_vn @ J_vnorm_v
#     J_vlocal_v = np.array([[vx, vy],[0,0]])/g
#     J_wlocal_v = np.array([[wx, wy],[wy, -wx]]) @ J_vnorm_v
#     J_wlocal_w = np.array([[vx, vy],[-vy, vx]])/g

#     return J_R_v, J_vlocal_v, J_wlocal_v, J_wlocal_w


_, global_bounce_params = get_np_params()
global_bounce_recode_weight = global_bounce_params['recode.weight']
global_bounce_recode_bias = global_bounce_params['recode.bias']


global_bounce_layer1_weight = global_bounce_params['layer1.0.weight']
global_bounce_layer1_bias = global_bounce_params['layer1.0.bias']
global_bounce_layer2_weight = global_bounce_params['layer2.0.weight']
global_bounce_layer2_bias = global_bounce_params['layer2.0.bias']
global_bounce_layer3_weight = global_bounce_params['layer3.0.weight']
global_bounce_layer3_bias = global_bounce_params['layer3.0.bias']

global_dec_0_weight = global_bounce_params['dec.0.weight']
global_dec_0_bias = global_bounce_params['dec.0.bias']
global_dec_2_weight = global_bounce_params['dec.2.weight']
global_dec_2_bias = global_bounce_params['dec.2.bias']



def forward_twin(v,w):
    return _forward_twin(global_bounce_recode_weight, global_bounce_recode_bias,
                        global_bounce_layer1_weight, global_bounce_layer1_bias,
                        global_bounce_layer2_weight, global_bounce_layer2_bias,
                        global_bounce_layer3_weight, global_bounce_layer3_bias,
                        global_dec_0_weight, global_dec_0_bias,
                        global_dec_2_weight, global_dec_2_bias,
                        v, w)

def _forward_twin(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, layer3_weight, layer3_bias,
                        dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, v, w):
    
    ## set up the parameters
    recode_weight = torch.tensor(recode_weight)
    recode_bias = torch.tensor(recode_bias)
    layer1_weight = torch.tensor(layer1_weight)
    layer1_bias = torch.tensor(layer1_bias)
    layer2_weight = torch.tensor(layer2_weight)
    layer2_bias = torch.tensor(layer2_bias)
    layer3_weight = torch.tensor(layer3_weight)
    layer3_bias = torch.tensor(layer3_bias)
    dec_0_weight = torch.tensor(dec_0_weight)
    dec_0_bias = torch.tensor(dec_0_bias)
    dec_2_weight = torch.tensor(dec_2_weight)
    dec_2_bias = torch.tensor(dec_2_bias)

    
    w = w@ recode_weight.T + recode_bias
    R2d, v2d_local, w2d_local = gs2d_torch(v[:2], w[:2])    
    


    v_local = torch.cat([v2d_local, v[2:3]])
    w_local = torch.cat([w2d_local, w[2:3]])

   
    # normalize, imperial unit
    v_normalize = v_local / 3.0
    w_normalize = w_local / 7.0


    x = torch.cat([v_normalize, w_normalize])
    
    relu = nn.ReLU()

    
    
    h0 = relu(x@layer1_weight.T + layer1_bias)
    h1 = relu(h0@layer2_weight.T + layer2_bias)*h0 + h0

    h2 = relu(h1@layer3_weight.T + layer3_bias)*h1 + h1


    x1 = relu(h2@dec_0_weight.T + dec_0_bias)
    x2 = x1@dec_2_weight.T + dec_2_bias
 
    
    v2d_local_new = x2[:2] * 3.0
    vz_new = x2[2:3] * 3.0
    w2d_local_new = x2[3:5] * 7.0
    wz_new = x2[5:6] * 7.0

    v2d_new = R2d@v2d_local_new
    w2d_new = R2d@w2d_local_new

    v_new = torch.cat([v2d_new, vz_new])
    w_new = torch.cat([w2d_new, wz_new])

    return v_new, w_new
                        




@numba.jit(nopython=True)
def phy(v,w):
    return np.cross(w,v)*0.1 + np.array([0.0, 0.0, -9.81])  - 0.5 * v * np.linalg.norm(v)


class TestCases:
    def __init__(self):
        pass
    @staticmethod
    def test_aero_forward():
        mnn, mnn_est, autoregr_MNN = get_original_model(compile=False)
        aero_params, bc_params = get_np_params()

        v = torch.rand(1,3).cuda()*10
        w = torch.rand(1,3).cuda()*10

        v_torch, w_torch = mnn.bc_layer(v, w)
        v_torch = v_torch.reshape(-1).cpu().detach().numpy().astype(DTYPE)
        w_torch = w_torch.reshape(-1).cpu().detach().numpy().astype(DTYPE)

        print('Torch:', v_torch, w_torch)

        # twin 
        v = v.reshape(-1).cpu().to(DTYPE_TORCH)
        w =  w.reshape(-1).cpu().to(DTYPE_TORCH)
        y_twin = forward_twin(v, w)
        print('Twin:', y_twin)
        


        v = v.reshape(-1).cpu().detach().numpy().astype(DTYPE)
        w =  w.reshape(-1).cpu().detach().numpy().astype(DTYPE)
        y_np = bounce_forward(v, w)
        print('Numpy:', y_np)
       

   
       

    @staticmethod
    def test_jacobian():
        mnn, mnn_est, autoregr_MNN = get_original_model(compile=False)

        model = mnn.bc_layer
        
        v = torch.rand(1,3).cuda()*10
        w = torch.rand(1,3).cuda()*10
        # input_tensor = (v, w)
        # forward_func = lambda x, y: aero_model(x[None, :], y[None, :])

        # Compute the Jacobian
        # jacobian_torch = F.jacobian(forward_func, input_tensor)
        # torch_time = -time.time()
        # for _ in range(100):
        #     jacobian_torch = F.jacobian(forward_func, input_tensor)
        # torch_time += time.time()
        # print('Torch time:', torch_time/100)

        # v_np = v.cpu().numpy().astype(DTYPE)
        # w_np = w.cpu().numpy().astype(DTYPE)
        # comp_time = -time.time()
        # jacobian_np = bounce_jacobian(v_np[0], w_np[0])
        # comp_time += time.time()
        # print('Compiled model loading time:', comp_time)
        # np_time = -time.time()
        # for _ in range(100):
        #     jacobian_np = bounce_jacobian(v_np[0], w_np[0])
        # np_time += time.time()
        # print('Compiled model time:', np_time/100)

        ## accuracy test
   
        

        jacobian_original = F.jacobian(model, (v,w))

        v = v.cpu().to(DTYPE_TORCH)
        w = w.cpu().to(DTYPE_TORCH)
        jacobian_twin = F.jacobian(forward_twin, (v[0],w[0]))


        
        v_np = v[0].cpu().numpy().astype(DTYPE)
        w_np = w[0].cpu().numpy().astype(DTYPE)
        jacobian_np = bounce_jacobian(v_np, w_np)

        print('Twin:\n', jacobian_twin)
        # print('Original:\n', jacobian_original)
        print('Numpy:\n', jacobian_np)

        error_v = np.abs(jacobian_twin[0].numpy().astype(DTYPE) - jacobian_np[0]).max()
        error_w = np.abs(jacobian_twin[1].numpy().astype(DTYPE) - jacobian_np[1]).max()

        print(f'error_v ={error_v}, \nerror_w={error_w}')



    @staticmethod
    def test_gs_gradient():
        num_loops = 100
        x_cuda = torch.rand(4).cuda()   
        dgs_torch = F.jacobian(gs2d_torch, (x_cuda[:2], x_cuda[2:]))
        torch_time = -time.time()
        for  _ in range(num_loops):
            dgs_torch = F.jacobian(gs2d_torch, (x_cuda[:2], x_cuda[2:]))
        torch_time += time.time()

        print('Torch time:', torch_time)

        # gs2d(x_cuda[:2].cpu().numpy().astype(DTYPE), x_cuda[2:].cpu().numpy().astype(DTYPE))
        # raise
        x = x_cuda.cpu().detach().numpy().astype(DTYPE)
        dgs_np = dgs2d(x[:2], x[2:])
        np_time = -time.time()
        for  i in range(num_loops):
            dgs_np = dgs2d(x[:2], x[2:])
        np_time += time.time()
        print('compiled version time:', np_time)

        ## accuracy test
        dgs_torch = F.jacobian(gs2d_torch, (x_cuda[:2], x_cuda[2:]))

        J_R_v, J_vlocal_v, J_wlocal_v, J_wlocal_w = dgs2d(x[:2], x[2:])
  
        print('Torch:\n', dgs_torch)
        print('Numpy:\n', J_R_v, J_vlocal_v, J_wlocal_v, J_wlocal_w)





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