import sys
import time
import numba.typed
from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch_tensorrt


from train_real_traj import RealTrajectoryDataset
from lfg.model_traj.mnn import MNN, autoregr_MNN, AeroModel, BounceModel



from lfg.estimator import OptimLayer
from draw_util import draw_util
import numba

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


@numba.jit(nopython=True)
def aero_forward_global_jitted(v,w):
    w = w @ global_recode_weight.T + global_recode_bias

    R, v_local, w_local = gs(v, w)     
    
    feat = np.array([v_local[0], w_local[0], w_local[1]])
    h = np.maximum(0, feat @ global_layer1_weight.T + global_layer1_bias)
    h2 = np.maximum(h @ global_layer2_weight.T + global_layer2_bias, 0) * h + h
    y = np.maximum(h2 @ global_dec_0_weight.T + global_dec_0_bias, 0)
    y = y @ global_dec_2_weight.T + global_dec_2_bias
    y = R@y
    y = y + global_bias.reshape(-1) 
    return y
    



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
    TestCases.test_aero_forward()


if __name__ == '__main__':
    main()