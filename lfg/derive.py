import numpy as np
import numba
import torch
from .model_traj.mnn import MNN


# encode parameters in the jitted functions
mnn = MNN(z0=0.010)
mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real/OptimLayer/run44/run44/model_MNN.pth'))
aero_model = mnn.aero_layer
global_aero_params = {}
for name, param in aero_model.named_parameters():
    global_aero_params[name] = param.cpu().detach().numpy().astype(np.float32)

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


@numba.jit(nopython=True, cache=True)
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

@numba.jit(nopython=True, cache=True)
def dgs(x):
    v = x[:3]
    w = x[3:]

    v_orthonormal = v / np.sqrt(np.dot(v,v)) + np.float32(1e-6)

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
    
    def J2(v,w):
        Jv = np.eye(3, dtype=np.float32)*np.dot(v,w) + np.outer(v,w)
        Jw = np.outer(v,v)  
        return Jv, Jw
    J_proj_vo, J_proj_w = J2(v_orthonormal, w)
    J_proj_v = J_proj_vo @ J_vo_v
    

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


@numba.jit(nopython=True, cache=True)
def aero_forward(v,w):
    return _aero_forward(global_recode_weight, global_recode_bias,
                                global_layer1_weight, global_layer1_bias,
                                global_layer2_weight, global_layer2_bias,
                                global_dec_0_weight, global_dec_0_bias,
                                global_dec_2_weight, global_dec_2_bias,
                                global_bias, v, w)

@numba.jit(nopython=True, cache=True)
def _aero_forward(recode_weight, recode_bias, layer1_weight, layer1_bias, 
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

@numba.jit(nopython=True, cache=True)
def aero_jacobian(v,w):
    return _aero_jacobian(global_recode_weight, global_recode_bias,
                            global_layer1_weight, global_layer1_bias,
                            global_layer2_weight, global_layer2_bias,
                            global_dec_0_weight, global_dec_0_bias,
                            global_dec_2_weight, global_dec_2_bias,
                            global_bias, v, w)

@numba.jit(nopython=True, cache=True) 
def _aero_jacobian(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, bias, v, w):
    
    w = w @ recode_weight.T + recode_bias
    Jv = np.eye(3, dtype=np.float32)
    Jw = recode_weight
    J_vw = np.concatenate((np.concatenate((Jv, np.zeros((3,3),dtype=np.float32)), axis=1), np.concatenate((np.zeros((3,3), dtype=np.float32), Jw), axis=1)), axis=0) # 6x6
    
    R, v_local, w_local = gs(v, w)     
    J_rvw = dgs(np.concatenate((v, w))) @ J_vw

    feat = np.array([v_local[0], w_local[0], w_local[1]])
    J_feat = np.stack((J_rvw[9,:], J_rvw[12,:], J_rvw[13,:])) # 3x6
    

    h1 =  feat @ layer1_weight.T + layer1_bias
    J_h1 = layer1_weight @ J_feat # 32x6

    h1m = np.maximum(np.float32(0.0), h1)
    tmp = np.zeros_like(h1, dtype=np.float32)
    tmp[h1 > 0] = np.float32(1.0)
    J_h1m = np.diag(tmp) @ J_h1 # 32x6
    
    h2 = h1m @ layer2_weight.T + layer2_bias
    J_h2_ = layer2_weight

    h2m = np.maximum(h2, 0)
    tmp = np.zeros_like(h2, dtype=np.float32)
    tmp[h2 > 0] = np.float32(1.0)
    J_h2m_ = np.diag(tmp) @ J_h2_ # 32x32
    J_h2m = J_h2m_ @ J_h1m # 32x6

    h2mul = h2m * h1m + h1m
    J_h2mul = np.diag(h1m)@J_h2m + np.diag(1 +  h2m) @ J_h1m

    # decode layer
    y1 = h2mul @ dec_0_weight.T + dec_0_bias
    y1m = np.maximum(y1, 0)
    tmp = np.zeros_like(y1, dtype=np.float32)
    tmp[y1 > 0] = np.float32(1.0)
    J_y1m_ = np.diag(tmp) @ dec_0_weight # 128x32
    J_y1m = J_y1m_ @ J_h2mul # 128x6

    y2 = y1m @ dec_2_weight.T + dec_2_bias
    J_y2 = dec_2_weight @ J_y1m # 3x3

    J_r = J_rvw[:9,:]
    J_y3_y2 = R # 3x3
    J_y3_r = np.stack((np.concatenate((y2, np.zeros(3,dtype=np.float32), np.zeros(3,dtype=np.float32))), 
                             np.concatenate((np.zeros(3,dtype=np.float32), y2, np.zeros(3,dtype=np.float32))),
                               np.concatenate((np.zeros(3,dtype=np.float32), np.zeros(3,dtype=np.float32), y2)),)) # 3x9
    J_y3 = J_y3_r @ J_r + J_y3_y2 @ J_y2

    return J_y3