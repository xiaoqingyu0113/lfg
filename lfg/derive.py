import numpy as np
import numba
import torch
from .model_traj.mnn import MNN
import gtsam
from typing import List, Optional

DTYPE = np.float64

# encode parameters in the jitted functions
mnn = MNN(z0=0.010)
mnn.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20/model_MNN.pth'))
aero_model = mnn.aero_layer
global_aero_params = {}
global_bc_params = {}
for name, param in aero_model.named_parameters():
    global_aero_params[name] = param.cpu().detach().numpy().astype(DTYPE)

for name, param in mnn.bc_layer.named_parameters():
    global_bc_params[name] = param.cpu().detach().numpy().astype(DTYPE)

global_aero_recode_weight = global_aero_params['recode.weight']
global_aero_recode_bias = global_aero_params['recode.bias']
global_aero_layer1_weight = global_aero_params['layer1.0.weight']
global_aero_layer1_bias = global_aero_params['layer1.0.bias']
global_aero_layer2_weight = global_aero_params['layer2.0.weight']
global_aero_layer2_bias = global_aero_params['layer2.0.bias']

global_aero_dec_0_weight = global_aero_params['dec.0.weight']
global_aero_dec_0_bias = global_aero_params['dec.0.bias']
global_aero_dec_2_weight = global_aero_params['dec.2.weight']
global_aero_dec_2_bias = global_aero_params['dec.2.bias']
global_aero_bias = global_aero_params['bias']


global_bc_recode_weight = global_bc_params['recode.weight']
global_bc_recode_bias = global_bc_params['recode.bias']
global_bc_layer1_weight = global_bc_params['layer1.0.weight']
global_bc_layer1_bias = global_bc_params['layer1.0.bias']
global_bc_layer2_weight = global_bc_params['layer2.0.weight']
global_bc_layer2_bias = global_bc_params['layer2.0.bias']
global_bc_layer3_weight = global_bc_params['layer3.0.weight']
global_bc_layer3_bias = global_bc_params['layer3.0.bias']

global_bc_dec_0_weight = global_bc_params['dec.0.weight']
global_bc_dec_0_bias = global_bc_params['dec.0.bias']
global_bc_dec_2_weight = global_bc_params['dec.2.weight']
global_bc_dec_2_bias = global_bc_params['dec.2.bias']


@numba.jit(nopython=True, cache=True)
def gs2d(v2d, w2d):
    '''
    passed the test
    '''
    v_normal = v2d / (np.linalg.norm(v2d) + 1e-8)
    R = np.array([
        [v_normal[0], -v_normal[1]], 
        [v_normal[1], v_normal[0]]   
    ])
    v_local = R.T@ v2d
    w_local = R.T@ w2d

    return R, v_local, w_local

@numba.jit(nopython=True, cache=True)
def dgs2d(v2d, w2d):
    '''
    passed accuracy test   
    '''
    vx, vy = v2d
    wx, wy = w2d
    eps=  DTYPE(1e-8)
    g = np.sqrt((vx**2 + vy**2 + eps))
    tmp_y2 = vy**2 + eps
    tmp_x2 = vx**2 + eps
    tmp_xy = -vx*vy
    J_vnorm_v = np.array([[tmp_y2, tmp_xy], [tmp_xy, tmp_x2]]) / g**3
    J_R_vn = np.array([[1.0, 0.0], [0.0, -1.0],[0.0, 1.0],[1.0, 0.0]], dtype=DTYPE)
    J_R_v = J_R_vn @ J_vnorm_v
    J_vlocal_v = np.array([[vx, vy],[0.0,0.0]])/g
    J_wlocal_v = np.array([[wx, wy],[wy, -wx]]) @ J_vnorm_v
    J_wlocal_w = np.array([[vx, vy],[-vy, vx]])/g

    return J_R_v, J_vlocal_v, J_wlocal_v, J_wlocal_w

@numba.jit(nopython=True, cache=True)
def _bounce_forward(recode_weight, recode_bias, layer1_weight, layer1_bias,
                    layer2_weight, layer2_bias, layer3_weight, layer3_bias,
                    dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, v, w):
    
    
    w = w @ recode_weight.T + recode_bias
    R2d, v2d_local, w2d_local = gs2d(v[:2], w[:2])


    v_local = np.concatenate((v2d_local, v[2:]))
    w_local = np.concatenate((w2d_local, w[2:]))

    v_normalize = v_local / 3.0
    w_normalize = w_local / 7.0


    x = np.concatenate((v_normalize, w_normalize))
    h0 = np.maximum(0, x @ layer1_weight.T + layer1_bias)
    h1 = np.maximum(0, h0 @ layer2_weight.T + layer2_bias) * h0 + h0
    h2 = np.maximum(0, h1 @ layer3_weight.T + layer3_bias) * h1 + h1

    y1 = np.maximum(0, h2 @ dec_0_weight.T + dec_0_bias)
    y2 = y1 @ dec_2_weight.T + dec_2_bias

    v2d_local_new = R2d @ y2[:2] * 3.0
    vz_new = y2[2:3] * 3.0
    w2d_local_new = R2d @ y2[3:5] * 7.0
    wz_new = y2[5:6] * 7.0

    v_new = np.concatenate((v2d_local_new, vz_new))
    w_new = np.concatenate((w2d_local_new, wz_new))


    return v_new, w_new

@numba.jit(nopython=True, cache=True)
def bounce_forward(v,w):
    return _bounce_forward(global_bc_recode_weight, global_bc_recode_bias,
                            global_bc_layer1_weight, global_bc_layer1_bias,
                            global_bc_layer2_weight, global_bc_layer2_bias,
                            global_bc_layer3_weight, global_bc_layer3_bias,
                            global_bc_dec_0_weight, global_bc_dec_0_bias,
                            global_bc_dec_2_weight, global_bc_dec_2_bias,
                             v, w)

@numba.jit(nopython=True, cache=True)
def _bounce_jacobian(recode_weight, recode_bias, layer1_weight, layer1_bias,
                    layer2_weight, layer2_bias, layer3_weight, layer3_bias,
                    dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, v, w):
        
    w = w @ recode_weight.T + recode_bias
    J_w = recode_weight

    R2d, v2d_local, w2d_local = gs2d(v[:2], w[:2])
    J_R2d_v2d, J_v2dlocal_v, J_w2dlocal_v, J_w2dlocal_w = dgs2d(v[:2], w[:2])


    v_local = np.concatenate((v2d_local, v[2:]))
    w_local = np.concatenate((w2d_local, w[2:]))

    J_vlocal_v = np.array([[J_v2dlocal_v[0,0], J_v2dlocal_v[0,1], 0.0],
                         [J_v2dlocal_v[1,0], J_v2dlocal_v[1,1], 0.0],
                         [0.0, 0.0, 1.0]])
    J_wlocal_v = np.array([[J_w2dlocal_v[0,0], J_w2dlocal_v[0,1], 0.0],
                            [J_w2dlocal_v[1,0], J_w2dlocal_v[1,1], 0.0],
                            [0.0, 0.0, 0.0]]) 
    J_wlocal_w = np.array([[J_w2dlocal_w[0,0], J_w2dlocal_w[0,1], 0.0],
                            [J_w2dlocal_w[1,0], J_w2dlocal_w[1,1], 0.0],
                            [0.0, 0.0, 1.0]]) @ J_w
    
    
    

    v_normalize = v_local / DTYPE(3.0)
    w_normalize = w_local / DTYPE(7.0)

    J_vnorm_v = J_vlocal_v / DTYPE(3.0)
    J_wnorm_v = J_wlocal_v / DTYPE(7.0)
    J_wnorm_w = J_wlocal_w / DTYPE(7.0)

    x = np.concatenate((v_normalize, w_normalize))
    J_x_v = np.concatenate((J_vnorm_v, J_wnorm_v), axis=0)
    J_x_w = np.concatenate((np.zeros((3,3), dtype=DTYPE), J_wnorm_w), axis=0)

    # ----------------------- passed accuracy test -----------------------

    h0 = np.maximum(DTYPE(0.0), x @ layer1_weight.T + layer1_bias)
    J_h0_v = layer1_weight @ J_x_v
    J_h0_w = layer1_weight @ J_x_w


    h0m = np.maximum(DTYPE(0.0), h0)
    tmp = np.zeros_like(h0, dtype=DTYPE)
    tmp[h0 > 0] = DTYPE(1.0)
    J_h0m_v = np.diag(tmp) @ J_h0_v
    J_h0m_w = np.diag(tmp) @ J_h0_w

    h1 = h0m @ layer2_weight.T + layer2_bias
    J_h1_v = layer2_weight @ J_h0m_v
    J_h1_w = layer2_weight @ J_h0m_w

    h1m = np.maximum(h1, DTYPE(0.0))
    tmp = np.zeros_like(h1, dtype=DTYPE)
    tmp[h1 > 0] = DTYPE(1.0)
    J_h1m_ = np.diag(tmp) 
    J_h1m_v = J_h1m_ @ J_h1_v
    J_h1m_w = J_h1m_ @ J_h1_w

    h1mul = h1m * h0m + h0m
    J_h1mul_v = np.diag(h0m) @ J_h1m_v + np.diag(1 + h1m) @ J_h0m_v
    J_h1mul_w = np.diag(h0m) @ J_h1m_w + np.diag(1 + h1m) @ J_h0m_w

    
    h2 = h1mul @ layer3_weight.T + layer3_bias
    J_h2_v = layer3_weight @ J_h1mul_v
    J_h2_w = layer3_weight @ J_h1mul_w

    h2m = np.maximum(h2, DTYPE(0.0))
    tmp = np.zeros_like(h2, dtype=DTYPE)
    tmp[h2 > 0] = DTYPE(1.0)
    J_h2m_ = np.diag(tmp)
    J_h2m_v = J_h2m_ @ J_h2_v
    J_h2m_w = J_h2m_ @ J_h2_w

    h2mul = h2m * h1mul + h1mul
    J_h2mul_v = np.diag(h1mul) @ J_h2m_v + np.diag(1 + h2m) @ J_h1mul_v
    J_h2mul_w = np.diag(h1mul) @ J_h2m_w + np.diag(1 + h2m) @ J_h1mul_w

    # decode layer
    y1 = h2mul @ dec_0_weight.T + dec_0_bias
    J_y1_v = dec_0_weight @ J_h2mul_v
    J_y1_w = dec_0_weight @ J_h2mul_w
    
    y1m = np.maximum(y1, DTYPE(0.0))
    tmp = np.zeros_like(y1, dtype=DTYPE)
    tmp[y1 > 0.0] = 1.0
    J_y1m_v = np.diag(tmp) @ J_y1_v
    J_y1m_w = np.diag(tmp) @ J_y1_w

    y2 = y1m @ dec_2_weight.T + dec_2_bias
    J_y2_v = dec_2_weight @ J_y1m_v
    J_y2_w = dec_2_weight @ J_y1m_w

    # --------------------------------- above passed accuracy test ------------------------------
                                
    J_vnew_v = np.array([
                        [J_R2d_v2d[0,0] * y2[0] + R2d[0,0] * J_y2_v[0, 0] + J_R2d_v2d[1,0] * y2[1] + R2d[0,1] * J_y2_v[1, 0],
                        J_R2d_v2d[0,1] * y2[0] + R2d[0,0] * J_y2_v[0, 1] + J_R2d_v2d[1,1] * y2[1] + R2d[0,1] * J_y2_v[1, 1],
                        R2d[0,0] * J_y2_v[0, 2] + R2d[0,1] * J_y2_v[1, 2]],

                        [J_R2d_v2d[2,0] * y2[0] + R2d[1,0] * J_y2_v[0, 0] + J_R2d_v2d[3,0] * y2[1] + R2d[1,1] * J_y2_v[1, 0],
                        J_R2d_v2d[2,1] * y2[0] + R2d[1,0] * J_y2_v[0, 1] + J_R2d_v2d[3,1] * y2[1] + R2d[1,1] * J_y2_v[1, 1],
                        R2d[1,0] * J_y2_v[0, 2] + R2d[1,1] * J_y2_v[1, 2]],

                        [J_y2_v[2,0], 
                         J_y2_v[2,1], 
                         J_y2_v[2,2]]
                        ]) * DTYPE(3.0)

    J_vnew_w = np.array([R2d[0,0] * J_y2_w[0,0] + R2d[0,1] * J_y2_w[1,0],
                        R2d[0,0] * J_y2_w[0,1] + R2d[0,1] * J_y2_w[1,1],
                        R2d[0,0] * J_y2_w[0,2] + R2d[0,1] * J_y2_w[1,2],
                        R2d[1,0] * J_y2_w[0,0] + R2d[1,1] * J_y2_w[1,0],
                        R2d[1,0] * J_y2_w[0,1] + R2d[1,1] * J_y2_w[1,1],
                        R2d[1,0] * J_y2_w[0,2] + R2d[1,1] * J_y2_w[1,2],
                        J_y2_w[2,0],
                        J_y2_w[2,1],
                        J_y2_w[2,2]]).reshape(3,3) * DTYPE(3.0)

    J_wnew_v = np.array([J_R2d_v2d[0,0] * y2[3] + R2d[0,0] * J_y2_v[3,0] + J_R2d_v2d[1,0] * y2[4] + R2d[0,1] * J_y2_v[4,0], #bad
                        J_R2d_v2d[0,1] * y2[3] + R2d[0,0] * J_y2_v[3,1] + J_R2d_v2d[1,1] * y2[4] + R2d[0,1] * J_y2_v[4,1], # bad
                        R2d[0,0] * J_y2_v[3,2] + R2d[0,1] * J_y2_v[4,2], #good          
                        J_R2d_v2d[2,0] * y2[3] + R2d[1,0] * J_y2_v[3,0] + J_R2d_v2d[3,0] * y2[4] + R2d[1,1] * J_y2_v[4,0], #bad
                        J_R2d_v2d[2,1] * y2[3] + R2d[1,0] * J_y2_v[3,1] + J_R2d_v2d[3,1] * y2[4] + R2d[1,1] * J_y2_v[4,1], #bad
                        R2d[1,0] * J_y2_v[3,2] + R2d[1,1] * J_y2_v[4,2], #good
                        J_y2_v[5,0], #bad
                        J_y2_v[5,1], #bad
                        J_y2_v[5,2]]).reshape(3,3) * DTYPE(7.0) #good 
    
    J_wnew_w = np.array([R2d[0,0] * J_y2_w[3,0] + R2d[0,1] * J_y2_w[4,0],
                        R2d[0,0] * J_y2_w[3,1] + R2d[0,1] * J_y2_w[4,1],
                        R2d[0,0] * J_y2_w[3,2] + R2d[0,1] * J_y2_w[4,2],
                        R2d[1,0] * J_y2_w[3,0] + R2d[1,1] * J_y2_w[4,0],
                        R2d[1,0] * J_y2_w[3,1] + R2d[1,1] * J_y2_w[4,1],
                        R2d[1,0] * J_y2_w[3,2] + R2d[1,1] * J_y2_w[4,2],
                        J_y2_w[5,0],
                        J_y2_w[5,1],
                        J_y2_w[5,2]]).reshape(3,3) * DTYPE(7.0)
    
    
    return J_vnew_v, J_vnew_w, J_wnew_v, J_wnew_w




   

def bounce_jacobian(v,w):
    return _bounce_jacobian(global_bc_recode_weight, global_bc_recode_bias,
                            global_bc_layer1_weight, global_bc_layer1_bias,
                            global_bc_layer2_weight, global_bc_layer2_bias,
                            global_bc_layer3_weight, global_bc_layer3_bias,
                            global_bc_dec_0_weight, global_bc_dec_0_bias,
                            global_bc_dec_2_weight, global_bc_dec_2_bias,
                            v, w)

@numba.jit(nopython=True, cache=True)
def gs(v,w):
    v_orthonormal = v / (np.linalg.norm(v) + DTYPE(1e-6))
    proj = np.dot(w, v_orthonormal) * v_orthonormal
    w_orthogonal = w - proj
    w_orthonormal = w_orthogonal / (np.linalg.norm(w_orthogonal) + DTYPE(1e-6))
    u_orthonormal = np.cross(v_orthonormal, w_orthonormal)

    R = np.stack((v_orthonormal, w_orthonormal, u_orthonormal), axis=-1)

    v_local = R.T@v
    w_local = R.T@ w
    return R, v_local, w_local

@numba.jit(nopython=True, cache=True)
def dgs(x):
    v = x[:3]
    w = x[3:]

    v_orthonormal = v / np.sqrt(np.dot(v,v)) + DTYPE(1e-6)

    def J1(v):
        g = np.dot(v,v) + DTYPE(1e-6)
        sqrt_g = np.sqrt(g) 
        J = -np.outer(v,v)
        J = J / (g*sqrt_g)
        J += np.eye(3,dtype=DTYPE)/sqrt_g
        return J
    
    J_vo_v =  J1(v)
    J_vo_w = np.zeros((3,3), dtype=DTYPE)

    proj = np.dot(w, v_orthonormal) * v_orthonormal
    
    def J2(v,w):
        Jv = np.eye(3, dtype=DTYPE)*np.dot(v,w) + np.outer(v,w)
        Jw = np.outer(v,v)  
        return Jv, Jw
    J_proj_vo, J_proj_w = J2(v_orthonormal, w)
    J_proj_v = J_proj_vo @ J_vo_v
    

    w_orthogonal = w - proj
    J_wo_w = np.eye(3, dtype=DTYPE) - J_proj_w
    J_wo_v = -J_proj_v

    w_orthonormal2 = w_orthogonal / (np.linalg.norm(w_orthogonal) + DTYPE(1e-6))
    J_wo2_wo = J1(w_orthogonal)

    J_wo2_w = J_wo2_wo @ J_wo_w
    J_wo2_v = J_wo2_wo @ J_wo_v

    u_orthonormal = np.cross(v_orthonormal, w_orthonormal2)

    def J3(v, w):
        return np.array([[DTYPE(0), w[2], -w[1]],[-w[2],DTYPE(0), w[0]],[w[1], -w[0], DTYPE(0)]]), np.array([[DTYPE(0), -v[2], v[1]],[v[2], DTYPE(0), -v[0]],[-v[1], v[0], DTYPE(0)]])
    
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
        o = DTYPE(0)
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
    return _aero_forward(global_aero_recode_weight, global_aero_recode_bias,
                                global_aero_layer1_weight, global_aero_layer1_bias,
                                global_aero_layer2_weight, global_aero_layer2_bias,
                                global_aero_dec_0_weight, global_aero_dec_0_bias,
                                global_aero_dec_2_weight, global_aero_dec_2_bias,
                                global_aero_bias, v, w)

@numba.jit(nopython=True, cache=True)
def _aero_forward(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, 
                        dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, bias, v, w):
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
    return _aero_jacobian(global_aero_recode_weight, global_aero_recode_bias,
                            global_aero_layer1_weight, global_aero_layer1_bias,
                            global_aero_layer2_weight, global_aero_layer2_bias,
                            global_aero_dec_0_weight, global_aero_dec_0_bias,
                            global_aero_dec_2_weight, global_aero_dec_2_bias,
                            global_aero_bias, v, w)

@numba.jit(nopython=True, cache=True) 
def _aero_jacobian(recode_weight, recode_bias, layer1_weight, layer1_bias, 
                        layer2_weight, layer2_bias, dec_0_weight, dec_0_bias, dec_2_weight, dec_2_bias, bias, v, w):
    
    w = w @ recode_weight.T + recode_bias
    Jv = np.eye(3, dtype=DTYPE)
    Jw = recode_weight
    J_vw = np.concatenate((np.concatenate((Jv, np.zeros((3,3),dtype=DTYPE)), axis=1), np.concatenate((np.zeros((3,3), dtype=DTYPE), Jw), axis=1)), axis=0) # 6x6
    
    R, v_local, w_local = gs(v, w)     
    J_rvw = dgs(np.concatenate((v, w))) @ J_vw

    feat = np.array([v_local[0], w_local[0], w_local[1]])
    J_feat = np.stack((J_rvw[9,:], J_rvw[12,:], J_rvw[13,:])) # 3x6
    

    h1 =  feat @ layer1_weight.T + layer1_bias
    J_h1 = layer1_weight @ J_feat # 32x6

    h1m = np.maximum(DTYPE(0.0), h1)
    tmp = np.zeros_like(h1, dtype=DTYPE)
    tmp[h1 > 0] = DTYPE(1.0)
    J_h1m = np.diag(tmp) @ J_h1 # 32x6
    
    h2 = h1m @ layer2_weight.T + layer2_bias
    J_h2_ = layer2_weight

    h2m = np.maximum(h2, 0)
    tmp = np.zeros_like(h2, dtype=DTYPE)
    tmp[h2 > 0] = DTYPE(1.0)
    J_h2m_ = np.diag(tmp) @ J_h2_ # 32x32
    J_h2m = J_h2m_ @ J_h1m # 32x6

    h2mul = h2m * h1m + h1m
    J_h2mul = np.diag(h1m)@J_h2m + np.diag(1 +  h2m) @ J_h1m

    # decode layer
    y1 = h2mul @ dec_0_weight.T + dec_0_bias
    y1m = np.maximum(y1, 0)
    tmp = np.zeros_like(y1, dtype=DTYPE)
    tmp[y1 > 0] = DTYPE(1.0)
    J_y1m_ = np.diag(tmp) @ dec_0_weight # 128x32
    J_y1m = J_y1m_ @ J_h2mul # 128x6

    y2 = y1m @ dec_2_weight.T + dec_2_bias
    J_y2 = dec_2_weight @ J_y1m # 3x3

    J_r = J_rvw[:9,:]
    J_y3_y2 = R # 3x3
    J_y3_r = np.stack((np.concatenate((y2, np.zeros(3,dtype=DTYPE), np.zeros(3,dtype=DTYPE))), 
                             np.concatenate((np.zeros(3,dtype=DTYPE), y2, np.zeros(3,dtype=DTYPE))),
                               np.concatenate((np.zeros(3,dtype=DTYPE), np.zeros(3,dtype=DTYPE), y2)),)) # 3x9
    J_y3 = J_y3_r @ J_r + J_y3_y2 @ J_y2

    return J_y3

@numba.jit(nopython=True, cache=True)
def pos_error(l1, v1, l2, t1, t2):
    return l2 - (l1 + v1*(t2-t1))

@numba.jit(nopython=True, cache=True)
def pos_jacobian(l1, v1, l2, t1, t2):
    return -np.eye(3, dtype=DTYPE), np.eye(3, dtype=DTYPE) * (t1-t2), np.eye(3, dtype=DTYPE)

@numba.jit(nopython=True, cache=True)
def vw_error(v1, w1, v2, w2, t1, t2):
    acc_v = aero_forward(v1, w1)
    error_v = v2 - (v1 + acc_v*(t2-t1))
    error_w = w2 - w1
    return np.concatenate((error_v, error_w))

@numba.jit(nopython=True, cache=True)
def vw_jacobian(v1, w1, v2, w2, t1, t2):
    J_acc_v1w1 = aero_jacobian(v1, w1)

    J_ev_v1 = -np.eye(3, dtype=DTYPE) + J_acc_v1w1[:,:3] * (t1-t2) 
    J_ew_v1 = np.zeros((3,3), dtype=DTYPE)

    J_ev_w1 =   J_acc_v1w1[:,3:] * (t1-t2)
    J_ew_w1 = -np.eye(3, dtype=DTYPE)

    J_ev_v2 = np.eye(3, dtype=DTYPE)
    J_ew_v2 = np.zeros((3,3), dtype=DTYPE)

    J_ev_w2 = np.zeros((3,3), dtype=DTYPE)
    J_ew_w2 = np.eye(3, dtype=DTYPE)

    return [np.concatenate((J_ev_v1, J_ew_v1),axis=0),
            np.concatenate((J_ev_w1, J_ew_w1),axis=0),
            np.concatenate((J_ev_v2, J_ew_v2),axis=0),
            np.concatenate((J_ev_w2, J_ew_w2),axis=0)]



# factors for gtsam
def assign_jacobians(jacobians: List[np.ndarray],J: List[np.ndarray]):
     for i, JJ in enumerate(J):
          jacobians[i] = JJ

class PriorFactor3(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1,mu):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            w1 = values.atVector(key1)
            error = w1 - mu
            if jacobians is not None:
                    jacobians[0] = np.eye(3, dtype=DTYPE)
            return error
        super().__init__(noiseModel, [key1], error_function) # may change to partial

class PositionFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1, key2, key3, t1, t2):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            l1, v1, l2 = values.atVector(key1), values.atVector(key2), values.atVector(key3)
            error = pos_error(l1,v1,l2,t1,t2)
            if jacobians is not None:
                    assign_jacobians(jacobians,pos_jacobian(l1,v1,l2,t1,t2))
            return error
        super().__init__(noiseModel, [key1, key2, key3], error_function) # may change to partial


class VWFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel,key1, key2, key3, key4, t1, t2):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            v1, w1, v2, w2 =  values.atVector(key1), values.atVector(key2), values.atVector(key3), values.atVector(key4)
            error = vw_error(v1,w1,v2,w2,t1,t2)
            if jacobians is not None:
                assign_jacobians(jacobians,vw_jacobian(v1,w1,v2,w2,t1,t2))
            return error
        super().__init__(noiseModel, [key1, key2, key3, key4], error_function) # may change to partial
