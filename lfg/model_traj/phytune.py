import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class PhyTune(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        # C_d=0.1196, C_m=0.015, mu = 0.22, ez = 0.79

        # self._init_random()
        self._init_gt()

        self.recode = nn.Linear(3, 3)   

        # self.mode_1_linear = nn.Linear(1, 1)
        # self.mode_2_linear = nn.Linear(1, 1)
        # self.sigmoid = nn.Sigmoid()

        # self.bc_linear = nn.Sequential(nn.Linear(6, 6),
        #                                 nn.ReLU(),
        #                                 nn.Linear(6, 6),)

    def _init_gt(self):
        self.param1 =  nn.Parameter(torch.tensor([[0.1196]])) # C_d
        self.param2 =  nn.Parameter(torch.tensor([[0.015]])) # C_m
        self.param3 =  nn.Parameter(torch.tensor([[0.0,0.0,0.0]]))

        self.param4 = nn.Parameter(torch.tensor(0.22)) # mu
        self.param5 = nn.Parameter(torch.tensor(0.79)) # ez
        self.param6 = nn.Parameter(torch.tensor(0.020)) # r
    
    def _init_random(self):
        self.param1 =  nn.Parameter(torch.rand(1,1)*1e-4) # C_d
        self.param2 =  nn.Parameter(torch.rand(1,1)*1e-4) # C_m
        # self.param2 =  nn.Parameter(torch.tensor([[0.015]])) # C_m
        self.param3 =  nn.Parameter(torch.rand(1,3)*1e-4) # g

        self.param4 = nn.Parameter(torch.rand(1)*1e-4) # mu
        self.param5 = nn.Parameter(torch.rand(1)*1e-4) # ez
        self.param6 = nn.Parameter(torch.rand(1)*1e-4) # r

    def _compute_alpha(self, v, w):
        v_ix = v[..., 0:1]
        v_iy = v[..., 1:2]
        v_iz = v[..., 2:3]
        w_ix = w[..., 0:1]
        w_iy = w[..., 1:2]
        w_iz = w[..., 2:3]

        
        al = self.param4 * (1.0 + self.param5) * torch.abs(v_iz) / torch.sqrt((v_ix - w_iy * self.param6)**2 + (v_iy + w_ix * self.param6)**2 + 1e-6)
        al = torch.where(al > 0.4, 0.4, al)
        
        return al
    
    def _predict_bounce(self, V1, W1):
        v_i = V1
        w_i = W1
        r = self.param6


        alpha = self._compute_alpha(V1, W1)

        k_v = self.param5.repeat(*alpha.size())
        
        A_diag = torch.cat((1.0 - alpha, 1.0 - alpha, -k_v), dim=-1)

        A = torch.diag_embed(A_diag)
   
   
        B_top = torch.cat([torch.zeros(v_i.size(0), 1, 1, device=v_i.device),
                       (alpha * r).view(v_i.size(0), 1, 1),
                       torch.zeros(v_i.size(0), 1, 1, device=v_i.device)], dim=2)
        
        B_middle = torch.cat([(alpha * -r).view(v_i.size(0), 1, 1),
                            torch.zeros(v_i.size(0), 1, 1, device=v_i.device),
                            torch.zeros(v_i.size(0), 1, 1, device=v_i.device)], dim=2)
        
        B_bottom = torch.zeros(v_i.size(0), 1, 3, device=v_i.device)
        B = torch.cat([B_top, B_middle, B_bottom], dim=1)

        
        C_top = torch.cat([torch.zeros(v_i.size(0), 1, 1, device=v_i.device),
                       (-1.5 * alpha / r).view(v_i.size(0), 1, 1),
                       torch.zeros(v_i.size(0), 1, 1, device=v_i.device)], dim=2)
        
        C_middle = torch.cat([(1.5 * alpha / r).view(v_i.size(0), 1, 1),
                            torch.zeros(v_i.size(0), 1, 2, device=v_i.device)], dim=2)
        
        C_bottom = torch.zeros(v_i.size(0), 1, 3, device=v_i.device)
        C = torch.cat([C_top, C_middle, C_bottom], dim=1)

        D_diag = torch.cat((1.0 - 1.5 * alpha, 1.0 - 1.5 * alpha, torch.ones_like(alpha)), dim=-1)
        D = torch.diag_embed(D_diag)
        
        # print(A.shape, B.shape, C.shape, D.shape)
        # print(v_i.shape, w_i.shape)
        if A.ndim != B.ndim:
            B = B.unsqueeze(1)
            C = C.unsqueeze(1)
        if v_i.ndim +1 == A.ndim:
            # print('pre shape', v_i.shape, w_i.shape)
            v_i = v_i.unsqueeze(-1)
            w_i = w_i.unsqueeze(-1)
            # print('after shape', v_i.shape, w_i.shape)
            # print(A.shape, B.shape, C.shape, D.shape)
            v_e = A@v_i + B@w_i
            w_e = C@v_i + D@w_i

            v_e = v_e.squeeze(-1)
            w_e = w_e.squeeze(-1)
            # print('post shape', v_e.shape, w_e.shape)

        else:

            v_e = A@v_i + B@w_i
            w_e = C@v_i + D@w_i


        return v_e, w_e


    def forward(self, b, v, w, dt):
        '''
        input can be [b,3] or [b,1,3]
        '''

        # check bounce
        # set ground truth bouncing condition

        # w = self.recode(w)

        condition1 = b < 0.010
        condition2 = v[..., 2:3] < 0.0
        condition = torch.logical_and(condition1, condition2)
        # print(torch.any(condition))
        # vw_bc = self.bc_linear(torch.cat([v, w], dim=-1))
        v_bc, w_bc = self._predict_bounce(v, w)

        

        v =   torch.where(condition, v_bc, v)
        w =   torch.where(condition, w_bc, w)

        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        cross_vw = torch.linalg.cross(w, v)
        acc_mode_1 = -self.param1 * norm_v*v + self.param2 * cross_vw + self.param3 + torch.tensor([0.0, 0.0, -9.81], device=DEVICE).view(1,3)


        v_new = v + acc_mode_1 * dt
        w_new = w
        

        # soft switch
        # gate1 = self.sigmoid(self.mode_1_linear(b)+ 10.0)
        # gate2 = 1.0 - gate1 # special case for softmax
        # v_new = gate1 *(v + acc_mode_1*dt) + gate2 * vw_mode_2[..., :3]
        # w_new = gate1 * w + gate2 * vw_mode_2[..., 3:]

        # hard switch
        # condition = gate1 > gate2
        # v_new = torch.where(condition, v + acc_mode_1 * dt, vw_mode_2[..., :3])
        # w_new = torch.where(condition, w, vw_mode_2[..., 3:])

        return v_new, w_new
    



def euler_updator(p, v, dt):
    return p + v*dt

def autoregr_PhyTune(data, model, est, cfg):
    '''
    data = [b, seq_len, 11]
    11: [traj_idx, time_stamp, p_xyz, v_xyz, w_xyz]
    '''
    tN = data[:,:, 1:2]
    pN = data[:,:, 2:5]
    p0 = pN[:, 0:1, :]
    v0 = data[:, 0:1, 5:8]
    w0 = data[:, 0:1, 8:11]


    w0 = model.recode(w0) # launcher to actual spin

    if est is not None:
        p0, v0, w0 = est(data[:,:est.size,1:5], w0=w0)
    
    

    d_tN = torch.diff(tN, dim=1)
    
    pN_est = [p0]
    for i in range(1, data.shape[1]):
        dt = d_tN[:, i-1:i, :]
        b0 = p0[:,:,2:3]
        p0 = euler_updator(p0, v0, dt)
        v0, w0 = model(b0, v0, w0, dt)
        pN_est.append(p0)

    pN_est = torch.cat(pN_est, dim=1)
    return pN_est
