import torch
import torch.nn as nn

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gram_schmidth_2d(v2d, w2d):
    '''
    v2d, w2d should be in shape either [b,2] or [b,1,2]

    output:
    - R2d_corrected: [b, 2, 2]
    - v2d_local: [b, 2] or [b, 1, 2]
    - w2d_local: [b, 2] or [b, 1, 2]

    '''

    shapes = v2d.shape
    batch_size = shapes[0]

    v2d = v2d.reshape(batch_size, 2)
    w2d = w2d.reshape(batch_size, 2)

    # Normalize the first vector
    v_orthonormal = v2d / (torch.linalg.vector_norm(v2d,dim=-1, keepdim=True) + 1e-8)

    # Project w onto v_orthonormal and subtract to make w orthogonal to v
    proj = torch.linalg.vecdot(w2d, v_orthonormal).unsqueeze(-1) * v_orthonormal
    w_orthogonal = w2d - proj
    
    # Normalize the second vector
    w_orthonormal = w_orthogonal / (torch.linalg.vector_norm(w_orthogonal, dim=-1, keepdim=True) + 1e-8)
    

    R2d = torch.stack((v_orthonormal, w_orthonormal), dim=-1)  # Shape: (batch_size, 2, 2)

    # Calculate the determinants for each batch
    determinants = torch.linalg.det(R2d)  # Shape: (batch_size,)
    negative_dets = determinants < 0
    w_orthonormal_corrected = torch.where(negative_dets.unsqueeze(-1), -w_orthonormal, w_orthonormal)
    R2d_corrected = torch.stack((v_orthonormal, w_orthonormal_corrected), dim=-1)  # Shape: (batch_size, 2, 2)

    # print(f"det R2d : {torch.det(R2d)}")
    RT2d = R2d_corrected.transpose(-1, -2)

    v2d = v2d.unsqueeze(-1)
    w2d = w2d.unsqueeze(-1)

    v2d_local = torch.matmul(RT2d, v2d).squeeze(-1)
    w2d_local = torch.matmul(RT2d, w2d).squeeze(-1)

    v2d_local = v2d_local.reshape(shapes)
    w2d_local = w2d_local.reshape(shapes)

    return R2d_corrected, v2d_local, w2d_local

def gram_schmidth(v, w):
    '''
    v, w should be in shape either [b,3] or [b,1,3]

    output:
    - R: [b, 3, 3]
    - v_local: [b, 3] or [b, 1, 3]
    - w_local: [b, 3] or [b, 1, 3]
    '''
    shapes = v.shape
    batch_size = shapes[0]

    v = v.reshape(batch_size, 3)
    w = w.reshape(batch_size, 3)

    # Normalize the first vector
    v_orthonormal = v / (torch.linalg.vector_norm(v,dim=-1, keepdim=True) + 1e-8)

    # Project w onto v_orthonormal and subtract to make w orthogonal to v
    proj = torch.linalg.vecdot(w, v_orthonormal).unsqueeze(-1) * v_orthonormal
    w_orthogonal = w - proj

    # Normalize the second vector
    w_orthonormal = w_orthogonal / (torch.linalg.vector_norm(w_orthogonal, dim=-1, keepdim=True) + 1e-8)


    # Compute the third orthonormal vector using the cross product
    u_orthonormal = torch.linalg.cross(v_orthonormal, w_orthonormal)

    # Construct the rotation matrix R
    R = torch.stack((v_orthonormal, w_orthonormal, u_orthonormal), dim=-1)

    
    RT  = R.transpose(-1, -2)
    # Compute the local frame coordinates
    
    v = v.unsqueeze(-1)
    w = w.unsqueeze(-1)
  
    v_local = torch.matmul(RT,v)
    w_local = torch.matmul(RT, w)
    v_local = v_local.reshape(shapes)
    w_local = w_local.reshape(shapes)
    
    return R, v_local, w_local

class BounceModel(nn.Module):
    def __init__(self):
        super(BounceModel, self).__init__()
        hidden_size=32

        # self.recode = nn.Sequential(
        #     nn.Linear(3, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 3)
        #     )
        
        self.recode = nn.Linear(3,3)


        self.layer1 = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 6)
        )
        
    def forward(self, v, w):
        '''
        v, w should be in shape either [b,3] or [b,1,3]
        '''

        shapes = v.shape
        batch_size = shapes[0]

        v = v.reshape(batch_size, 3)
        w = w.reshape(batch_size, 3)
        # if self.recode:
        w = self.recode(w)

        R2d, v2d_local, w2d_local = gram_schmidth_2d(v[...,:2], w[...,:2])     

        v_local = torch.cat([v2d_local, v[...,2:3]], dim=-1)
        w_local = torch.cat([w2d_local, w[...,2:3]], dim=-1)

        # normalize
        v_normalize = v_local / 3.0
        w_normalize = w_local / 7.0

        x = torch.cat([v_normalize, w_normalize], dim=-1)
        # x = self.layer1(x)
        # x = x + self.layer2(x)*x
        # x = x + self.layer3(x)*x
        h0 = self.layer1(x)
        h1 = self.layer2(h0) + h0
        h2 = self.layer3(h1) + h1
        x = self.dec(h2)
        # x = self.dec(x)

        # unnormalize
        v2d_local_new = x[..., :2] * 3.0
        vz_new = x[..., 2:3] * 3.0
        w2d_local_new = x[..., 3:5] * 7.0
        wz_new = x[..., 5:6] * 7.0

        v2d_new = torch.matmul(R2d, v2d_local_new.unsqueeze(-1)).squeeze(-1)
        w2d_new = torch.matmul(R2d, w2d_local_new.unsqueeze(-1)).squeeze(-1)

        v_new = torch.cat([v2d_new, vz_new], dim=-1)
        w_new = torch.cat([w2d_new, wz_new], dim=-1) 

        v_new = v_new.reshape(shapes)
        w_new = w_new.reshape(shapes)

        return v_new, w_new
    
class AeroModel(nn.Module):
    def __init__(self):
        super(AeroModel, self).__init__()
        hidden_size= 32

        # self.recode = nn.Sequential(
        #     nn.Linear(3, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 3)
        #     )
        self.is_recode = True
        self.recode = nn.Linear(3,3)

        self.layer1 = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

        
        self.bias = nn.Parameter(torch.tensor([[0.0, 0.0, -9.81]]))
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.normal_(m.bias, mean=0, std=1e-4)
    def forward(self, v, w):
        '''
        v, w should be in shape either [b,3] or [b,1,3]
        '''

        shapes = v.shape
        batch_size = shapes[0]
        v = v.reshape(batch_size, 3)
        w = w.reshape(batch_size, 3)
        w = self.recode(w)

        v_normalize = v 
        w_normalize = w

        R, v_local, w_local = gram_schmidth(v_normalize, w_normalize)     

        feat = torch.cat([v_local[...,:1], w_local[...,:2]], dim=-1)
        h = self.layer1(feat)
        h  = self.layer2(h) + h
        h = self.layer3(h) + h

        y = self.dec(h)
        y =torch.matmul(R, y.unsqueeze(-1)).squeeze(-1)       
     
        y = y + self.bias #+  torch.tensor([[0.0, 0.0, -9.81]]).to(v.device)
        y = y.reshape(shapes)
    
        return y
    
class Skip(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.z0 = 0.010
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.bc_layer = BounceModel()
        self.aero_layer = AeroModel()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    
    def forward(self, b, v, w, dt):
        '''
        input can be [b,3] or [b,1,3]
        '''

        # check bounce
        condition1 = b < self.z0
        condition2 = v[..., 2:3] < 0.0
        condition = torch.logical_and(condition1, condition2)

        v_bc , w_bc= self.bc_layer(v,w)
        v =   torch.where(condition, v_bc, v)
        w =   torch.where(condition, w_bc, w)

        acc = self.aero_layer(v,w)

        v_new = v + acc * dt
        w_new = w
        
        return v_new, w_new
    

def euler_updator(p, v, dt):
    return p + v*dt

def autoregr_Skip(data, model, est, cfg):
    '''
    data = [b, seq_len, 11]
    11: [traj_idx, time_stamp, p_xyz, v_xyz, w_xyz]
    '''
    tN = data[:,:, 1:2]
    pN = data[:,:, 2:5]
    p0 = pN[:, 0:1, :]
    v0 = data[:, 0:1, 5:8]
    w0 = data[:, 0:1, 8:11]

    # print(f"p0_gt : {p0[0]}")
    # print(f"v0_gt : {v0[0]}")
    # print(f"w0_gt : {w0[0]}")

    if est is not None:
        p0, v0, w0 = est(data[:,:est.size,1:5], w0=w0)
        # failed to estimate
        if p0 is None:
            return None
        # print(f"p0_est : {p0[0]}")
        # print(f"v0_est : {v0[0]}")
        # print(f"w0_est : {w0[0]}")    

    # raise
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
