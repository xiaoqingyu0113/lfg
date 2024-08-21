import sys
import os
sys.path.append(os.path.abspath('tests/test_synthetic_training/synthetic'))
sys.path.append(os.path.abspath('tests/test_synthetic_training'))

from dynamics import (bounce_roll_spin_forward,
                            bounce_roll_velocity_forward, 
                            bounce_slide_spin_forward, 
                            bounce_slide_velocity_forward, 
                            compute_alpha)
import torch
import numpy as np
from typing import Tuple
import torch.nn as nn





def generate_bounce_data(N: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = 0.22
    ez = 0.79
    vx = np.random.rand(N) * 10.0 - 5.0
    vy = np.random.rand(N)* 10.0 - 5.0
    vz = -np.random.rand(N) * 5.0 - 0.2

    wx = np.random.rand(N) * 30.0*np.pi*2 - 15.0*np.pi*2
    wy = np.random.rand(N) * 30.0*np.pi*2 - 15.0*np.pi*2
    wz = np.random.rand(N) * 30.0*np.pi*2 - 15.0*np.pi*2

    v_new_N = []
    w_new_N = []
    for i in range(N):
        v = np.array([vx[i], vy[i], vz[i]])
        w = np.array([wx[i], wy[i], wz[i]])
        alpha = compute_alpha(v,w,[mu,ez])
        if alpha < 0.4:
            v_new = bounce_slide_velocity_forward(v,w,[mu,ez]).flatten()
            w_new = bounce_slide_spin_forward(v,w,[mu,ez]).flatten()
        else:
            v_new = bounce_roll_velocity_forward(v,w,[mu,ez]).flatten()
            w_new = bounce_roll_spin_forward(v,w,[mu,ez]).flatten()

        v_new_N.append(v_new)
        w_new_N.append(w_new)

    v_new_N = np.array(v_new_N)
    w_new_N = np.array(w_new_N)

    v_N = np.concatenate([vx.reshape(-1,1), vy.reshape(-1,1), vz.reshape(-1,1)], axis=1)
    w_N = np.concatenate([wx.reshape(-1,1), wy.reshape(-1,1), wz.reshape(-1,1)], axis=1)
    return v_N, w_N, v_new_N, w_new_N


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.linear_v = nn.Linear(4, 128)
        self.linear_w = nn.Linear(4, 128)
        self.dec = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        self.relu = nn.ReLU()

    def forward(self, v, w):
        v = torch.cat([v, torch.ones_like(v[:, :1])], dim=1)
        w = torch.cat([w, torch.ones_like(w[:, :1])], dim=1)

        h1 = self.relu(self.linear_v(v))
        h2 = self.relu(self.linear_w(w))
        # h = torch.cat([h1, h2], dim=1)
        h = h1 * h2

        h = self.dec(h)
        return h[:, :3], h[:, 3:]

class TestModel2(nn.Module):
    def __init__(self):
        super(TestModel2, self).__init__()
        hidden_size=32
        self.layer1 = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 6)
        )

    def forward(self, v, w):
        v = v / 3.0
        w = w / (30.0*np.pi*2)

        x = torch.cat([v, w], dim=1)
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = self.dec(x)

        v_new = x[:, :3] * 3.0
        w_new = x[:, 3:] * (30.0*np.pi*2)
        return v_new, w_new

def gram_schmidth_2d(v2d, w2d):
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

    return R2d_corrected, v2d_local, w2d_local

class TestModel3(nn.Module):
    def __init__(self):
        super(TestModel3, self).__init__()
        hidden_size=32

        self.recode = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
            )
        
        self.layer1 = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 6)
        )
        
    def forward(self, v, w):
        
        w = self.recode(w)
        R2d, v2d_local, w2d_local = gram_schmidth_2d(v[...,:2], w[...,:2])     
        # x = torch.cat([v2d_local[...,:2], v_normalize[...,2:], w2d_local,  w_normalize[...,2:]], dim=-1)

        v_local = torch.cat([v2d_local, v[...,2:3]], dim=-1)
        w_local = torch.cat([w2d_local, w[...,2:3]], dim=-1)

        # print(f"model3 input v local : {v_local}")
        # print(f"model3 input w local : {w_local}")

        v_normalize = v_local / 3.0
        w_normalize = w_local / (30.0*np.pi*2)

        x = torch.cat([v_normalize, w_normalize], dim=-1)
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = self.dec(x)

        
        v2d_local_new = x[:, :2] * 3.0
        vz_new = x[:, 2:3] * 3.0
        w2d_local_new = x[:, 3:5] * (30.0*np.pi*2)
        wz_new = x[:, 5:6] * (30.0*np.pi*2)

        # print(f"model3 output v local new : {v2d_local_new}")
        # print(f"model3 output w local new : {w2d_local_new}")

        v2d_new = torch.matmul(R2d, v2d_local_new.unsqueeze(-1)).squeeze(-1)
        w2d_new = torch.matmul(R2d, w2d_local_new.unsqueeze(-1)).squeeze(-1)

        v_new = torch.cat([v2d_new, vz_new], dim=-1)
        w_new = torch.cat([w2d_new, wz_new], dim=-1) 

            
        return v_new, w_new


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, v_N, w_N, v_new_N, w_new_N):
        self.v_N = v_N
        self.w_N = w_N
        self.v_new_N = v_new_N
        self.w_new_N = w_new_N

    def __len__(self):
        return len(self.v_N)

    def __getitem__(self, idx):
        return self.v_N[idx], self.w_N[idx], self.v_new_N[idx], self.w_new_N[idx]

def test_bounce_training():
    v_N, w_N, v_new_N, w_new_N = generate_bounce_data(N=1000)
    v_N = torch.tensor(v_N, dtype=torch.float32)
    w_N = torch.tensor(w_N, dtype=torch.float32)
    v_new_N = torch.tensor(v_new_N, dtype=torch.float32)
    w_new_N = torch.tensor(w_new_N, dtype=torch.float32)

    v_N = v_N.to('cuda')
    w_N = w_N.to('cuda')
    v_new_N = v_new_N.to('cuda')
    w_new_N = w_new_N.to('cuda')

    model = TestModel3()
    model = model.to('cuda')
    # model.load_state_dict(torch.load('bounce_model.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = MyDataset(v_N, w_N, v_new_N, w_new_N)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.L1Loss()
    for epoch in range(500):
        for v_N, w_N, v_new_N, w_new_N in dataloader: 
            optimizer.zero_grad()
            v_new_pred, w_new_pred = model(v_N, w_N)
            loss = criterion(v_new_pred, v_new_N)*100 + criterion(w_new_pred, w_new_N)
            loss.backward()
            optimizer.step()
            print(f'loss: {loss.item()}')

    print('-------------------')
    for i in range(len(v_new_N)):
        v_new_pred, w_new_pred = model(v_N[i:i+1], w_N[i:i+1])
        print('v_new_pred:', v_new_pred.detach())
        print('v_new_N:', v_new_N[i])
        print('w_new_pred:', w_new_pred.detach())
        print('w_new_N:', w_new_N[i])
        print('-------------------')

    torch.save(model.state_dict(), 'data/archive/bounce_model.pth')


def compare():
    v_N, w_N, v_new_N, w_new_N = generate_bounce_data(N=1000)
    v_N = torch.tensor(v_N, dtype=torch.float32)
    w_N = torch.tensor(w_N, dtype=torch.float32)
    v_new_N = torch.tensor(v_new_N, dtype=torch.float32)
    w_new_N = torch.tensor(w_new_N, dtype=torch.float32)

    v_N = v_N.to('cuda')
    w_N = w_N.to('cuda')
    v_new_N = v_new_N.to('cuda')
    w_new_N = w_new_N.to('cuda')

    model = TestModel3()
    model.load_state_dict(torch.load('data/archive/bounce_model.pth'))
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = MyDataset(v_N, w_N, v_new_N, w_new_N)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.L1Loss()

    print('-------------------')
    for i in range(len(v_new_N)):
        print(f"input v : {v_N[i]}")
        print(f"input w : {w_N[i]}")
        v_new_pred, w_new_pred = model(v_N[i:i+1], w_N[i:i+1])
        print('v_new_pred:', v_new_pred.detach())
        print('v_new_N:', v_new_N[i])
        print('w_new_pred:', w_new_pred.detach())
        print('w_new_N:', w_new_N[i])
        print('-------------------')

def tmp():
    '''
    input v : tensor([-3.0179, -2.4999, -3.1978], device='cuda:0')
    input w : tensor([ 77.7324,  88.7545, -63.0044], device='cuda:0')
    v_new_pred: tensor([[-2.6993, -1.3501,  2.5300]], device='cuda:0')
    v_new_N: tensor([-1.7824, -2.2563,  2.5262], device='cuda:0')
    w_new_pred: tensor([[-12.8132, 113.1073, -63.3327]], device='cuda:0')
    w_new_N: tensor([ 96.0074,  -3.9071, -63.0044], device='cuda:0')
    '''
    v = torch.tensor([ 4.3418,  3.6620, -3.5193], dtype=torch.float32)
    w = torch.tensor([3.0748, -65.9051, -61.9739], dtype=torch.float32)

    R, v_local, w_local = gram_schmidth_2d(v[:2], w[:2])


    
    v_local = torch.cat([v_local, v[2:3]], dim=-1)
    w_local = torch.cat([w_local, w[2:3]], dim=-1)
  
    model = TestModel3()
    model.load_state_dict(torch.load('data/archive/bounce_model.pth'))

    v_new, w_new = model(v.unsqueeze(0), w.unsqueeze(0)) 


    model2 = TestModel2()
    model2.load_state_dict(torch.load('data/archive/bounce_model.pth'))
    # print(f"model 2 input v : {v_local}")
    # print(f"model 2 input w : {w_local}")

    v_new, w_new = model2(v_local.unsqueeze(0), w_local.unsqueeze(0))
    
    # print(f"model 2 output v : {v_new}")
    # print(f"model 2 output w : {w_new}")


    v = v_local.cpu().numpy()
    w = w_local.cpu().numpy()
    R = R.cpu().numpy()

    def rotz(th):
        return np.array([[np.cos(th), -np.sin(th), 0],
                         [np.sin(th), np.cos(th), 0],
                         [0, 0, 1]])

    th = np.arctan2(R[1,0], R[0,0])
    R = np.block([[R, np.zeros((2,1))], [np.zeros((1,2)), np.array([[1]])]])
    # print(R)
    # print(rotz(th))



    def bounce_gt(v,w):
        mu = 0.22
        ez = 0.79
        alpha = compute_alpha(v,w,[mu,ez])
        print(f"alpha : {alpha}")
        if alpha < 0.4:
            v_new = bounce_slide_velocity_forward(v,w,[mu,ez]).flatten()
            w_new = bounce_slide_spin_forward(v,w,[mu,ez]).flatten()
        else:
            v_new = bounce_roll_velocity_forward(v,w,[mu,ez]).flatten()
            w_new = bounce_roll_spin_forward(v,w,[mu,ez]).flatten()

        return v_new, w_new
    
    
    
    # R = rotz(th)
    # print(R)
    v_new, w_new = bounce_gt(v,w)
    print(f"ground truth v : {v_new}")
    print(f"ground truth w : {w_new}")

    v_new2, w_new2 = bounce_gt(R.T@v, R.T@w)
    print(f"ground truth v : {R@v_new2}")
    print(f"ground truth w : {R@w_new2}")


if __name__ == '__main__':
    test_bounce_training()
    # compare()
    # tmp()