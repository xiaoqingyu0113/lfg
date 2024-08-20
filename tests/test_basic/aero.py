import torch 
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bounce import TestModel2 as BC_Model

device = torch.device("cuda:0")
torch.manual_seed(0)

def generate_data():
    N = 400
    vx = torch.rand(N, 1) * 10.0 -5.0
    vy = torch.rand(N, 1) * 10.0 -5.0
    vz = torch.rand(N, 1) * 10.0 -5.0
    wx = torch.rand(N, 1) * 60.0 * torch.pi*2 - 30.0 * torch.pi * 2
    wy = torch.rand(N, 1) * 60.0 * torch.pi*2 - 30.0 * torch.pi * 2
    wz = torch.rand(N, 1) * 60.0 * torch.pi*2 - 30.0 * torch.pi * 2

    return torch.cat([vx, vy, vz, wx, wy, wz], dim=1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = generate_data().to(device)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    @classmethod
    def get_loader(cls, batch_size=16, shuffle=True):
        dataset = cls()
        # split dataset into train and test
        train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        return train_loader, test_loader



class TestModel2(nn.Module):
    def __init__(self):
        super(TestModel2, self).__init__()
        hidden_size = 32
        self.layer1 = nn.Sequential(
            nn.Linear(9, hidden_size),
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
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, v, w, dt):
        v = v / 5.0
        w = w / (30.0*torch.pi*2)

        x = torch.cat([v, w, torch.linalg.cross(w,v)], dim=1)
        h0 = self.layer1(x)
        h = h0
        h =  self.layer2(h)*h0 
        h =  self.layer3(h)*h0 
        # h = self.layer4(h)*h0 
        x = self.dec(h) + torch.tensor([[0.0, 0.0, -9.81]], device=v.device)
        return (v  + x * dt) * 5.0
    
class TestModel3(nn.Module):
    def __init__(self):
        super(TestModel3, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(6, 64), nn.LeakyReLU())
        self.layer_2 = nn.Sequential(nn.Linear(6, 64), nn.LeakyReLU())

        self.dec = nn.Sequential(nn.Linear(64, 3))

    def forward(self, v, w, dt):
        v = v / 5.0
        w = w / (30.0*torch.pi*2)

        x = torch.cat([v, w], dim=1)
        x1 = self.layer_1(x)
        x2 = self.layer_2(x)
        x = x1 * x2
        x = self.dec(x)+ torch.tensor([[0.0, 0.0, -9.81]], device=v.device)
        
        return (v  + x * dt)*5.0

def gram_schmidth(v, w):

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
  
  
    v_local = torch.matmul(RT,v).squeeze(-1)
    w_local = torch.matmul(RT, w).squeeze(-1)
    
    return R, v_local, w_local

class TestModel4(nn.Module):
    def __init__(self):
        super(TestModel4, self).__init__()
 
        self.layer_gt = nn.Linear(3,3)
        self.layer_gt.weight = nn.Parameter(torch.tensor([[-0.0, 0.0, 0.0],
                                                          [0.0, 0.0, 0.0],
                                                          [-0.0, 0.0, 0.0]], device=device))
        self.bias = nn.Parameter(torch.tensor([[0.0, 0.0, -9.8]], device=device))

    def forward(self, v, w, dt):
        v_normalize = v 
        w_normalize = w

        R, v_local, w_local = gram_schmidth(v_normalize, w_normalize)     

        feat = torch.cat([v_local[...,:1], w_local[...,:2]], dim=-1)
        h = self.layer_gt(feat)

        y = h * feat
        y =torch.matmul(R, y.unsqueeze(-1)).squeeze(-1)       
     
        y = y + self.bias #+  torch.tensor([[0.0, 0.0, -9.81]]).to(v.device)

        out = v + y * dt 
    
        return out

class TestModel5(nn.Module):
    def __init__(self):
        super(TestModel5, self).__init__()
        hidden_size= 32
        self.recode = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
            )
        
        self.layer1 = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 3)
        )

        
        self.bias = nn.Parameter(torch.tensor([[0.0, 0.0, -9.8]]))
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.normal_(m.bias, mean=0, std=1e-4)
    def forward(self, v, w, dt):
        v_normalize = v 
        w_normalize = w
        w_normalize = self.recode(w_normalize)

        R, v_local, w_local = gram_schmidth(v_normalize, w_normalize)     

        feat = torch.cat([v_local[...,:1], w_local[...,:2]], dim=-1)
        h = self.layer1(feat)
        h  = self.layer2(h)*h

        y = self.dec(h)
        y =torch.matmul(R, y.unsqueeze(-1)).squeeze(-1)       
     
        y = y + self.bias #+  torch.tensor([[0.0, 0.0, -9.81]]).to(v.device)

        out = v + y * dt 
    
        return out
    
def compute_gt(v,w, dt):
    cd = 0.1123
    cm = 0.015
    # cd = 1.0
    # cm = 1.0


    g = torch.tensor([0.0, 0.0, -9.81], device=v.device)
    acc_f =  - cd * v * torch.linalg.norm(v,dim=-1, keepdim=True)
    acc_m = cm * torch.linalg.cross(w, v)
    # acc = g + acc_m + acc_f
    acc = acc_f + acc_m + g
    v_new = v + acc * dt
    return v_new

def batch_rotation_matrix(th):
    """
    th: (N, 3)
    rot_matrix: (N, 3, 3)
    """
    N = th.shape[0]
    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    zeros = torch.zeros_like(th[:, 0])
    ones = torch.ones_like(th[:, 0])
    rot_matrix = torch.stack([
        torch.stack([cos_th[:, 1]*cos_th[:, 2], -cos_th[:, 1]*sin_th[:, 2], sin_th[:, 1]], dim=1),
        torch.stack([sin_th[:, 0]*sin_th[:, 1]*cos_th[:, 2] + cos_th[:, 0]*sin_th[:, 2], -sin_th[:, 0]*sin_th[:, 1]*sin_th[:, 2] + cos_th[:, 0]*cos_th[:, 2], -sin_th[:, 0]*cos_th[:, 1]], dim=1),
        torch.stack([-cos_th[:, 0]*sin_th[:, 1]*cos_th[:, 2] + sin_th[:, 0]*sin_th[:, 2], cos_th[:, 0]*sin_th[:, 1]*sin_th[:, 2] + sin_th[:, 0]*cos_th[:, 2], cos_th[:, 0]*cos_th[:, 1]], dim=1)
    ], dim=1)
    return rot_matrix

def augment_data(data):
    v = data[:, :3]
    w = data[:, 3:]
    th = torch.rand_like(v) * 2 * torch.pi - torch.pi
    th[:,0:2] = 0.0
    rot_matrix = batch_rotation_matrix(th)
    v_new = torch.einsum("ijk,ik->ij", rot_matrix, v)
    w_new = torch.einsum("ijk,ik->ij", rot_matrix, w)
    return torch.cat([v_new, w_new], dim=1)


def euler_update(p, v, dt):
    p = p + v * dt
    return p 

def euler_integrate(model, v0, w0, tspan):
    t = torch.linspace(0, tspan, 100).to(v0.device)
    dt = torch.diff(t, dim=-1).squeeze(-1)

    p = torch.zeros_like(v0).to(v0.device)
    p_his = [p]
    for i in range(99):
        v = model(v0, w0, dt[i])
        p = euler_update(p, v, dt[i])
        p_his.append(p)
    p_his = torch.stack(p_his, dim=1)
    return p_his

def train_loop(task='train'):
    train_loader, test_loader = Dataset.get_loader(batch_size=64, shuffle=True)
    model = TestModel5()
    model.to(device)

    if task == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        for epoch in range(1000):
            model.train()
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()

                # for param in model.parameters():
                #     if param.grad is not None:
                #         param.grad.zero_()

                # data = augment_data(data)
                v = data[:, :3]
                w = data[:, 3:]
                
                p_pred = euler_integrate(model, v, w, 2.0)
                p_gt = euler_integrate(compute_gt, v, w, 2.0)
                loss = criterion(p_pred, p_gt)

                loss.backward()
                # print(f"Loss: {loss.item()}")
                
                optimizer.step()

                # update gradient and prevent overflow of both gradient and data
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         param.grad = torch.where((param.grad < 1e-8) & (param.grad> -1e-8), torch.zeros_like(param.grad), param.grad)
                #         param.data = param.data - 1e-5 * param.grad

                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient of {name}: \n{param.grad}")

                for name, param in model.named_parameters():
                    if 'dec' in name:
                        print(f"New data of {name}: \n{param.data}")    
                    # print(f"New data of {name}: \n{param.data}")

                if i % 1 == 0:
                    print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")

                print('--------')
                
        torch.save(model.state_dict(), 'data/archive/aero_model.pth')

    if task == 'test':
        model.load_state_dict(torch.load('data/archive/aero_model.pth'))
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                v = data[:, :3]
                w = data[:, 3:]

                # dt = torch.rand(1).item() + 1.0
                # v_new_gt = compute_gt(v, w, dt)
                # v_new = model(v, w, dt)
                p_pred = euler_integrate(model, v, w, 2.0)
                p_gt = euler_integrate(compute_gt, v, w, 2.0)
                v_all = torch.cat([p_pred[:,-1,:], p_gt[:,-1,:]], dim=-1)
                
                v0 = torch.cat([v,w],dim=-1)

                print(p_pred.shape)
                print(p_gt.shape)
                print(v_all.shape)
                print(v_all[:5])
                print(v0[:5])
                criterion = nn.MSELoss()
                loss = criterion(p_gt, p_pred)
                print(f"Test Loss: {loss.item()}")

                
                break

if __name__ == '__main__':
    train_loop(task='train')