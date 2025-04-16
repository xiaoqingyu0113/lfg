import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from draw_util import draw_util

TRAJ_DATASET_PATH = Path("data/real/tennis_triangulated_spin")

DEBUG = True

def parse_filename(filename: Path):
    stem = filename.stem 
    # Regex pattern:
    # - spin_n1 or spin_p1 → capture sign and value
    # - vel_15 → capture velocity
    # - bag1 → capture bag number
    pattern = r"spin_([np])(\d+)_vel_(\d+)_bag(\d+)"
    match = re.match(pattern, stem)

    if match:
        sign = -1 if match.group(1) == 'n' else 1
        spin = sign * int(match.group(2))
        velocity = int(match.group(3))
        return spin, velocity
    else:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")


def read_single_traj_file(filename):
    '''
    read a single trajectory file and return the data
    '''
    data = np.loadtxt(filename)
    vw_ref = parse_filename(filename)
    return data, vw_ref

def read_traj_files(dirname):
    '''
    the res is a data frame with vw_ref from launcher
    '''
    
    traj_files = Path(dirname).glob('*.txt')
    traj_files_data = []
    for f in traj_files:
        data, vw_ref = read_single_traj_file(f)
        traj_files_data.append((data, vw_ref))
    return traj_files_data

def unroll_by_tid(traj_files_data):
    '''
    Unroll the data by trajectory id
    '''
    data = []
    

    g_tid = 0
    for (data_frame, vw_ref) in traj_files_data:
        print(f"processing {g_tid} trajectory")
        tid = 0
        data_by_tid = [g_tid, [data_frame[0]], vw_ref]
        for row in data_frame[1:]:
            if row[0] == tid:
                data_by_tid[1].append(row)
            else:
                data_by_tid[1] = np.array(data_by_tid[1]) # convert to numpy array
                data.append(data_by_tid)
                g_tid += 1
                tid = row[0]
                data_by_tid = [g_tid, [row], vw_ref]
    return data




def predict_vel_aero(v:torch.tensor ,w: torch.tensor, dt:float, cd:float = 0.001, cm:float = 0.0006):
    acc = -cd * v * torch.linalg.norm(v)  + cm * torch.linalg.cross(w,v) + torch.tensor([0, 0, -9.81])
    return v + acc * dt

def predict_bounce_roll(v1: torch.Tensor, w1: torch.Tensor, ez=0.85):
    r = 0.020  # radius in meters
    alpha = 0.2
    k_v = ez

    # A, B, C, D matrices
    A = torch.diag(torch.tensor([1 - alpha, 1 - alpha, -k_v]))
    
    B = torch.zeros(3, 3)
    B[0, 1] = alpha * r
    B[1, 0] = -alpha * r

    C = torch.zeros(3, 3)
    C[0, 1] = -1.5 * alpha / r
    C[1, 0] = 1.5 * alpha / r

    D = torch.diag(torch.tensor([1.0 - 1.5 * alpha, 1.0 - 1.5 * alpha, 1.0]))

    v_e = A @ v1 + B @ w1
    w_e = C @ v1 + D @ w1

    return v_e, w_e

class PhyxModel(torch.nn.Module):
    def __init__(self):
        super(PhyxModel, self).__init__()
        self.v0 = torch.nn.Parameter(torch.tensor([-0.01,  0.01  ,0.01]))
        self.w0 = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

    def forward(self,p0, time_stamps, z0 = 0.020):
        
        p_out = [p0]
        v = self.v0
        w = self.w0
        p = p0
        for i in range(len(time_stamps)-1):
            # print(f"p: {p[2]} v: {v[2]}")
            if p[2] < z0 and v[2] < 0:
                # print("bounce")
                v, w = predict_bounce_roll(v, w)
                p = p + v * (time_stamps[i+1] - time_stamps[i])
                p_out.append(p)
            else:
                dt = time_stamps[i+1] - time_stamps[i]
                v = predict_vel_aero(v, w, dt)
                p = p + v * dt
                p_out.append(p)
        
        return torch.stack(p_out, dim=0)


# PhysxModel = PhyxModel()

# pout = PhysxModel(torch.tensor([-18.24397 ,  0.0  ,0.6]), np.linspace(0, 2.0, 100))

# # print(pout)
# raise

if __name__ == '__main__':
    # traj_files_data = read_traj_files(TRAJ_DATASET_PATH)
    # unrolled_data = unroll_by_tid(traj_files_data)
    # tid1, data1, vw_ref1 = unrolled_data[8]

    single_file_data = read_single_traj_file(TRAJ_DATASET_PATH / 'spin_n2_vel_15_bag1.txt')


    tid = 0
    data, vw_ref = single_file_data
    data1 = data[data[:, 0].astype(int) == tid]  # select only the first trajectory
    data1  = data1[~np.isnan(data1).any(axis=1)]
    data1 = data1[:600,:]

    # # raise
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(data1[:,2], data1[:,3], data1[:,4], c='r', marker='o') 
 
    # ax.scatter(data1[:, 2], data1[:, 3], data1[:, 4], c='b', marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')    
    # draw_util.set_axes_equal(ax)
    # plt.show()
    # raise

    data1 = torch.tensor(data1)
    timestamps = data1[:,1]

    phyx_model = PhyxModel()
    phyx_model.train()
    
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(phyx_model.parameters(), lr=0.3)
   
    # training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        pout = phyx_model(data1[0,2:5], timestamps)
        loss = loss_fn(pout[:-1,:], data1[:-1,2:5])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            print(f"Curent est v0: {phyx_model.v0.detach().cpu().numpy()} | Curent est w0: {phyx_model.w0.detach().numpy()}")
    

    fig = plt.figure()
    data1 = data1.detach().cpu().numpy()
    pout = pout.detach().cpu().numpy()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data1[:,2], data1[:,3], data1[:,4], c='r', marker='o') 
 
    ax.scatter(data1[:, 2], data1[:, 3], data1[:, 4], c='b', marker='o')
    ax.plot(pout[:,0], pout[:,1], pout[:,2], c='r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')    
    draw_util.set_axes_equal(ax)
    plt.show()