import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from draw_util import draw_util
import tqdm

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




def predict_vel_aero(v:torch.tensor ,w: torch.tensor, dt:float, cd:float = 0.0191, cm:float = 0.001):
    acc = -cd * v * torch.linalg.norm(v)  + cm * torch.linalg.cross(w,v) + torch.tensor([0, 0, -9.81])
    # print("w", w)
    # print("v", v)
    # print("magnus", cm * torch.linalg.cross(w,v))
    # print("drag", -cd * v * torch.linalg.norm(v))
    return v + acc * dt





def predict_bounce_roll(v1: torch.Tensor, w1: torch.Tensor, ez=0.9):
    r = 0.020  # radius in meters
    alpha = 0.05
    k_v = ez

    # A, B, C, D matrices
    A = torch.diag(torch.tensor([1 - alpha-0.3, 1 - alpha-0.3, -k_v]))
    
    B = torch.zeros(3, 3)
    B[0, 1] = alpha * r
    B[1, 0] = -alpha * r

    C = torch.zeros(3, 3)
    C[0, 1] = -1.5 * alpha / r
    C[1, 0] = 1.5 * alpha / r

    D = torch.diag(torch.tensor([1.0 - 1.5 * alpha, 1.0 - 1.5 * alpha, 1.0]))

    v_e = A @ v1 + B @ w1
    w_e = C @ v1*0.2 + D @ w1

    # print("bouce")
    return v_e, w_e

    



def estimate_velocity(points: torch.Tensor, timestamps: torch.Tensor):
    """
    Estimates constant velocity from 3D points and timestamps.
    Args:
        points: (N, 3) tensor of x, y, z positions.
        timestamps: (N,) tensor of timestamps.
    Returns:
        velocity: (3,) tensor of estimated velocities [v_x, v_y, v_z].
    """

    def estimate_initial_velocity(t: torch.Tensor, y: torch.Tensor, g: float = 9.81):
        """
        Estimates initial vertical velocity (v0) given known gravity (g) and initial height (y0).
        Args:
            t: 1D tensor of timestamps (shape [N]).
            y: 1D tensor of observed heights (shape [N]).
            y0: Known initial height (scalar).
            g: Gravity (default 9.81 m/s²).
        Returns:
            v0: Estimated initial velocity (scalar).
        """
        y0 = y[0]  # Use the first point as the initial height
        t = t - t[0]  # Normalize time to start from 0
        y_adjusted = y - y0 + 0.5 * g * t**2  # y_adj = v0 * t
        A = t.unsqueeze(1)  # Design matrix [N, 1] (t)
        solution = torch.linalg.lstsq(A, y_adjusted).solution  # [v0]
        v0 = solution.item()
        return v0
    
    # def fit(x, y):
    #     A = torch.vstack([x, torch.ones_like(x)]).T  # Shape (N, 2)
    #     solution = torch.linalg.lstsq(A, y).solution  # Shape (2,)
    #     k, b = solution[0], solution[1]
    #     return k 
    
    velocity = []
    for i in range(3):
        g = 9.8  if i == 2 else 0.0
        v_i = estimate_initial_velocity(timestamps, points[:, i], g=g)
        velocity.append(v_i)
    velocity = torch.tensor(velocity, dtype=torch.float32)  # Convert to tensor

    return velocity

class PhyxModel(torch.nn.Module):
    def __init__(self):
        super(PhyxModel, self).__init__()
        # self.v0 = torch.nn.Parameter(torch.tensor([-0.01,  0.01  ,0.01]))
        self.w0 = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

    def forward(self,p0,v0, time_stamps, z0 = 0.020):
        p_out = [p0]
        w = self.w0
        p = p0
        v = v0
        first_bounce_idx = None
        for i in range(len(time_stamps)-1):
            # print(f"p: {p[2]} v: {v[2]}")
            if p[2] < z0 and v[2] < 0:
                # print("bounce")
                v, w = predict_bounce_roll(v, w)
                p = p + v * (time_stamps[i+1] - time_stamps[i])
                p_out.append(p)
                first_bounce_idx = i
            else:
                dt = time_stamps[i+1] - time_stamps[i]
                v = predict_vel_aero(v, w, dt)
                p = p + v * dt
                p_out.append(p)
        
        return torch.stack(p_out, dim=0), first_bounce_idx


def grad_descent_inference():
    single_file_data = read_single_traj_file(TRAJ_DATASET_PATH / 'spin_p2_vel_30_bag1.txt')

    tid = 2
    data, vw_ref = single_file_data
    tid += int(data[0,0])
    data1 = data[data[:, 0].astype(int) == tid, :]  # select only the first trajectory
    data1  = data1[~np.isnan(data1).any(axis=1)]

    if len(data1) < 20:
        print(f"\t - Trajectory {tid} is too short. Skipping.")
        return
    
    data1 = torch.tensor(data1)
    data1 = data1[:,:]  # limit to 300 points for testing
    timestamps = data1[:,1]

    loss_fn = torch.nn.L1Loss()
    phyx_model = PhyxModel()

    


    # phyx_model.w0 = torch.nn.Parameter(torch.tensor([ 0, -40.895191, 0]))
    phyx_model.train()
    

    optimizer = torch.optim.SGD(phyx_model.parameters(), lr=2000)
   
    est_len = 30
    v0 = estimate_velocity(data1[:est_len,2:5], timestamps[:est_len])
    v0 = v0.to(torch.float32)
    
    # training loop

    best_loss = float('inf')
    best_w0 = None
    # for epoch in tqdm.tqdm(range(300)):
    for epoch in range(500):
        optimizer.zero_grad()
        
        pout, first_bounce_idx = phyx_model(data1[0,2:5], v0, timestamps)

        # if first_bounce_idx is not None:
        #     pout[first_bounce_idx:,:] += data1[first_bounce_idx,2:5] - pout[first_bounce_idx,:]

        loss = loss_fn(pout[:-1,:], data1[:-1,2:5])
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss: 
            best_loss = loss.item() 
            best_w0 = phyx_model.w0.detach().numpy()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            print(f"Curent est v0: {v0} | Curent est w0: {phyx_model.w0.detach().numpy()}")
        
            print(f"Best loss: {best_loss:.4f} | w0: {best_w0}")
            # raise
    fig = plt.figure()
    data1 = data1.detach().cpu().numpy()
    pout = pout.detach().cpu().numpy()
    ax = fig.add_subplot(111, projection='3d')
 
    ax.scatter(data1[:, 2], data1[:, 3], data1[:, 4], c='b', marker='o')
    ax.plot(pout[:,0], pout[:,1], pout[:,2], c='r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')    
    draw_util.set_axes_equal(ax)
    plt.show()


def grad_descent_validate():
    single_file_data = read_single_traj_file(TRAJ_DATASET_PATH / 'spin_p2_vel_30_bag1.txt')

    tid = 2
    data, vw_ref = single_file_data
    tid += int(data[0,0])
    data1 = data[data[:, 0].astype(int) == tid, :]  # select only the first trajectory
    data1  = data1[~np.isnan(data1).any(axis=1)]

    if len(data1) < 20:
        print(f"\t - Trajectory {tid} is too short. Skipping.")
        return
    
    data1 = torch.tensor(data1)
    data1 = data1[:,:]  # limit to 300 points for testing
    timestamps = data1[:,1]

    loss_fn = torch.nn.L1Loss()
    phyx_model = PhyxModel()
    phyx_model.w0 = torch.nn.Parameter(torch.tensor([  42.736404 ,-19.29259 ,  41.166912]))
    
    
   
    est_len = 30
    v0 = estimate_velocity(data1[:est_len,2:5], timestamps[:est_len])
    v0 = v0.to(torch.float32)
    # training loop
        
    pout,first_bounce_idx = phyx_model(data1[0,2:5], v0, timestamps)
    
    if first_bounce_idx is not None:
        pout[first_bounce_idx:,:] += data1[first_bounce_idx,2:5] - pout[first_bounce_idx,:]


    loss = loss_fn(pout[:-1,:], data1[:-1,2:5])
    print(f"Validation Loss: {loss.item():.4f}")
    

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


def gradient_descent_labeling(traj_files):
    """
    Gradient descent labeling for trajectory files.
    Args:
        traj_files: List of trajectory files.
    """

    for traj_file in traj_files:
        # read data
        data, vw_ref = read_single_traj_file(traj_file)
        print(f"Processing file: {traj_file} | Spin: {vw_ref[0]} | Velocity: {vw_ref[1]}")   
        # loop over all trajectories in the file
        for tid in range(int(data[0,0]), int(data[-1,0])+1):
            print(f"\t - processing {tid} trajectory")
            data1 = data[data[:, 0].astype(int) == tid, :]  # select only the single trajectory
            if len(data1) < 20:
                print(f"\t - Trajectory {tid} is too short. Skipping.")
                continue
            data1  = data1[~np.isnan(data1).any(axis=1)]
            data1 = torch.tensor(data1)
            timestamps = data1[:, 1]
            # timestamps = torch.tensor(timestamps)
            
            phyx_model = PhyxModel()
            phyx_model.train()

            loss_fn = torch.nn.L1Loss()

            optimizer = torch.optim.Adam(phyx_model.parameters(), lr=.5, betas=(0.99, 0.999))
            
            est_len = 30
            v0 = estimate_velocity(data1[:est_len, 2:5], timestamps[:est_len])
            v0 = v0.to(torch.float32)

            min_loss = float('inf')
            optim_w0 = None
            # training loop
            # for epoch in range(300):
            with tqdm.tqdm(range(1000), desc="Training", ncols=150) as pbar: # Assign tqdm to a variable

                for epoch in pbar:
        
                    optimizer.zero_grad()
                    
                    pout = phyx_model(data1[0, 2:5], v0, timestamps)
                    loss = loss_fn(pout[:-1, :], data1[:-1, 2:5])
                    loss.backward()
                    optimizer.step()

                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        optim_w0 = phyx_model.w0.detach().numpy()
                        # print(f"\t\t- New best loss: {min_loss:.4f} | w0: {optim_w0}")

                    pbar.set_postfix({
                        'Curr Loss': f'{loss.item():.4f}',
                        'Best Loss': f'{min_loss:.4f}',
                        'Best w0': f'{optim_w0[0]:.2f}, {optim_w0[1]:.2f}, {optim_w0[2]:.2f}' if optim_w0 is not None else 'N/A'
                    })


                    # print("\033[F", end='')

                    # if epoch % 10 == 0:
                    #     print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
                    #     print(f"Curent est v0: {v0} | Curent est w0: {phyx_model.w0.detach().numpy()}") 

if __name__ == '__main__':
    # traj_files_data = read_traj_files(TRAJ_DATASET_PATH)
    # unrolled_data = unroll_by_tid(traj_files_data)
    # tid1, data1, vw_ref1 = unrolled_data[8]
    import argparse
    parser = argparse.ArgumentParser(description='Process trajectory files.')
    
    # if parser set inference
    parser.add_argument('--inference', action='store_true', help='Run inference on a single trajectory file.')  
    parser.add_argument('--validate', action='store_true', help='Run validation on a single trajectory file.')
    parser.add_argument('--label', action='store_true', help='Run labeling on trajectory files.')
    args = parser.parse_args()
    if args.inference:
        grad_descent_inference()
    elif args.validate:
        grad_descent_validate()
    elif args.label:
        traj_files = list(TRAJ_DATASET_PATH.glob('spin*.txt'))
        gradient_descent_labeling(traj_files)
    else:
        print("Please specify --inference, --validate, or --label.")
    # grad_descent_inference()
    # grad_descent_validate()
    # gradient_descent_labeling(Path("data/real/tennis_triangulated_spin").glob('spin*.txt'))