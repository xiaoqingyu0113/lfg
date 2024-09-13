from lfg.model_traj.phytune import PhyTune
from lfg.model_traj.mnn import MNN
from lfg.estimator import OptimLayer
from train_synthetic_traj import TrajectoryDataset
from ekf import ExtendedKalmanFilter
import torch

import numpy as np
import matplotlib.pyplot as plt

# Define the state transition function (F_func) and measurement function (H_func)
phytune = PhyTune()
phytune.cuda()
phytune._init_gt()
graph = OptimLayer(phytune, size=2, max_iterations=30, allow_grad=False, damping=0.001)    
graph.cuda()

def F_func(x, dt):
    """
    Nonlinear state transition function.
    For example, a simple 2D constant velocity model with some nonlinearity.
    """
    # dt = 1.0/200  # time step
    x = x.cuda()
    p = x[None,0:3]
    v = x[None, 3:6]
    w = x[None,6:9]

    v_e, w_e = phytune(p[:,2:3], v, w, dt)

    pos = p + v * dt
    
    pos = pos.squeeze()
    v_e = v_e.squeeze()
    w_e = w_e.squeeze()

    return torch.cat([pos, v_e, w_e])




def H_func(x):
    """
    Nonlinear measurement function.
    For example, we observe positions with some nonlinearity.
    """
    return x[: 3]


def run_compare():
    print('Running compare: Make sure etimator is changed to etimated the latest state !!!!')
    dataset = TrajectoryDataset('data/synthetic/traj_general.csv', noise=0.0) # noise added later in iteration
    # # Process noise covariance (Q)
    Q = torch.diag(torch.tensor([0.010, 0.010, 0.010, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1])).cuda()

    # Measurement noise covariance (R)
    R = torch.eye(3).cuda() * 0.010

    # Initial estimate of state covariance (P)
    P = torch.diag(torch.tensor([0.010, 0.010, 0.010, 1, 1, 1, 0.1, 0.1, 0.1])).cuda()



    all_results = []
    for d_id in range(25):
        data = dataset[d_id].cpu()
        data[: , 2:5] = data[:, 2:5] + (torch.randn_like(data[:, 2:5])-0.5) *2 * 0.001 # add random noise (std =1 mm)

        p_init = data[0, 2:5]
        v_init = torch.diff(data[:5, 2:5], dim=0) / torch.diff(data[:5, 1:2], dim=0) # guess the initial velocity by the first 5 frames
        v_init = v_init.mean(dim=0)
        w_init = data[0, 8:11]

        x_init = torch.cat([p_init, v_init, w_init]).cuda()
        filter = ExtendedKalmanFilter(
            state_dim=9,
            meas_dim=3,
            F_func=F_func,
            H_func=H_func,
            Q=Q,
            R=R,
            P=P,
            x=x_init
        )
        all_results.append([])
        i = 1
        while len(all_results[-1]) < 40:
            # mean vel and p
            if i > 1:
                win_size = 2
            elif i == 1:
                win_size = 1
            else:
                win_size = 0    
            p = data[i-win_size:i+win_size, 2:5].mean(dim=0)
            v_all = torch.diff(data[i-win_size:i+win_size, 2:5], dim=0) / torch.diff(data[i-win_size:i+win_size, 1:2], dim=0)
            v = v_all.mean(dim=0)
            w = data[i, 8:11]
            state_mean = torch.cat([p, v, w])

            # Kalman filter
            z = data[i][2:5].cuda() + torch.randn(3).cuda() * 0.01
            dt = data[i][1] - data[i-1][1]
            filter.predict(dt)  # Predict the next state
            filter.update(z)  # Update with the new measurement

            state_kf = filter.get_state().numpy()


            # factor graph
            graph.size = i+1
            p, v, w = graph(data[None,:i+1, 1:5].to('cuda:0'), w0=data[i][None, None, 8:11].to('cuda:0'))
            if p is not None:
                state_fg = np.concatenate([p.cpu().numpy().flatten(), v.cpu().numpy().flatten(), w.cpu().numpy().flatten()])
                result = np.c_[data[i][2:].numpy(), state_mean, state_kf, state_fg]
                all_results[-1].append(result)
                print(f'traj {len(all_results)} frame {i} result shape {result.shape}')
                print(result[3:6,:])

            i += 1  
        np.save('data/synthetic/compare_estimator.npy', np.array(all_results))

    # save np array
    np.save('data/synthetic/compare_estimator.npy', np.array(all_results))

  
def read_compare():
    all_results = np.load('data/synthetic/compare_estimator.npy')
    print(all_results.shape)
    # print(all_results[0])

def statistics():
    all_results = np.load('data/synthetic/compare_estimator.npy')
    stats = {'gt':{'mean':[], 'std': []}, 'SW':{'mean':[], 'std': []}, 'EKF':{'mean':[], 'std': []}, 'FG':{'mean':[], 'std': []}}
    VEL = range(3, 6)

    print(all_results.shape)

    for i in range(all_results.shape[1]):
        data = all_results[:, i, ...]
        gt = data[:, VEL, 0]
        sw = data[:, VEL, 1]
        ekf = data[:, VEL, 2]
        fg = data[:, VEL, 3]

        print(gt.shape, sw.shape, ekf.shape, fg.shape)
        stats['SW']['mean'].append(np.mean(np.linalg.norm(sw - gt, axis=1)))
        stats['SW']['std'].append(np.std(np.linalg.norm(sw - gt, axis=1)))

        stats['EKF']['mean'].append(np.mean(np.linalg.norm(ekf - gt, axis=1)))
        stats['EKF']['std'].append(np.std(np.linalg.norm(ekf - gt, axis=1)))

        stats['FG']['mean'].append(np.mean(np.linalg.norm(fg - gt, axis=1)))
        stats['FG']['std'].append(np.std(np.linalg.norm(fg - gt, axis=1)))


    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax.set_title('Estimator Comparison', fontsize=16)
    ax.set_xlabel('Number of Observations', fontsize=16)
    ax.set_ylabel('Estimation Error', fontsize=16)

    frames = list(range(all_results.shape[1]))[4:]

    # Plotting with error bars
    ax.errorbar(frames, stats['SW']['mean'][4:], yerr=stats['SW']['std'][4:], label='Sliding Window', fmt='o-', capsize=5)
    ax.errorbar(frames, stats['EKF']['mean'][4:], yerr=stats['EKF']['std'][4:], label='EKF', fmt='o-', capsize=5)
    ax.errorbar(frames, stats['FG']['mean'][4:], yerr=stats['FG']['std'][4:], label='Factor Graph', fmt='o-', capsize=5)


    # yaxis log
    # ax.set_yscale('log')
    ax.legend(fontsize=12)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    # increase tick size
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Show plot
    plt.tight_layout()
    plt.show()
    




        

if __name__ == "__main__":
    # run_compare()
    # read_compare()
    statistics()

   


