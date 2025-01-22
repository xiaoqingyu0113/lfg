from train import RealTrajectoryDataset
import torch
from torch.utils.data import DataLoader
from lfg.model_traj.mnn import MNN, autoregr_MNN
from lfg.estimator import OptimLayer
import os
import matplotlib.pyplot as plt
import numpy as np


WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..',
                            '..',
                            'logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run40/'))

# load model
mnn = MNN(z0=0.076)
mnn.load_state_dict(torch.load(WEIGHTS_DIR + '/model_MNN.pth'))
mnn.eval()
mnn = mnn.cuda()

est = OptimLayer(mnn, size=130, damping = 0.1, max_iterations=30, allow_grad=False)
est.eval()
est = est.cuda()


dataset = RealTrajectoryDataset('data/real/tennis_kpts',interpolate=500)
print(f"Loaded dataset with {len(dataset)} samples.")

total_data_size = len(dataset)
split_ratio = 0.8
train_data_size = int(total_data_size * split_ratio)
test_data_size = total_data_size - train_data_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


test_loader = iter(test_loader)
test_data = next(test_loader)

pN_est = autoregr_MNN(test_data, mnn, est, None)
skip = 1
test_data = test_data[:,::skip,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(30, 400):
    est.size = i

    # pN_est = autoregr_MNN(test_data, mnn, est, None)
    pN_est = autoregr_MNN(test_data, mnn, est, None)



    # start plotting
    ax.clear()
    #  - plot current ball
    curr_ball = test_data[0,i,2:5].cpu().detach().numpy()
    ax.scatter(curr_ball[0], curr_ball[1], curr_ball[2], color='red') # current position
    # - plot predicted trajectory
    pN_est = pN_est.cpu().detach().numpy()[0, i:, :]
    ax.plot(pN_est[:,0], pN_est[:,1], pN_est[:,2], color='g') # predicted trajectory
    # - plot raw points
    raw_points = test_data[0, :, :].cpu().detach().numpy()
    ax.plot(raw_points[:,2], raw_points[:,3], raw_points[:,4]) # raw
   

    # axes equal
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-10, 20)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-10, 15)
    # extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    # centers = np.mean(extents, axis=1)
    # max_range = np.ptp(extents, axis=1).max() / 2
    # for ctr, axis in zip(centers, 'xyz'):
    #     getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)

    # save it!
    save_filename = f'plots/increm_theseus_GN/{i:04d}.png'
    if not os.path.exists(save_filename):
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    plt.savefig(save_filename)    
    print(save_filename)


