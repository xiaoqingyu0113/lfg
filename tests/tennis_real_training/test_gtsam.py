import numpy as np
import gtsam
import time
from gtsam.symbol_shorthand import X, L, V, W
import matplotlib.pyplot as plt
from lfg.derive import PriorFactor3, PositionFactor, VWFactor, predict
from lfg.estimator import OptimLayer
from lfg.model_traj import MNN, autoregr_MNN
import torch

DTYPE = np.float64

def axes_equal(ax):
    """
    Set equal aspect ratio for a 3D plot.

    Parameters:
    ax : matplotlib.axes._subplots.Axes3DSubplot
        A 3D axes object.
    """
    extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.ptp(extents, axis=1).max() / 2

    for ctr, axis in zip(centers, 'xyz'):
        getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)

def interpolate_data( data: np.ndarray, interpolate: int) -> np.ndarray:
        '''
        interpolate data to have equal length for batch learning
        '''
        from scipy.interpolate import interp1d
        data_tmp_i = data
        interp_f = interp1d(data_tmp_i[:, 1], data_tmp_i, axis=0)
        tmax = data_tmp_i[-1, 1]
        tmin = data_tmp_i[0, 1]
        t_avg_spacing = (tmax - tmin) / interpolate
        t_random_noise = np.random.uniform(-t_avg_spacing/2.5, t_avg_spacing/2.5, interpolate)
        t = np.linspace(tmin, tmax, interpolate) # + t_random_noise
        t[0] = tmin
        t[-1] = tmax
        data_i = interp_f(t)
        return data_i

def theseus_results(data):
    model = MNN()
    model.load_state_dict(torch.load('logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20/model_MNN.pth'))
    est = OptimLayer(model,size=120, max_iterations=30,allow_grad=False, damping =0.1)
    est.state_dict = torch.load('logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20/est_OptimLayer.pth')
    est.cuda()

    data = interpolate_data(data, 500)
 
    data = torch.from_numpy(data).float()[None, :].cuda()
    # p0, v0, w0 = est(data[:,:120,1:5], torch.tensor(data[:,0:1,8:11]).float().cuda())
    # print(p0, v0, w0)
    pN_est = autoregr_MNN(data, model, est, None)
    pN_est = pN_est.cpu().detach().numpy()
    pN_est = pN_est[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = data.cpu().detach().numpy()[0]
    ax.scatter(data[ :, 2], data[:, 3], data[ :, 4], c='b')
    ax.plot(pN_est[:, 0], pN_est[:, 1], pN_est[:, 2], c='g')
    axes_equal(ax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

def test_gtsam(data):
    # Create a factor graph
    print(data.shape)
    data = interpolate_data(data, 500)
    print(data.shape)
    graph = gtsam.NonlinearFactorGraph()
    parameters = gtsam.ISAM2Params()
    isam2 = gtsam.ISAM2(parameters)
    initial_estimate = gtsam.Values()
    optim_estimate = None
    start_time = None
    minimum_graph_size = 30 # minimum number of factors to optimize

    # Create noise models

    pPriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.010, 0.010, 0.010], dtype=DTYPE))
    vwNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1], dtype=DTYPE))
    wPriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1], dtype=DTYPE))

    N_seq, N_data = data.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N_seq):
        t, p, v, w = data[i, 1], data[i, 2:5], data[i, 5:8], data[i, 8:11]
        
        graph.push_back(PriorFactor3(pPriorNoise, L(i), p))
        initial_estimate.insert(L(i), p)
        
        # add w prior only at the beginning
        if i == 0:
            graph.push_back(PriorFactor3(wPriorNoise, W(i), w))
            initial_estimate.insert(W(i), w)
            initial_estimate.insert(V(i), np.array([-3.0,0,0]).astype(DTYPE))
        
        # add forward dynamics factor for i > 0
        else:
            t_prev = data[i-1, 1]
            graph.push_back(PositionFactor(pPriorNoise, L(i-1), V(i-1), L(i), t_prev, t))
            graph.push_back(VWFactor(vwNoise,L(i-1), V(i-1), W(i-1), V(i), W(i), t_prev, t, 0.200))

            if optim_estimate is None:
                initial_estimate.insert(V(i), 1e-3*np.random.rand(3).astype(DTYPE))
                initial_estimate.insert(W(i), 1e-3*np.random.rand(3).astype(DTYPE))
            else:
                v_prev = optim_estimate.atVector(V(i-1))
                w_prev = optim_estimate.atVector(W(i-1))
                initial_estimate.insert(V(i), v_prev)
                initial_estimate.insert(W(i), w_prev)

        if i > minimum_graph_size:
            start_time = time.time()
            isam2.update(graph, initial_estimate)
            optim_estimate = isam2.calculateEstimate()
            print(f'i={i}, inference time = {time.time() - start_time}')
            graph.resize(0)
            initial_estimate.clear()

        if i >= 80:
            est = isam2.calculateEstimate()
        
            p = np.array([est.atVector(L(ii)) for ii in range(i)])
            p_curr = est.atVector(L(i))
            v_curr = est.atVector(V(i))
            w_curr = est.atVector(W(i))

            points = predict(p_curr, v_curr, w_curr, 3.0, 300)


            ax.clear()
            ax.scatter(data[:, 2], data[:, 3], data[:, 4], c='b',s=0.5)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], c='g', linewidth=3)
            ax.plot(p[:, 0], p[:, 1], p[:, 2], c='y', linewidth=3)
            ax.scatter(p_curr[0], p_curr[1], p_curr[2], c='r', s=20)
            axes_equal(ax)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


            # plt.show()
            # break
            fig.savefig(f'plots/gtsam_no_1_6_interp/{i:04d}.png')

    plt.show()
def get_data(i):
    import glob
    data_files = list(glob.glob('data/real/tennis_no_1_6/*.txt'))
    data_files.sort()

    selected_file = data_files[i]

    data = np.loadtxt(selected_file)

    return data

# theseus_results(get_data(0))
test_gtsam(get_data(0))