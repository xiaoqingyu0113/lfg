import torch 
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
from draw_util import draw_util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bounce import TestModel3 as BC_Model
from aero import TestModel5 as Aero_Model
from test_synthetic_training.synthetic.predictor import predict_trajectory
from lfg.model_traj.lstm import LSTM




def load_random_data(batch_size = 64, seq_len = 300):
    # v0 = np.array([2.0, 3.0, 1.0])
    # w0 = np.array([20.0, 10.0, 5.0]) * np.pi * 2.0
    # p0 = np.array([0.0, 0.0, 1.0])
    # t = np.linspace(0, 1, 100)

    # xN = predict_trajectory(p0, v0, w0, t)

    v0 = np.array([2.0, 3.0, 1.0]) + (np.random.randn(batch_size, 3)-0.5)*1.0
    w0 = (np.array([20.0, 10.0, 5.0]) + (np.random.randn(batch_size, 3)-0.5)*3.0) * np.pi * 2.0
    p0 = np.array([0.0, 0.0, 1.0]) + (np.random.randn(batch_size, 3)-0.5)*1.0

    t = np.linspace(0, 2, seq_len)
    t_batch = np.tile(t, (batch_size, 1))
    dt = np.diff(t_batch, axis=-1)
    # append 0 to the first time step
    dt = np.concatenate([np.zeros((batch_size, 1)), dt], axis=-1)
    
    data = []
    for b in range(batch_size):
        xN = predict_trajectory(p0[b], v0[b], w0[b], t_batch[b])
        data.append(np.concatenate([t.reshape(-1,1), xN], axis=-1))

    data = np.array(data)

    ret_data = data[:,:,:4]
    p = torch.from_numpy(data[:,:,1:4]).float().to('cuda')
    dt = torch.from_numpy(dt).float().to('cuda')
    w0 = torch.from_numpy(w0).float().to('cuda')

    return p, dt, w0

def train():
    model = LSTM().to('cuda')
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(500):
        p ,dt ,w0 = load_random_data()
        optimizer.zero_grad()
        pN = model(p, dt, w0)
        loss = criterion(pN, p)
        loss.backward()
        optimizer.step()
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))
    
    torch.save(model.state_dict(), 'data/archive/LSTM_model.pth')
    

if __name__ == '__main__':
    train()