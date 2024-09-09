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

DEVICE = torch.device("cuda:0")


def save_png(*args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    labels = ['gt model', 'mnn model', 'craft aerodynamics']
    for label, pN in zip(labels, args):
        ax.plot(pN[:,0], pN[:,1], pN[:,2], label=label)
    
    # rotate camera view
    ax.legend()
    ax.view_init(elev=10, azim=45)
    draw_util.set_axes_equal(ax,zoomin=3.0)
    fig.savefig('data/archive/test_warmstart.png')

def model_trajectory(p0, v0, w0, t):
    p0 = torch.tensor(p0, device=DEVICE).view(1,3).float()
    v0 = torch.tensor(v0, device=DEVICE).view(1,3).float()
    w0 = torch.tensor(w0, device=DEVICE).view(1,3).float()
    t = torch.tensor(t, device=DEVICE)

    model = Aero_Model()
    model.load_state_dict(torch.load('data/archive/aero_model.pth'))
    model = model.to(DEVICE)

    bounce_model = BC_Model()
    bounce_model.load_state_dict(torch.load('data/archive/bounce_model.pth'))
    bounce_model = bounce_model.to(DEVICE)


    pN = [p0]

    with torch.no_grad():
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            p0 = p0 + v0 * dt
            if p0[0,2] < 0.0 and v0[0,2] < 0.0:
                v0, w0 = bounce_model(v0, w0)
                print('bounce')
            v0 = model(v0, w0, dt)
            pN.append(p0)
            print(p0)
        pN = torch.cat(pN, dim=0)
    pN = pN.cpu().numpy()
    return pN

def craft_trajectoy(p0, v0, w0, t):
    pN = [p0]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        p0 = p0 + v0 * dt
        acc = np.array([0.0, 0.0, -9.81]) + 0.014948 * np.cross(w0, v0) - 0.1118 * np.linalg.norm(v0) * v0
        v0 = v0 + acc * dt
        pN.append(p0)
    pN = np.array(pN)
    return pN

def test():
    v0 = np.array([2.0, 3.0, 1.0])
    w0 = np.array([20.0, 10.0, 5.0]) * np.pi * 2.0
    p0 = np.array([0.0, 0.0, 1.0])
    t = np.linspace(0, 1, 100)

    xN = predict_trajectory(p0, v0, w0, t)
    print(xN.shape)
    # pN_est = model_trajectory(p0, v0, w0, t)
    # pN_craft = craft_trajectoy(p0, v0, w0, t)
    

    # save_png(xN, pN_est, pN_craft)



if __name__ == '__main__':
    test()