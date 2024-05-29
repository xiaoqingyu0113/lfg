import torch
import torch.nn as nn
import theseus as th

from lfg.graph import InvariantFactorGraph
from lfg.util import plot_to_tensorboard

import matplotlib.pyplot as plt
from mcf4pingpong.draw_util import set_axes_equal
from typing import Tuple, Sequence
from functools import partial
from tensorboardX import SummaryWriter

'''
This is a toy example for trajectory g = [0,0,-9.8]
    1. generate trajectory for a = g, t = [0,2] sec
    2. add random noise 
    3. train mlp([3,3]) to regress v0 + mlp *dt - v1
    4. compare mlp[3,3] with [0,0,9.8]

     p0 --MLP-- p1---MLP--p2
          |         |
         V0 --MLP-- V1
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tb_writer = SummaryWriter(log_dir='logdir/run1')

def generate_trajectory(visualize = False):
    def euler_int(carry, dt):
       p0, v0 = carry
       p1 = p0 + v0 * dt
       v1 =  v0 + torch.tensor([0.0,0.0,-9.8])*dt + torch.linalg.cross(v0, torch.tensor([0.0,3.0,3.0],device='cpu'))*dt
       return (p1, v1), p1
    N=40
    v0 = torch.tensor([5.0,0.0,0.0])
    p0 = torch.tensor([0.0,0.0,0.0])
    t = torch.linspace(0.0,2.0,N)
    dts = torch.diff(t)
    
    pN = [p0]
    for dt in dts:
        (p0,v0), p0 = euler_int((p0,v0),dt)
        pN.append(p0)
    pN = torch.vstack(pN)
    noise_scale = 0.30  # Standard deviation of the noise
    noise = noise_scale *  torch.randn((N, 3))
    pN_noisy = pN + noise
    if visualize:
        fig= plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*pN_noisy.T)
        set_axes_equal(ax)
        plt.show()

    return t, pN_noisy, pN


@torch.jit.script
def rotz(batch_angles):
    """
    Convert batch angles to rotation matrices about the z-axis using PyTorch.

    Args:
    batch_angles (torch.Tensor): Batch of angles in radians, shape (B, 1).

    Returns:
    torch.Tensor: Batch of rotation matrices, shape (B, 3, 3).
    """
    cos_theta = torch.cos(batch_angles)
    sin_theta = torch.sin(batch_angles)
    zeros = torch.zeros_like(batch_angles)
    ones = torch.ones_like(batch_angles)

    # Construct the rotation matrices
    rotation_matrices = torch.stack([
        torch.stack([cos_theta.squeeze(1), -sin_theta.squeeze(1), zeros.squeeze(1)], dim=1),
        torch.stack([sin_theta.squeeze(1), cos_theta.squeeze(1), zeros.squeeze(1)], dim=1),
        torch.stack([zeros.squeeze(1), zeros.squeeze(1), ones.squeeze(1)], dim=1)
    ], dim=1)

    return rotation_matrices


def pos_prior_errfn(optim_vars: Tuple[th.Variable,...], aux_vars: Tuple[th.Variable,...]):
    li, = optim_vars 
    li_prior, = aux_vars
    err = li.tensor[:,:3] - li_prior.tensor[:,:3]
    return err



def dynamic_forward(model: nn.Module, x:torch.Tensor, dt:torch.Tensor):
    p, th, v, w = x[:,:3], x[:,3:4], x[:,4:6], x[:, 6:7] # keep the batch

    pdot = rotz(th)@torch.cat((v[:, 0], torch.tensor([0.0]), v[:, 1]), dim=0)
    thdot = w
    vwdot = model(torch.cat((v,w),dim=1))

    x_dot = torch.cat((pdot,thdot, vwdot), dim=1)

    return x + x_dot * dt

def dynamic_errfn(model: nn.Module, optim_vars:Sequence[th.Variable], aux_vars:Sequence[th.Variable]):
    x1_var, x2_var = optim_vars
    dt_var, = aux_vars

    x1, x2 = x1_var.tensor, x2_var.tensor
    dt = dt_var.tensor

    x2_est = dynamic_forward(model, x1, dt)

    return x2_est - x2

def inner_loop(graph, model, pN_noisy,t):
    theseus_inputs = {}
    weight_l_prior = th.ScaleCostWeight(0.1)

    for i, (ti, pi) in enumerate(zip(t, pN_noisy)):
        xi = th.Vector(7, name=f"x{i}")
        theseus_inputs.update({f'x{i}': torch.zeros(1,7)})
        
        xi_prior_tensor = torch.zeros((1,7))
        xi_prior_tensor[:, :3] = pi
        xi_prior = th.Variable(xi_prior_tensor, name=f"x{i}_prior")

        graph.add(th.AutoDiffCostFunction(
            (xi,), 
            pos_prior_errfn, 
            3, 
            aux_vars=(xi_prior,), 
            name=f"x{i}_prior_erfn", 
            cost_weight=weight_l_prior))
            
        if i>0:
            dt_prev = th.Variable(torch.tensor([[ti-t_prev]]),name=f'dt{i-1}') 
            theseus_inputs.update({f'dt{i-1}': torch.tensor([[ti-t_prev]])})

            graph.add(th.AutoDiffCostFunction(
                (xi_prev, xi), 
                partial(dynamic_errfn, model), 
                7, 
                aux_vars=(dt_prev,), 
                name=f"pose_between{i}"))
            
        xi_prev = xi
        t_prev = ti

    return graph, theseus_inputs

def main():
    t, pN_noisy, pN_clean = generate_trajectory(visualize=False)
    

    graph = InvariantFactorGraph()
    model = nn.Linear(3,3)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    graph, theseus_inputs = inner_loop(graph, model, pN_noisy, t)
    fig= plt.figure()
    ax = fig.add_subplot(projection='3d')
    for epoch in range(100):
        optimizer.zero_grad()
        updated_inputs, info = graph.forward(theseus_inputs)
        graph.update(updated_inputs)

        loss = graph.compute_loss().sum()
        print(f"{epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()
        tb_writer.add_scalars(main_tag='loss', tag_scalar_dict={'training':loss.item()}, global_step=epoch)


        xN_opt = []
        for k,v in updated_inputs.items():
            if 'x' in k:
                xN_opt.append(v)
    

        xN_opt = torch.vstack(xN_opt)
        pN_opt = xN_opt[:,:3].detach().cpu()

        ax.cla()
        
        ax.scatter(*pN_noisy.T)
        ax.scatter(*pN_opt.T)
        ax.plot(*pN_clean.numpy().T)
        set_axes_equal(ax)
        # plot_to_tensorboard(tb_writer, 'visualze', fig, epoch)
        fig.savefig('plot.png')
if __name__ == '__main__':
    main()
