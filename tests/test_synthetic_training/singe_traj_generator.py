import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  'synthetic')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))

print (sys.path)

from synthetic.predictor import  predict_trajectory
from draw_util import draw_util
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def draw_traj_given_initial(ax, p0, v0, w0, tspan, **kwargs):
        xN = predict_trajectory(p0, v0, w0, tspan,z0=0.010)
        ax.plot(xN[:, 0], xN[:, 1], xN[:, 2], **kwargs)
        return xN


def plot_traj_random_pos():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p0 = np.array([0.5, 0.3, 0.8])  # initial position
    v0 = np.array([0.0, -2.0, 0.6])  # initial velocity
    w0 = np.array([0.0, 10.0, 0.0])  # initial angular velocity

    tspan = np.linspace(0, 2, 500)  # time span

    pNs = []
    for i in range(5):
        p0 = p0 + np.array([0,0.010,0.0])
        xN = draw_traj_given_initial(ax, p0, v0, w0, tspan, linewidth=1)
        pNs.append(xN[:,:3])

    pN0 = pNs[0]
    pN1 = pNs[1]
    rmse = np.sqrt(np.mean(np.square(pN0 - pN1)))
    print('RMSE:', rmse)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # axis pane color white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color (RGBA format)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
    draw_util.set_axes_equal(ax)
    draw_util.draw_pinpong_table_outline(ax)
    plt.show()

def animate_traj_random_vel():


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p0 = np.array([0.5, 0.1, 0.8])  # initial position
    v0 = np.array([0.0, -2.0, 0.6])  # initial velocity
    w0 = np.array([0.0, 10.0, 0.0])  # initial angular velocity

    tspan = np.linspace(0, 2, 500)  # time span

    pNs = []
    colors = ['r', 'g', 'b', 'c', 'm']  
    global_frames = 0
    xNs = []
    for i in range(5):
        print(f'traj_id={i},frame {global_frames}')
        v0 = v0 + np.array([0,0.1,0.100])
        vscale = 0.2
        xN = predict_trajectory(p0, v0, w0, tspan,z0=0.010)
        skip = 8
        for j in range(xN.shape[0]//skip):
            ax.clear()
            for jj, xn in enumerate(xNs):
                ax.plot(xn[:, 0], xn[:, 1], xn[:, 2], linewidth = 2, color = colors[jj])
            ax.quiver(p0[0], p0[1], p0[2], v0[0]*vscale, v0[1]*vscale, v0[2]*vscale, color=colors[i])
            ax.plot(xN[:j*skip, 0], xN[:j*skip, 1], xN[:j*skip, 2], linewidth = 2, color = colors[i])
            pNs.append(xN[:,:3])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # axis pane color white
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color (RGBA format)
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White color
            ax.view_init(19,-20)
            # draw_util.set_axes_equal(ax,zoomin=1.5)
            ax.set_xlim([-0.2, 1.2])
            ax.set_ylim([-1.8, 0.2])
            ax.set_zlim([-0.3, 1.2])
            draw_util.draw_pinpong_table_outline(ax)
            fig.savefig(f'plots/traj_random_vel/frame_{global_frames:04d}.png')
            global_frames += 1
        xNs.append(xN)

    # plt.show()


   

if __name__ == '__main__':
    # plot_traj_random_pos()
    animate_traj_random_vel()
