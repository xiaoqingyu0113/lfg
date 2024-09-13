import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np
# import mcf4ball.parameters as param

def axis_equal(ax,X,Y,Z,zoomin=1):
   # Set the limits of the axes to be equal

    x = np.array(X).flatten()
    y = np.array(Y).flatten()
    z = np.array(Z).flatten()

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim3d((mid_x - max_range)/zoomin, (mid_x + max_range)/zoomin)
    ax.set_ylim3d((mid_y - max_range)/zoomin, (mid_y + max_range)/zoomin)
    ax.set_zlim3d((mid_z - max_range)/zoomin, (mid_z + max_range/zoomin))

    # Set labels for the axes
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

def set_axes_pane_white(ax):
    ax.xaxis.pane.fill = False  # Set the pane's fill to False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

def plot_sphere(ax,xc,yc,zc,r):
    # Make data
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = r * np.outer(np.cos(u), np.sin(v)) + xc
    y = r * np.outer(np.sin(u), np.sin(v)) + yc
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zc
    ax.plot_surface(x, y, z,color='orange')

def plot_spheres(ax,xc,yc,zc,r):
    for i in range(len(xc)):
        plot_sphere(ax,xc[i],yc[i],zc[i],r)
        
def set_axes_equal(ax, zoomin = 1.0):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5/zoomin*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def axis_bgc_white(ax):
    ax.set_facecolor('white')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(color='black')





def draw_tennis_court_outline(ax,z0 = 0.0):
    rects = dict()
    m_per_ft = 0.3048
    rects['bottom_1'] = (np.array([0.0, 27/2.0, z0]),np.array([18.0, -27/2.0, z0]) )
    rects['bottom_2'] = (np.array([18+21+21, 27/2.0, z0]),np.array([39*2, -27/2.0, z0]) )
    rects['left_service'] = (np.array([18,13.5,z0]),np.array([39+21,0,z0]))
    rects['right_service'] = (np.array([18,0,z0]),np.array([39+21,-13.5,z0]))
    rects['left_side'] = (np.array([0,18,z0]),np.array([39*2,13.5,z0]))
    rects['right_side'] = (np.array([0,-13.5,z0]),np.array([39*2,-18,z0]))
    rects['net'] = (np.array([39,22,z0]), np.array([39,-22,z0+3]))

    def draw_rectangle(ax, corner1,corner2,**argv):
        if np.abs(corner1[2] - corner2[2]) <0.1:
            ax.plot([corner1[0],corner2[0]],[corner1[1],corner1[1]],[corner1[1],corner1[1]],**argv)
            ax.plot([corner1[0],corner2[0]],[corner2[1],corner2[1]],[corner1[1],corner1[1]],**argv)
            ax.plot([corner1[0],corner1[0]],[corner1[1],corner2[1]],[corner1[1],corner1[1]],**argv)
            ax.plot([corner2[0],corner2[0]],[corner1[1],corner2[1]],[corner1[1],corner1[1]],**argv)
        else:
            ax.plot([corner1[0],corner1[0]],[corner2[1],corner2[1]],[corner1[1],corner2[1]],**argv)
            ax.plot([corner1[0],corner1[0]],[corner1[1],corner1[1]],[corner1[1],corner2[1]],**argv)
            ax.plot([corner1[0],corner1[0]],[corner1[1],corner2[1]],[corner1[1],corner1[1]],**argv)
            ax.plot([corner1[0],corner1[0]],[corner1[1],corner2[1]],[corner2[1],corner2[1]],**argv)

    for k,v in rects.items():
        if k == 'net':
            draw_rectangle(ax,v[0]*m_per_ft,v[1]*m_per_ft,linewidth = 3, color='black')

        else:
            draw_rectangle(ax,v[0]*m_per_ft,v[1]*m_per_ft,linewidth = 3, color='black')

def draw_pinpong_table_outline(ax,z0 = 0.0):
    rects = dict()
    m_per_ft = 0.3048

    origin_offset = np.array([-0.900, 1.2, 0.0]) 
    rects['1'] = (np.array([5.0, 0.0, z0]),np.array([2.5,-4.5, z0]) )
    rects['2'] = (np.array([5.0, -9.0, z0]),np.array([2.5,-4.5, z0]) )
    rects['3'] = (np.array([0.0, 0.0, z0]),np.array([2.5,-4.5, z0]) )
    rects['4'] = (np.array([0.0, -9.0, z0]),np.array([2.5,-4.5, z0]) )


    rects['net'] = (np.array([0.0,-4.5,z0]), np.array([5.0,-4.5,z0+0.5]))

    def draw_rectangle(ax, corner1,corner2,**argv):
        corner1 = corner1 + origin_offset * m_per_ft
        corner2 = corner2 + origin_offset * m_per_ft
        if np.abs(corner1[2] - corner2[2]) <0.1:
            ax.plot([corner1[0],corner2[0]],[corner1[1],corner1[1]],[corner1[2],corner1[2]],**argv)
            ax.plot([corner1[0],corner2[0]],[corner2[1],corner2[1]],[corner1[2],corner1[2]],**argv)
            ax.plot([corner1[0],corner1[0]],[corner1[1],corner2[1]],[corner1[2],corner1[2]],**argv)
            ax.plot([corner2[0],corner2[0]],[corner1[1],corner2[1]],[corner1[2],corner1[2]],**argv)
        else:
            ax.plot([corner1[0],corner1[0]],[corner1[1],corner1[1]],[corner1[2],corner2[2]],**argv)
            ax.plot([corner1[0],corner2[0]],[corner1[1],corner1[1]],[corner2[2],corner2[2]],**argv)
            ax.plot([corner2[0],corner2[0]],[corner1[1],corner1[1]],[corner1[2],corner2[2]],**argv)
            ax.plot([corner1[0],corner2[0]],[corner1[1],corner1[1]],[corner1[2],corner1[2]],**argv)

    for k,v in rects.items():
        if k == 'net':
            draw_rectangle(ax,v[0]*m_per_ft,v[1]*m_per_ft,linewidth = 3, color='black')

        else:
            draw_rectangle(ax,v[0]*m_per_ft,v[1]*m_per_ft,linewidth = 3, color='black')


