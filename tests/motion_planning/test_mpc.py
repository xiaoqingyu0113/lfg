import gtsam
from gtsam.symbol_shorthand import X, U
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time

def draw_robot(ax, X, linewidth=2, arrow_length=0.6):
    x, y , theta = X

    d = 0.8
    r = 0.2


    kpts = np.array([[-r, d/2], [r, d/2], [0, d/2], [0, -d/2],[-r, -d/2],[r, -d/2], [0, 0],[arrow_length, 0]])

    tf_kpts = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ kpts.T + np.array([[x], [y]]) # 2 x N


    ax.plot(tf_kpts[0, 0:2], tf_kpts[1,0:2], 'b-', linewidth=linewidth)
    ax.plot(tf_kpts[0, 2:4], tf_kpts[1,2:4], 'b-', linewidth=linewidth)
    ax.plot(tf_kpts[0, 4:6], tf_kpts[1,4:6], 'b-', linewidth=linewidth)
    ax.arrow(tf_kpts[0,6], tf_kpts[1,6], tf_kpts[0,7]-tf_kpts[0,6], tf_kpts[1,7]-tf_kpts[1,6], head_width=0.3, head_length=0.3, fc='k', ec='k')





# factors for gtsam
def assign_jacobians(jacobians: List[np.ndarray],J: List[np.ndarray]):
     for i, JJ in enumerate(J):
          jacobians[i] = JJ

def compute_pose(x1, u1, t):
    """
    Compute the predicted pose after applying controls.
    x1: [x, y, theta] (initial pose)
    u1: [v_x, theta_dot] (control inputs)
    t: time duration
    Returns: [x, y, theta] (predicted pose)
    """
    x, y, theta = x1
    v_x, theta_dot = u1
    
    if abs(theta_dot) > 1e-6:  # Arc motion
        R = v_x / theta_dot
        delta_theta = theta_dot * t
        x_new = x - R * np.sin(theta) + R * np.sin(theta + delta_theta)
        y_new = y + R * np.cos(theta) - R * np.cos(theta + delta_theta)
        theta_new = theta + delta_theta
    else:  # Straight-line motion
        x_new = x + v_x * np.cos(theta) * t
        y_new = y + v_x * np.sin(theta) * t
        theta_new = theta
    # Normalize theta to [-pi, pi]
    # theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new, theta_new])

def Jerr_compute_pose(x1, u1, x2, t):
    """
    Compute the Jacobians of the error function with respect to x1, u1, and x2.
    x1: [x, y, theta] (initial pose)
    u1: [v_x, theta_dot] (control inputs)
    x2: [x, y, theta] (final pose)
    t: time duration
    Returns: Jacobians (J_x1, J_u1, J_x2)
    """
    x, y, theta = x1
    v_x, theta_dot = u1
    
    if abs(theta_dot) > 1e-6:  # Arc motion
        R = v_x / theta_dot
        delta_theta = theta_dot * t
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_dt = np.sin(theta + delta_theta)
        cos_theta_dt = np.cos(theta + delta_theta)
        
        # Jacobian w.r.t x1
        J_x1 = np.array([
            [1, 0, -R * (cos_theta - cos_theta_dt)],
            [0, 1, -R * (sin_theta_dt - sin_theta)],
            [0, 0, 1]
        ])
        
        # Jacobian w.r.t u1
        dR_dv_x = 1 / theta_dot
        dR_dtheta_dot = -v_x / (theta_dot ** 2)
        J_u1 = np.array([
            [dR_dv_x * (-sin_theta + sin_theta_dt), dR_dtheta_dot * (-sin_theta + sin_theta_dt) + R * t * cos_theta_dt],
            [dR_dv_x * (cos_theta - cos_theta_dt), dR_dtheta_dot * (cos_theta - cos_theta_dt) - R * t * sin_theta_dt],
            [0, t]
        ])
        
    else:  # Straight-line motion
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Jacobian w.r.t x1
        J_x1 = np.array([
            [1, 0, -v_x * t * sin_theta],
            [0, 1, v_x * t * cos_theta],
            [0, 0, 1]
        ])
        
        # Jacobian w.r.t u1
        J_u1 = np.array([
            [t * cos_theta, 0],
            [t * sin_theta, 0],
            [0, t]
        ])
    
    # Jacobian w.r.t x2
    J_x2 = np.eye(3)
    
    return -J_x1, -J_u1, J_x2

class DynFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel,x1_key, u1_key, x2_key, t1, t2):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x1, u1, x2 =  values.atVector(x1_key), values.atVector(u1_key), values.atVector(x2_key)
            error = x2 - compute_pose(x1, u1, t2 - t1)
            if jacobians is not None:
                assign_jacobians(jacobians,Jerr_compute_pose(x1, u1, x2, t2 - t1))
            return error
        super().__init__(noiseModel, [x1_key, u1_key, x2_key], error_function) # may change to partial


class PriorFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, x_key, x_val):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x = values.atVector(x_key)
            error = x - x_val
            if jacobians is not None:
                jacobians[0] = np.eye(3)
            return error
        super().__init__(noiseModel, [x_key], error_function) # may change to partial

N = 50
gif_name = "trajectory_following_2.gif"
# Create a graph
isam2 = gtsam.ISAM2(gtsam.ISAM2Params())
initial_estimate = gtsam.Values()
graph = gtsam.NonlinearFactorGraph()


start = np.array([0, 0, 0])
goal = np.array([10, 10, 2*np.pi/3])
for i in range(N-1):
    graph.push_back(DynFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.01])), X(i), U(i), X(i+1), i, i+1))
graph.push_back(PriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001])), X(0), start))
graph.push_back(PriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001])), X(N-1), goal))



initial_estimate.insert(X(0), start)
initial_estimate.insert(X(N-1), goal)
initial_estimate.insert(U(0), np.array([0, 0]))


for i in range(1, N-1):
    initial_estimate.insert(X(i), (goal-start)/N*i + start)
    initial_estimate.insert(U(i), np.array([0, 0]))

# isam2.update(graph, initial_estimate)
# result = isam2.calculateEstimate()

params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

INFERENCE_TIME = -time.time()
result = optimizer.optimize()
print("Inference Time: ", INFERENCE_TIME + time.time())

rst_x = np.array([result.atVector(X(i)) for i in range(N)])
rst_u = np.array([result.atVector(U(i)) for i in range(N-1)])
# print(rst_u)

# plot the result
fig, ax = plt.subplots(figsize=(6, 6))

# for i in range(N):
#     ax.clear()
#     draw_robot(ax, rst_x[i])
#     # plot planned trajectory
#     ax.plot(rst_x[:,0], rst_x[:,1])
#     # plot start and goal
#     ax.arrow(start[0], start[1], 0.6*np.cos(start[2]), 0.6*np.sin(start[2]), head_width=0.3, head_length=0.3, fc='r', ec='r')
#     ax.scatter(start[0], start[1], 5, c='r')
#     ax.text(start[0]-0.5, start[1]-0.5, "start", color='red')
#     ax.arrow(goal[0], goal[1], 0.6*np.cos(goal[2]), 0.6*np.sin(goal[2]), head_width=0.3, head_length=0.3, fc='g', ec='g')
#     ax.scatter(goal[0], goal[1], 5, c='g')
#     ax.text(goal[0]+0.5, goal[1]+0.5, "goal", color='green')

#     ax.set_xlim(-5, 14)
#     ax.set_ylim(-5, 14)
#     plt.draw()
#     plt.pause(0.01)


# plt.show()


def update(i):
    ax.clear()
    draw_robot(ax, rst_x[i])
    # Plot planned trajectory
    ax.plot(rst_x[:, 0], rst_x[:, 1])
    # Plot start and goal
    ax.arrow(start[0], start[1], 0.6 * np.cos(start[2]), 0.6 * np.sin(start[2]), head_width=0.3, head_length=0.3, fc='r', ec='r')
    ax.scatter(start[0], start[1], 5, c='r')
    ax.text(start[0] - 0.5, start[1] - 0.5, "start", color='red')
    ax.arrow(goal[0], goal[1], 0.6 * np.cos(goal[2]), 0.6 * np.sin(goal[2]), head_width=0.3, head_length=0.3, fc='g', ec='g')
    ax.scatter(goal[0], goal[1], 5, c='g')
    ax.text(goal[0] + 0.5, goal[1] + 0.5, "goal", color='green')

    # Set axis limits
    ax.set_xlim(-5, 14)
    ax.set_ylim(-5, 14)
    ax.set_title(f"Frame {i + 1}/{N}")

# Create the animation
ani = FuncAnimation(fig, update, frames=N, interval=50)

# Save as a GIF
ani.save(gif_name, writer=PillowWriter(fps=20))import gtsam
from gtsam.symbol_shorthand import X, U
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import numba

def draw_robot(ax, X, linewidth=2, arrow_length=0.6):
    x, y , theta = X

    d = 0.8
    r = 0.2


    kpts = np.array([[-r, d/2], [r, d/2], [0, d/2], [0, -d/2],[-r, -d/2],[r, -d/2], [0, 0],[arrow_length, 0]])

    tf_kpts = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ kpts.T + np.array([[x], [y]]) # 2 x N


    ax.plot(tf_kpts[0, 0:2], tf_kpts[1,0:2], 'b-', linewidth=linewidth)
    ax.plot(tf_kpts[0, 2:4], tf_kpts[1,2:4], 'b-', linewidth=linewidth)
    ax.plot(tf_kpts[0, 4:6], tf_kpts[1,4:6], 'b-', linewidth=linewidth)
    ax.arrow(tf_kpts[0,6], tf_kpts[1,6], tf_kpts[0,7]-tf_kpts[0,6], tf_kpts[1,7]-tf_kpts[1,6], head_width=0.3, head_length=0.3, fc='k', ec='k')





# factors for gtsam
def assign_jacobians(jacobians: List[np.ndarray],J: List[np.ndarray]):
     for i, JJ in enumerate(J):
          jacobians[i] = JJ

@numba.jit(nopython=True, cache=True)
def compute_pose(x1, u1, t):
    """
    Compute the predicted pose after applying controls.
    x1: [x, y, theta] (initial pose)
    u1: [v_x, theta_dot] (control inputs)
    t: time duration
    Returns: [x, y, theta] (predicted pose)
    """
    x, y, theta = x1
    v_x, theta_dot = u1
    
    if abs(theta_dot) > 1e-6:  # Arc motion
        R = v_x / theta_dot
        delta_theta = theta_dot * t
        x_new = x - R * np.sin(theta) + R * np.sin(theta + delta_theta)
        y_new = y + R * np.cos(theta) - R * np.cos(theta + delta_theta)
        theta_new = theta + delta_theta
    else:  # Straight-line motion
        x_new = x + v_x * np.cos(theta) * t
        y_new = y + v_x * np.sin(theta) * t
        theta_new = theta
    # Normalize theta to [-pi, pi]
    # theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new, theta_new])

@numba.jit(nopython=True, cache=True)
def Jerr_compute_pose(x1, u1, x2, t):
    """
    Compute the Jacobians of the error function with respect to x1, u1, and x2.
    x1: [x, y, theta] (initial pose)
    u1: [v_x, theta_dot] (control inputs)
    x2: [x, y, theta] (final pose)
    t: time duration
    Returns: Jacobians (J_x1, J_u1, J_x2)
    """
    x, y, theta = x1
    v_x, theta_dot = u1
    
    if abs(theta_dot) > 1e-6:  # Arc motion
        R = v_x / theta_dot
        delta_theta = theta_dot * t
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_dt = np.sin(theta + delta_theta)
        cos_theta_dt = np.cos(theta + delta_theta)
        
        # Jacobian w.r.t x1
        J_x1 = np.array([
            [1.0, 0.0, -R * (cos_theta - cos_theta_dt)],
            [0.0, 1.0, -R * (sin_theta_dt - sin_theta)],
            [0.0, 0.0, 1.0]
        ])
        
        # Jacobian w.r.t u1
        dR_dv_x = 1 / theta_dot
        dR_dtheta_dot = -v_x / (theta_dot ** 2)
        J_u1 = np.array([
            [dR_dv_x * (-sin_theta + sin_theta_dt), dR_dtheta_dot * (-sin_theta + sin_theta_dt) + R * t * cos_theta_dt],
            [dR_dv_x * (cos_theta - cos_theta_dt), dR_dtheta_dot * (cos_theta - cos_theta_dt) - R * t * sin_theta_dt],
            [0.0, t]
        ])
        
    else:  # Straight-line motion
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Jacobian w.r.t x1
        J_x1 = np.array([
            [1.0, 0.0, -v_x * t * sin_theta],
            [0.0, 1.0, v_x * t * cos_theta],
            [0.0, 0.0, 1.0]
        ])
        
        # Jacobian w.r.t u1
        J_u1 = np.array([
            [t * cos_theta, 0.0],
            [t * sin_theta, 0.0],
            [0.0, t]
        ])
    
    # Jacobian w.r.t x2
    J_x2 = np.eye(3)
    
    return -J_x1, -J_u1, J_x2

class DynFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel,x1_key, u1_key, x2_key, t1, t2):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x1, u1, x2 =  values.atVector(x1_key), values.atVector(u1_key), values.atVector(x2_key)
            error = x2 - compute_pose(x1, u1, t2 - t1)
            if jacobians is not None:
                assign_jacobians(jacobians,Jerr_compute_pose(x1, u1, x2, t2 - t1))
            return error
        super().__init__(noiseModel, [x1_key, u1_key, x2_key], error_function) # may change to partial


class PriorFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, x_key, x_val):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x = values.atVector(x_key)
            error = x - x_val
            if jacobians is not None:
                jacobians[0] = np.eye(3)
            return error
        super().__init__(noiseModel, [x_key], error_function) # may change to partial

N = 50

# Create a graph
isam2 = gtsam.ISAM2(gtsam.ISAM2Params())
initial_estimate = gtsam.Values()
graph = gtsam.NonlinearFactorGraph()


start = np.array([0, 0, 0])
goal = np.array([10, 10, 2*np.pi/3])
for i in range(N-1):
    graph.push_back(DynFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.01])), X(i), U(i), X(i+1), i, i+1))
graph.push_back(PriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001])), X(0), start))
graph.push_back(PriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001])), X(N-1), goal))



initial_estimate.insert(X(0), start)
initial_estimate.insert(X(N-1), goal)
initial_estimate.insert(U(0), np.array([0, 0]))


for i in range(1, N-1):
    initial_estimate.insert(X(i), (goal-start)/N*i + start)
    initial_estimate.insert(U(i), np.array([0, 0]))

# isam2.update(graph, initial_estimate)
# result = isam2.calculateEstimate()

params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)



INFERENCE_TIME = -time.time()
result = optimizer.optimize()
print("Inference Time: ", INFERENCE_TIME + time.time())

rst_x = np.array([result.atVector(X(i)) for i in range(N)])
rst_u = np.array([result.atVector(U(i)) for i in range(N-1)])
# print(rst_u)

# plot the result
fig, ax = plt.subplots(figsize=(6, 6))

def update(i):
    ax.clear()
    draw_robot(ax, rst_x[i])
    # Plot planned trajectory
    ax.plot(rst_x[:, 0], rst_x[:, 1])
    # Plot start and goal
    ax.arrow(start[0], start[1], 0.6 * np.cos(start[2]), 0.6 * np.sin(start[2]), head_width=0.3, head_length=0.3, fc='r', ec='r')
    ax.scatter(start[0], start[1], 5, c='r')
    ax.text(start[0] - 0.5, start[1] - 0.5, "start", color='red')
    ax.arrow(goal[0], goal[1], 0.6 * np.cos(goal[2]), 0.6 * np.sin(goal[2]), head_width=0.3, head_length=0.3, fc='g', ec='g')
    ax.scatter(goal[0], goal[1], 5, c='g')
    ax.text(goal[0] + 0.5, goal[1] + 0.5, "goal", color='green')

    # Set axis limits
    ax.set_xlim(-5, 14)
    ax.set_ylim(-5, 14)
    ax.set_title(f"Frame {i + 1}/{N}")

# Create the animation
ani = FuncAnimation(fig, update, frames=N-1, interval=50)

# Save as a GIF
ani.save("trajectory_following_2.gif", writer=PillowWriter(fps=20))
plt.close(fig)