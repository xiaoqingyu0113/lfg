import torch

class ExtendedKalmanFilter:
    def __init__(self, state_dim, meas_dim, F_func, H_func, Q, R, P, x):
        """
        Initialize the Extended Kalman Filter.
        
        Parameters:
        state_dim: dimension of the state vector
        meas_dim: dimension of the measurement vector
        F_func: nonlinear state transition function
        H_func: nonlinear measurement function
        Q: process noise covariance matrix (state_dim x state_dim)
        R: measurement noise covariance matrix (meas_dim x meas_dim)
        P: initial estimate covariance matrix (state_dim x state_dim)
        x: initial state estimate (state_dim)
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.F_func = F_func  # Nonlinear state transition function
        self.H_func = H_func  # Nonlinear observation function
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # State covariance
        self.x = x  # Initial state estimate
        self.dt = 0

    def compute_jacobian(self, func, x):
        """
        Compute the Jacobian matrix of a function using torch.autograd.functional.jacobian.
        
        Parameters:
        func: function to compute the Jacobian for
        x: input state vector
        
        Returns:
        jacobian: the Jacobian matrix
        """
        return torch.autograd.functional.jacobian(func, x)

    def predict(self, dt):
        """
        Predict the next state and covariance using the state transition function.
        """
        # Compute the Jacobian of the state transition function (F_func) with respect to x
        
        F = self.compute_jacobian(lambda x: self.F_func(x,dt), self.x)

        # Predict state using the nonlinear state transition function
        self.x = self.F_func(self.x, dt)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update the state and covariance using the measurement z.
        
        Parameters:
        z: observation vector
        """
        # Compute the Jacobian of the measurement function (H_func) with respect to x
        H = self.compute_jacobian(self.H_func, self.x)

        # Innovation (residual)
        # print(z.shape)
        # print(self.H_func(self.x).shape)
        # print(self.x.shape)
        y = z - self.H_func(self.x)
        # print(z)
        # print(self.x)

        # Innovation covariance
        # print(H.shape)
        # print(self.P.shape)
        # print(self.R.shape)
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ torch.inverse(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance estimate
        I = torch.eye(self.state_dim).cuda()  # Identity matrix
        self.P = (I - K @ H) @ self.P

    def get_state(self):
        """
        Returns the current state estimate.
        """
        return self.x.cpu().detach()


# Define the state transition function (F_func) and measurement function (H_func)

def F_func(x):
    """
    Nonlinear state transition function.
    For example, a simple 2D constant velocity model with some nonlinearity.
    x = [position_x, velocity_x, position_y, velocity_y]
    """
    dt = 1.0  # time step
    pos_x = x[0] + x[1] * dt + 0.5 * torch.sin(x[0])  # nonlinear transition for x position
    vel_x = x[1]
    pos_y = x[2] + x[3] * dt + 0.5 * torch.sin(x[2])  # nonlinear transition for y position
    vel_y = x[3]
    return torch.tensor([pos_x, vel_x, pos_y, vel_y])


def H_func(x):
    """
    Nonlinear measurement function.
    For example, we observe positions with some nonlinearity.
    """
    pos_x = x[0] + 0.1 * torch.sin(x[0])  # Nonlinear observation of x position
    pos_y = x[2] + 0.1 * torch.sin(x[2])  # Nonlinear observation of y position
    return torch.tensor([pos_x, pos_y])

if __name__ == '__main__':
        # Initialize matrices and state

        # Process noise covariance (Q)
        Q = torch.eye(4) * 0.01

        # Measurement noise covariance (R)
        R = torch.eye(2) * 0.1

        # Initial estimate of state covariance (P)
        P = torch.eye(4)

        # Initial state (position_x, velocity_x, position_y, velocity_y)
        x_init = torch.tensor([0.0, 1.0, 0.0, 1.0])

        # Create the Extended Kalman Filter object
        filter = ExtendedKalmanFilter(
            state_dim=4,
            meas_dim=2,
            F_func=F_func,
            H_func=H_func,
            Q=Q,
            R=R,
            P=P,
            x=x_init
        )

        # Simulate some measurements (noisy observations of the positions)
        measurements = [
            torch.tensor([1.0, 0.5]),
            torch.tensor([2.0, 1.5]),
            torch.tensor([3.0, 2.0]),
            torch.tensor([4.0, 2.5]),
        ]

        for z in measurements:
            filter.predict()  # Predict the next state
            filter.update(z)  # Update with the new measurement
            print("Updated state estimate:", filter.get_state().numpy())
