import numpy as np
import torch
from torch.autograd.functional import jacobian

def jacobian_orthonormal(R_r,v):
    v0, v1, v2 = v
    o = np.float32(0.0)
    return np.array([[v0, o, o, v1, o, o, v2, o, o],
                    [o, v0, 0, o, v1, o, o, v2, o],
                    [o, o, v0, o, o, v1, o, o, v2]]),R_r.reshape(3,3).T

def forward_torch(R, v):
    R = R.reshape(3,3)
    return R.T@v

np.set_printoptions(precision=4)
# Example usage
v = np.random.rand(3)*5
w = np.array([4.0, 5.0, 6.0])
R = np.random.rand(9)*5

J = jacobian_orthonormal(R,v)
print(J[0])
print(J[1])

J_torch = jacobian(forward_torch, inputs=(torch.from_numpy(R), torch.from_numpy(v)))
print(J_torch)