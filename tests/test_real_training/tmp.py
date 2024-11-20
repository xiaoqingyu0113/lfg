import numpy as np
import torch
from torch.autograd.functional import jacobian

def jacobian_orthonormal(v,w):
    Jv = np.eye(3)*np.dot(v,w) + np.outer(v,w)
    Jw = np.outer(v,v)  
    return Jv, Jw

def forward_torch(v, w):
    return torch.dot(w, v) * v


# Example usage
v = np.random.rand(3)*5
w = np.array([4.0, 5.0, 6.0])

J = jacobian_orthonormal(v,w)
print(J)

J_torch = jacobian(forward_torch, inputs=(torch.from_numpy(v), torch.from_numpy(w)))
print(J_torch)