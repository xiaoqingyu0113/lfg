import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")*10
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=32,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=3,    # Block expansion factor
).to("cuda")
y = model(x)

print(x[0, :10, 0 ])
print(y[0, :10, 0 ])
