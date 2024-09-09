import torch
import torch.nn as nn
from tests.test_basic.kan import FastKANLayer, FastKAN

class Devide(nn.Module):
    def __init__(self, num_layers=3, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(6, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x1= self.fc1(x)
        x2= self.fc2(x)
        x = self.fc3(x1*x2)
        return x
    
def generate_data():
    return torch.randn(batch_size, 6, device='cuda')  *10 + 0.001

def compute_devide(x):
    return x[:, :3] / x[:, 3:]

epoch_num = 3000
batch_size = 16
model = Devide(num_layers=3, hidden_size=128)
model = model.cuda()  # Move model to GPU

optim = torch.optim.Adam(model.parameters(), lr=1e-2)
criteria = torch.nn.MSELoss()
for epoch in range(epoch_num):
    optim.zero_grad()
    x = generate_data()
    y = model(x)
    y_gt = compute_devide(x)  

    loss = criteria(y, y_gt)
    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        print('-----------------')
        print(f"Epoch {epoch}, loss: {loss.item()}")
        with torch.no_grad():
            x = generate_data()
            y = model(x)
            y_gt = compute_devide(x) 

            print("y_gt: ", y_gt[0])
            print("y: ", y[0])
