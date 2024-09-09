import torch
import torch.nn as nn
class Norm(nn.Module):
    def __init__(self, num_layers=3, hidden_size=128):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(6, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])  # Use ModuleList
        self.fc1 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.fc0(x)
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.fc1(x)
        return x
class Cross(nn.Module):
    def __init__(self, num_layers=3, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(6, hidden_size)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x3 = self.fc1(x)
        x4 = self.fc2(x)
        x = x3*x4
        x = self.fc(x)
        return x

class Gate(n.Module):
    def __init__(self, num_layers=3, hidden_size=128):
        super().__init__()
        self.cross = Cross(num_layers, hidden_size)
        self.norm = Norm()
        
def generate_data():
    return torch.randn(batch_size, 6, device='cuda')  *100-50



def compute_norm(x):
    return torch.linalg.norm(x,dim=-1, keepdim=True)



epoch_num = 1000
batch_size = 64
model = Norm(num_layers=2, hidden_size=128)
model = model.cuda()  # Move model to GPU

optim = torch.optim.Adam(model.parameters(), lr=1e-2)
criteria = torch.nn.MSELoss()
for epoch in range(epoch_num):
    optim.zero_grad()
    x =generate_data()
    y = model(x)
    y_gt = compute_norm(x)  
    loss = criteria(y, y_gt)
    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        print('-----------------')
        print(f"Epoch {epoch}, loss: {loss.item()}")
        with torch.no_grad():
            x = generate_data()
            y = model(x)
            y_gt = compute_norm(x) 

            print("y_gt: ", y_gt[0])
            print("y: ", y[0])


           
