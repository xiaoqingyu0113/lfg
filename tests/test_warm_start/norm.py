import torch
import torch.nn as nn



class Norm(nn.Module):
    def __init__(self, num_layers=4, hidden_size=128):
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
        
class MNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        hidden_size = 16
        self.layer1 = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        x = x/ 50.0
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = x + self.layer3(x)*x
        x = self.dec(x)*50.0
        return x
        
def generate_data():
    return torch.randn(batch_size, 6, device='cuda')  *100-50



def compute_norm(x):
    return torch.linalg.norm(x,dim=-1, keepdim=True)


if __name__ == '__main__':
    epoch_num = 400
    batch_size = 16
    # model = Norm(num_layers=4, hidden_size=128)
    model= MNN()
    model = model.cuda()  # Move model to GPU

    optim = torch.optim.Adam(model.parameters(), lr=3e-2)
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


           
