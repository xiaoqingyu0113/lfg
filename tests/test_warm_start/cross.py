import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
#import grucell from torch.nn.GRUCell   



class MultiplyLayer32(nn.Module):
    def __init__(self, hidden_size=32):
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
            nn.Linear(hidden_size, 3)
        )
    def forward(self, x):
        x = x/10.0
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = x + self.layer3(x)*x
        x = self.dec(x) 
        return x

class WidePerceptron1024(nn.Module):
    def __init__(self, hidden_size = 1024):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

def generate_data(batch_size):
    return torch.randn(batch_size, 6, device='cuda')  *10



def compute_gt_cross(x):
    u = x[:, :3]
    v = x[:, 3:]
    return torch.linalg.cross(u, v)


def get_model(choose_model):
    model = getattr(sys.modules[__name__], choose_model)()
        
    model = model.cuda()  # Move model to GPU
    return model


def train(choose_model, epoch_num = 400):
    epoch_num = 400
    batch_size = 16
    model = get_model(choose_model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criteria = torch.nn.MSELoss()

    loss_container = []
    for epoch in range(epoch_num):
        optim.zero_grad()
        x =generate_data(batch_size)
        y = model(x)
        y_gt = compute_gt_cross(x)  
        loss = criteria(y, y_gt)
        loss.backward()
        optim.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                x = generate_data(batch_size)
                y = model(x)
                y_gt = compute_gt_cross(x) 
                loss_valid = criteria(y, y_gt)
                loss_container.append(loss_valid.item())
                compared = torch.cat([y, y_gt], dim=-1)
                print(compared)
    return loss_container

if __name__ == '__main__':
    
    test_model = ['MultiplyLayer32', 'WidePerceptron1024']

    loss_history = train('MNN')
    print(loss_history)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for model in test_model:
    #     loss_history = train(model)
    #     ax.plot(loss_history, label=model)
    # ax.set_yscale('log')
    # ax.legend()
    # plt.show()

           
