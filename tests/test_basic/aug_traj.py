from lfg.util import augment_trajectory, unaugment_trajectory
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device('cpu')

def test_augment_trajectory():
    t = torch.arange(0, 10, 1).float()
    x = torch.arange(0, 10, 1).float()
    y = x*2
    z = x + y
    data = torch.stack((t, x, y, z), dim=1).to(DEVICE)
    print("data.shape = ",data.shape)

    batch = 5
    max_angle = torch.tensor(2*torch.pi).to(DEVICE)

    augmented_data, angle = augment_trajectory(data, batch, max_angle, DEVICE)
    print("augmented_data.shape = ",augmented_data.shape)
    unaugmented_data = unaugment_trajectory(augmented_data[:,:,1:], angle)
    print("unaugmented_data.shape = ",unaugmented_data.shape)

    augmented_data = augmented_data.cpu().numpy()
    unaugmented_data = unaugmented_data.cpu().numpy()
    data = data.cpu().numpy()
    print(unaugmented_data.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[:,1], data[:,2], data[:,3], label="original")
    for i in range(batch):
        ax.plot(augmented_data[i,:,1], augmented_data[i,:,2], augmented_data[i,:,3], label=f"angle {angle[i,0]/torch.pi:.2f}\pi batch {i} augmented")
        ax.scatter(unaugmented_data[i,:,0], unaugmented_data[i,:,1], unaugmented_data[i,:,2], s= 10)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    test_augment_trajectory()
