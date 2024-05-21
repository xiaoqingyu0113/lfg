import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf
from lfg.util import timeit, augment_trajectory, unaugment_trajectory

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUHisModel(nn.Module):
    def __init__(self, input_size, his_len, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.his_len = his_len
        self.gru_cell = nn.GRUCell(input_size* (his_len-1), hidden_size)
        self.fc_out1 = nn.Linear(hidden_size, 32)
        self.fc_out2 = nn.Linear(32, output_size)
        nn.init.constant_(self.fc_out2.weight, 0)  # Set all weights to 0 to stable the initial training
        nn.init.constant_(self.fc_out2.bias, 0)  
        
        self.fc0 = nn.Linear(3, 32)
        self.fc01 = nn.Linear(32, hidden_size)
        self.relu = nn.ReLU()


    def forward(self, x, h, dt, w0=None):
        '''
            x: (batch_size, his_len, 4): 
                - input_size: [B, seq_len, 4 (vx, vy, vz, z)]
        '''
        x = x.view(x.shape[0], -1)

        if w0 is not None:
            h = self.fc01(self.relu(self.fc0(w0)))
        else:
            h = h
        h_new = h + self.gru_cell(x, h) * dt # newtonian hidden states
        v_new =  self.fc_out2(self.relu(self.fc_out1(h_new))) 


        return v_new, h_new

def compute_velocities(stamped_positions):
    '''
    parameters:
        stamped_positions: (B, N, 4) where B is batch size, N is sequence length, last axis is [t, x, y, z]
    '''  
    position_differences = stamped_positions[:,1:,1:] - stamped_positions[:,:-1,1:]
    time_differences = stamped_positions[:,1:,0:1] - stamped_positions[:,:-1,0:1]
    time_differences = time_differences.clamp(min=1e-6)
    velocities = position_differences / time_differences
    return velocities


def gruhis_autoregr(model, data, camera_param_dict, fraction_est, aug_batch=30, aug_angle=2*torch.tensor(torch.pi, device=DEVICE)):
    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()

    # stamped_positions N x [t, x, y ,z]
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)

    # [B, seq_len, 4]
    augmented_stamped_positions, aug_angles = augment_trajectory(stamped_positions, batch=aug_batch, max_angle=aug_angle, device=DEVICE)


    N = int(augmented_stamped_positions.shape[1] * fraction_est)
    his_len = model.his_len

    # Forward pass
    w0 = data[:aug_batch, 6:9].float().to(DEVICE) # w0 is same across all sequence
    h = None

    t_prev = augmented_stamped_positions[0,0,0]
    stamped_history = augmented_stamped_positions[:,:his_len,:] # shape is (B, his_len, 4), where 4 is [t, x, y, z]

    y = []

    for i in range(augmented_stamped_positions.shape[1]):
        t_curr = augmented_stamped_positions[0,i,0] # current time, step i
        dt = t_curr - t_prev

        if i < his_len:
            y.append(augmented_stamped_positions[:,i:i+1,1:])

        # predict the position, @ step i
        elif i == his_len:
            '''
            prepare x_input for the model.
            x_input is (1, his_len-1, 4), the last axis is [vx, vy, vz, z]
            '''
            augmented_velocities = compute_velocities(stamped_history)
            x_input = torch.cat((augmented_velocities, stamped_history[:,1:,3:]), dim=2) # x_input is (B, his_len-1, 4)
            v, h = model(x_input, h, dt, w0=w0) # v is (B, 3), h is (B, hidden_size)
            v = v.unsqueeze(1) # v is (B, 1, 3)
            x = stamped_history[:,-1:,1:] + v * dt


            stamped_history = augmented_stamped_positions[:, 1:i+1,:]
            y.append(x)

        elif i < N:
            augmented_velocities = compute_velocities(stamped_history)
            x_input = torch.cat((augmented_velocities, stamped_history[:,1:,3:]), dim=2) # x_input is (B, his_len-1, 4)
            # use the last hidden state
            v, h = model(x_input, h, dt)
            v = v.unsqueeze(1) # v is (B, 1, 3)
            x = stamped_history[:,-1:,1:] + v * dt


            stamped_history = augmented_stamped_positions[:, i-his_len+1 : i+1, :]
            y.append(x)

        else:
            augmented_velocities = compute_velocities(stamped_history)
            x_input = torch.cat((augmented_velocities, stamped_history[:,1:,3:]), dim=2) # x_input is (B, his_len-1, 4)
            v, h = model(x_input, h, dt)
            v = v.unsqueeze(1) # v is (B, 1, 3)
            x = stamped_history[:,-1:,1:] + v * dt # 

            # x is (B, 1, 3), t_curr is scalar
            t_curr_tensor = torch.full((x.shape[0], 1, 1), t_curr, device=DEVICE) 
            stamped_x = torch.cat((t_curr_tensor, x), dim=2) # shape is (B, 1, 4)

            stamped_history = torch.cat((stamped_history[:,1:,:],stamped_x), dim=1) # shape is (B, his_len, 4)
            y.append(x)

        t_prev = t_curr
    

    y = torch.cat(y, dim=1) # shape is (B, seq_len, 3)   
    y = unaugment_trajectory(y, aug_angles) # Note: here input y does not have time axis

    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]

    batch_uv_pred = []
    for yi in y:
        uv_pred = get_uv_from_3d(yi, cam_id_list, camera_param_dict) # shape is (seq_len, 2)
        batch_uv_pred.append(uv_pred.unsqueeze(0)) 
    batch_uv_pred = torch.cat(batch_uv_pred, dim=0) # shape is (B, seq_len, n_cam, 2)
    return batch_uv_pred

