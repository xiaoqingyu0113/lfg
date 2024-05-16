import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf
from mamba_ssm import Mamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, output_size):
        super().__init__()
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        self.fc_out1 = nn.Linear(d_model, output_size)

    def forward(self, x):
        '''
            x: (batch_size, seq_len, input_size): 
                - input_size: [time_stamp, x,y,z]
        '''

        y = self.mamba(x) # shape is (batch_size, seq_len, d_model)
        vel = y.view(-1, y.shape[-1]) # shape is (batch_size*seq_len, d_model)
        vel = self.fc_out1(vel) # shape is (batch_size*seq_len, output_size)
        vel = vel.view(y.shape[0], y.shape[1], -1) # shape is (batch_size, seq_len, output_size)

        t = x[:,:,0:1]
        dt = t[:,1:,:] - t[:,:-1,:] # 1, seq_len-1, 1
        dt = torch.cat([torch.zeros(dt.shape[0],1,1).to(DEVICE), dt], dim=1) # 1, seq_len, 1


        return x[:,:,1:4] + vel * dt # shape is (batch_size, seq_len, 3)
    
def mamba_pass(model, data, camera_param_dict):
    '''
        Predict the result
        model: Mamba
            input: (batch, seq_len, model_dim)
        data: [seq_len, input_size]
        camera_param_dict: Dict[str, CameraParam]
    '''
    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE) # shape is (seq_len, 4)
    
    # Forward pass
    w0 = data[0, 6:9].float().to(DEVICE)
    w0  = w0.repeat(stamped_positions.shape[0], 1)
    stamped_data = torch.hstack([stamped_positions,w0]) # shape is (seq_len, 7)
    stamped_data = stamped_data.unsqueeze(0) # shape is (1, seq_len, 7)


    y = model(stamped_data).squeeze(0) # shape is (seq_len, 3)


    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)

    return uv_pred




def mamba_autoregr(model, data, camera_param_dict, fraction_est):
    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)

    N = int(stamped_positions.shape[0] * fraction_est)

    # Forward pass
    w0 = data[0, 6:9].float().to(DEVICE)
    w0N  = w0.repeat(stamped_positions.shape[0], 1)
    stamped_data = torch.hstack([stamped_positions,w0N]) # shape is (seq_len, 7)
    stamped_data = stamped_data.unsqueeze(0) # shape is (1, seq_len, 7)

    # t_prev = stamped_positions[0,0]

    x = stamped_data[:,:N,:] # first N steps
    for i in range(N, stamped_positions.shape[0]):
        # t_curr = stamped_positions[i,0] # current time, step i
        # dt = t_curr - t_prev

        # predict the position, @ step i
        y_new = model(x) # return shape is (1, seq_len, 3)

        # update x
        t = stamped_data[:,i:i+1,0:1] # shape is (1, 1, 1)
        w0 = stamped_data[:,0:1,4:] # shape is (1, 1, 3)

        y_new_last = torch.cat([t, y_new[:,-1:,:], w0], dim=2) # append w0 and t, shape is (1, 1, 7)

        x = torch.cat([x, y_new_last], dim=1) # update auto-regressive input

    y_new = torch.cat([x[:,0:1,1:4], y_new], dim=1) # shape is (1, seq_len, 7)
    y = y_new.squeeze(0) # shape is (seq_len, 3)
    

    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)
    
    return uv_pred

