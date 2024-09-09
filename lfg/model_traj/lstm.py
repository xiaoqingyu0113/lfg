import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.z0 = 0.010
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.hidden_size = 128
        self.layer_c0 = nn.Linear(3, self.hidden_size)
        self.lstm = nn.LSTM(4,128,1, batch_first=True, proj_size=3)
        self.fc = nn.Linear(self.hidden_size, 3)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    
    def forward(self, pN, dt, w0):
        '''
        pN should be in shape [b, seq_len, 3]
        dt should be in shape [b, seq_len]
        w0 should be in shape [b, 3]
        '''

        # concat pN and dt
        b = pN.shape[0]
        # print(pN.shape)
        # print(dt.shape)
        x = torch.cat([pN, dt.unsqueeze(-1)], dim=-1)
        h0 = torch.zeros(1, b,3).to(pN.device)
        c0 = self.layer_c0(w0).unsqueeze(0)
        pN, _ = self.lstm(x, (h0, c0))

   
        
        return pN
    



def autoregr_LSTM(data, model, est, cfg):
    '''
    data = [b, seq_len, 11]
    11: [traj_idx, time_stamp, p_xyz, v_xyz, w_xyz]
    '''
    batch_size = data.shape[0]
    seq_len = data.shape[1]

    tN = data[:,:, 1] - data[:, 0:1, 1] # start from 0
    pN = data[:,:, 2:5]
    
    w0 = data[:, 0, 8:11]

    dt = torch.diff(tN, dim=-1)
    dt = torch.cat([torch.zeros(batch_size, 1).to(pN.device), dt], dim=-1)

    obs_len = est.size
    p_pred = model(pN[:,:obs_len,:], dt[:, :obs_len], w0)
    p_pred = torch.cat([pN[:,0:1,:], p_pred], dim=1) # obs_len + 1

    # auto regressive prediction
    for i in range(obs_len+1, seq_len):
        p_pred = model(p_pred, dt[:,:i], w0)
        p_pred = torch.cat([pN[:,0:1,:], p_pred], dim=1)
    
    return p_pred
