import torch
from torch import nn
import theseus as th
import numpy as np 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class OptimLayer(nn.Module):
    def __init__(self, model:nn.Module, **kwargs):
        super().__init__()
        self.model = model
        for key, value in kwargs.items():
            setattr(self, key, value)

        # pingpong
        # p_noise = 0.010
        # v_noise = 0.001
        # w_noise = 0.1

        # tennis
        p_noise = 0.010
        v_noise = 0.001
        w_noise = 0.1

        self.p_weight_param = nn.Parameter(1.0/torch.tensor([p_noise, p_noise, p_noise]))
        self.wv_weight_param = nn.Parameter(1.0/torch.tensor([v_noise, v_noise, v_noise, w_noise, w_noise, w_noise]))
        self.w0_noise = nn.Parameter(1.0/torch.tensor([w_noise, w_noise, w_noise]))

        self._build_graph()
        

    def _build_graph(self):
        p_weight = th.DiagonalCostWeight(self.p_weight_param)
        wv_weight = th.DiagonalCostWeight(self.wv_weight_param)
        w0_weight = th.DiagonalCostWeight(self.w0_noise)

        # objective and variables
        objective = th.Objective()
        vars = {}
        
        # build graph
        for i in range(self.size):
            vars.update({f'p{i}': th.Vector(3,  name=f'p{i}')})
            vars.update({f'v{i}': th.Vector(3,  name=f'v{i}')})
            vars.update({f'w{i}': th.Vector(3,  name=f'w{i}')})

            vars.update({f'p_prior{i}': th.Vector(3,  name=f'p_prior{i}')})
            vars.update({f'dt{i}': th.Vector(1,  name=f'dt{i}')})

            objective.add(th.AutoDiffCostFunction([vars[f'p{i}']],
                                                   self._error_pos_prior, 
                                                   3, 
                                                   aux_vars=[vars[f'p_prior{i}']],
                                                   cost_weight=p_weight))
            if i > 0:
                objective.add(th.AutoDiffCostFunction([vars[f'p{i-1}'], vars[f'v{i-1}'], vars[f'p{i}']], 
                                                      self._error_pos, 
                                                      3, 
                                                      aux_vars=[vars[f'dt{i-1}']],
                                                      cost_weight=p_weight))
                
                objective.add(th.AutoDiffCostFunction([vars[f'p{i-1}'], vars[f'v{i-1}'], vars[f'w{i-1}'], vars[f'v{i}'], vars[f'w{i}']],
                                                      self._error_vw, 
                                                      6, 
                                                      aux_vars=[vars[f'dt{i-1}']],
                                                      cost_weight=wv_weight))

        vars.update({'w_prior0': th.Vector(3, name='w_prior0')})
        objective.add(th.AutoDiffCostFunction([vars[f'w{0}']],
                                                   self._error_pos_prior, # same procedure
                                                   3, 
                                                   aux_vars=[vars[f'w_prior{0}']],
                                                   cost_weight= w0_weight))

        optimizer = th.LevenbergMarquardt(objective, max_iterations=self.max_iterations)
        layer = th.TheseusLayer(optimizer)
        return layer
        # self.layer.to(DEVICE)

    def _error_pos_prior(self, optim_vars, aux_vars):
        p0, = optim_vars
        p0_prior = aux_vars[0]
        return p0.tensor - p0_prior.tensor
    
    def _error_pos(self, optim_vars, aux_vars):
        p0, v0, p1 = optim_vars
        dt = aux_vars[0]
        return p1.tensor - (p0.tensor + v0.tensor*dt.tensor)
    
    def _error_vw(self, optim_vars, aux_vars):
        p0, v0, w0, v1, w1 = optim_vars
        dt = aux_vars[0]

        b0 = p0.tensor[:, 2:3]
        v1_est, w1_est = self.model(b0, v0.tensor, w0.tensor, dt.tensor)

        error = torch.cat([v1_est - v1.tensor, w1_est - w1.tensor], dim=-1)

        return error
    
    def forward(self, x, w0= None):
        '''
        args:
            - x = [b, seq_len, 4]
            - w0 = [b, 1, 3]
        outputs:
            - p0 = [b, 1, 3]
            - v0 = [b, 1, 3]
            - w0 = [b, 1, 3]
        '''
        batch, seq_len, _ = x.shape
        if w0 is None:
            w0 = torch.zeros(batch, 1, 3, device=DEVICE)
        w0 = w0[:,0,:]

        dtN = torch.diff(x[:,:,0], dim=1)
        pN = x[:, :, 1:4]

        sol = {}
        sol.update({'w_prior0': w0})
        for i in range(self.size):
            p = pN[:, i, :]
            sol.update({f'p_prior{i}': p.clone()})
            sol.update({f'p{i}': p.clone()})
            sol.update({f'v{i}': torch.rand_like(p, device=DEVICE)})
            sol.update({f'w{i}': w0.clone()})
            if i > 0:
                sol.update({f'dt{i-1}': dtN[:,i-1:i].clone()})

        
        layer = self._build_graph()
        layer.to(x.device)
        if self.allow_grad:   
            sol,info = layer(sol, {'damping': self.damping})
            if np.any(info.status == th.NonlinearOptimizerStatus.MAX_ITERATIONS.FAIL):
                    return None, None, None
        else:
            with torch.no_grad():
                sol,info = layer(sol,{'damping': self.damping})
                if np.any(info.status == th.NonlinearOptimizerStatus.MAX_ITERATIONS.FAIL):
                    return None, None, None

        return sol['p0'].unsqueeze(1), sol['v0'].unsqueeze(1), w0.unsqueeze(1) 
        # return sol[f'p{self.size-1}'].unsqueeze(1), sol[f'v{self.size-1}'].unsqueeze(1), sol['w0'].unsqueeze(1) # temporary, compare with ekf

    




class SlideWindowEstimator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.conv1d = nn.Conv1d(4, 4, 3, padding=1)

        # self.layer_dim: [in_channel, hidden1, hidden2, ..., out_channel]
        self.layer = nn.Sequential()
        for i in range(len(self.layer_dim)-1):
            self.layer.add_module(f'linear{i}', nn.Linear(self.layer_dim[i], self.layer_dim[i+1]))
            self.layer.add_module(f'activation{i}', nn.ReLU())


    def forward(self, x, w0= None):
        # x: [batch, seq_len, 4]
        # 4: [t, x, y, z]
        batch, his_len, _ = x.shape

        p0 = x[:, 0:1, 1:4]
        if w0 is None:
            w0 = torch.zeros(batch, 1, 3, device=DEVICE)

        # conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)

        # mlp
        x = x.reshape(batch, 1, -1)
        x  = self.layer(x)


        return p0, x, w0