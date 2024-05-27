import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf
import theseus as th
from functools import partial

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def error_p_prior(optim_vars, aux_vars):
    p_opt = optim_vars[0].tensor
    p_prior = aux_vars[0].tensor
    return p_opt - p_prior

def error_p(optim_vars, aux_vars):
    p_prev, v_prev, p = optim_vars
    dt = aux_vars[0]

    p_prev = p_prev.tensor
    v_prev = v_prev.tensor
    p = p.tensor
    dt = dt.tensor

    return p - p_prev - v_prev * dt

def error_v_with_model(model_v, optim_vars, aux_vars):
    p_prev, v_prev, w_prev, v = optim_vars
    dt = aux_vars[0]

    p_prev = p_prev.tensor
    v_prev = v_prev.tensor
    w_prev = w_prev.tensor
    v = v.tensor
    dt = dt.tensor

    v_dot = model_v(p_prev, v_prev, w_prev)

    return v - v_prev - v_dot * dt

def error_w_with_model(model_w, optim_vars, aux_vars):
    p_prev, v_prev, w_prev, w = optim_vars
    dt = aux_vars[0]

    p_prev = p_prev.tensor
    v_prev = v_prev.tensor
    w_prev = w_prev.tensor
    w = w.tensor
    dt = dt.tensor

    w_dot = model_w(p_prev, v_prev, w_prev)

    return w - w_prev - w_dot * dt

def state_estimation(stamped_positions,w0, model_v, model_w):
    t_data = stamped_positions[:, 0]
    p_data = stamped_positions[:, 1:]
    objective = th.Objective()
    p_prior_cost_weight = th.ScaleCostWeight(0.1)
    p_cost_weight = th.ScaleCostWeight(0.010)
    v_cost_weight = th.ScaleCostWeight(0.3)
    w_cost_weight = th.ScaleCostWeight(5.0)
    w0_prior_cost_weight = th.ScaleCostWeight(5.0)
    error_v = partial(error_v_with_model, model_v)
    error_w = partial(error_w_with_model, model_w)
    theseus_inputs = {}

    N = stamped_positions.shape[0]

    for i in range(N):
        p = th.Point3( name=f'p{i}')
        v = th.Point3(name=f'v{i}')
        w = th.Point3(name=f'w{i}')

        p_prior = th.Point3(p_data[i].view(1,-1), name=f'p{i}_prior')

        # update theseus initial inputs
        theseus_inputs.update({f'p{i}': p_data[i].view(1,-1).clone()})
        theseus_inputs.update({f'v{i}': torch.zeros(1,3, device=DEVICE).clone()})
        theseus_inputs.update({f'w{i}': torch.zeros(1,3, device=DEVICE).clone()})
        theseus_inputs.update({f'p{i}_prior': p_data[i].view(1,-1).clone()})

        # add prior cost
        objective.add(
        th.Difference(p, p_prior, p_prior_cost_weight, name=f"p_proir_cost{i}")
        )

        # add spin prior to the first frame only
        if i ==0:
            w0_prior = th.Point3(w0.view(1,-1), name=f'w0_prior')
            objective.add(
                th.Difference(w, w0_prior, w0_prior_cost_weight, name=f"w0_proir_cost0")
            )
            theseus_inputs.update({f'w0_prior': w0.view(1,-1).clone()})

        if i > 0:
            dt = th.Variable((t_data[i]-t_data[i-1]).view(1,1), name=f'dt{i-1}')
            theseus_inputs.update({f'dt{i-1}': (t_data[i]-t_data[i-1]).view(1,1).clone()})

            


            # add p objective
            objective.add(th.AutoDiffCostFunction([p_prev, v_prev, p], error_p, 3, aux_vars=(dt,), 
                                                  cost_weight=p_cost_weight, name=f'p_cost{i}'))
            # add v objective
            objective.add(th.AutoDiffCostFunction([p_prev, v_prev, w_prev, v], error_v, 3, aux_vars=(dt,), 
                                                  cost_weight=v_cost_weight, name=f'v_cost{i}'))
            # add w objective
            objective.add(th.AutoDiffCostFunction([p_prev, v_prev, w_prev, w], error_w, 3, aux_vars=(dt,), 
                                                  cost_weight=w_cost_weight, name=f'w_cost{i}'))

        p_prev = p
        v_prev = v
        w_prev = w
   
    # for key, value in theseus_inputs.items():
    #     if value.device != DEVICE:
    #         print(f"key: {key}, value: {value.device}, current device: {DEVICE}")

    layer = th.TheseusLayer(th.LevenbergMarquardt(objective,max_iterations=20))
    layer.to(DEVICE)
    solution, info = layer.forward(input_tensors=theseus_inputs)
    return solution


def set_param(m, val):
    if isinstance(m, nn.Sequential):
        for mm in m:
            set_param(mm, val)
    elif isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -val, val)
        torch.nn.init.uniform_(m.bias, -val, val)


class OptimModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_vw = nn.Sequential(nn.Linear(6, 128), nn.ReLU())
        self.fc_b = nn.Sequential(nn.Linear(1, 128), nn.Sigmoid())
        self.decoder = nn.Linear(128, 3)

        set_param(self.fc_vw, 1e-5)
        set_param(self.fc_b, 1e-5)
        set_param(self.decoder, 1e-5)

    def forward(self, p, v, w):
        vw = torch.cat((v,w), dim=1)
        h_vw = self.fc_vw(vw)

        b = p[:,2:3]
        # print('p shape = ',p.shape)
        g_b = self.fc_b(b)

        h = h_vw #* g_b
        return self.decoder(h)
    
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_v = OptimModel()
        self.model_w = OptimModel()

    def forward(self, p, v, w):
        v_dot = self.model_v(p, v, w)
        w_dot = self.model_w(p, v, w)
        return v_dot, w_dot

def physics_optim_autoregr(model, data, camera_param_dict, config):

    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()

    N = int(data.shape[0] * config.model.estimation_fraction)
    N_truncate = int(data.shape[0] * config.model.use_data)
    data = data[:N_truncate,:]

    # stamped_positions N x [t, x, y ,z]
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)



    w0 = data[0, 6:9].float().to(DEVICE)
    solution = state_estimation(stamped_positions[:N, :], w0, model.model_v, model.model_w)

    y = []
    for i in range(N):
        y.append(solution[f'p{i}'])

    # auto regressive
    vi = solution[f'v{N-1}']
    wi = solution[f'w{N-1}']
    for i in range(N, stamped_positions.shape[0]):
        dt = stamped_positions[i,0] - stamped_positions[i-1,0]
        v_dot, w_dot = model(y[-1], vi, wi)
        vo = vi +  v_dot * dt
        wo = wi +  w_dot * dt
        y.append(y[-1] + vi * dt)

        vi = vo
        wi = wo

    y = torch.cat(y, dim=0) # shape is (seq_len-1, 3)


    # back project to image
    # for cm in camera_param_dict.values():
    #     cm.to_torch(device=DEVICE)
    # cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    # uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)
    
    return y, stamped_positions[:,1:]

if __name__ == '__main__':
    pass