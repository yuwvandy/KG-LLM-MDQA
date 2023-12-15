import random
import numpy as np
import torch



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    


def move_to_cuda(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        # elif isinstance(maybe_tensor, list):
        #     return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def move_to_cuda2(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        # elif isinstance(maybe_tensor, list):
        #     return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    def filter(x): return x[7:] if x.startswith('module.') else x
    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict)
    
    return model