import torch
import numpy as np


def to_torch_tensor(val, torch_type, device='cpu', requires_grad=False):
    if type(val) == np.ndarray:
        val = torch.from_numpy(val).to(torch_type)
    elif val is None:
        pass
    elif type(val) != torch.Tensor:
        val = torch.tensor(val,
                           dtype=torch_type,
                           requires_grad=requires_grad)

    return val


def to_numpy_array(val):
    if type(val) == torch.Tensor:
        val = val.detach().numpy()
    elif val is None:
        pass
    elif type(val) != np.ndarray:
        val = np.array(val)

    return val
