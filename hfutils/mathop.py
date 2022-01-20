import torch
import numpy as np

def u_power(arr):
    if isinstance(arr, torch.Tensor):
        return torch.float_power(arr)
    else:
        return np.power(arr)

def u_max(arr, axis=None):
    if isinstance(arr, torch.Tensor):
        return torch.max(arr, dim=axis)[0]
    else:
        return np.max(arr, axis=axis)

def u_any(arr):
    if isinstance(arr, torch.Tensor):
        return torch.max(arr)
    else:
        return np.any(arr)