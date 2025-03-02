import math
from typing import Union
import numpy as np
try:
    import torch
except ModuleNotFoundError as e:
    pass

def linear(r: Union[np.ndarray, torch.Tensor]):
    return r

def gaussian(r: Union[np.ndarray, torch.Tensor]):
    return math.e ** (-1. * (r ** 2))

def quadratic(r: Union[np.ndarray, torch.Tensor]):
    return 1. + r ** 2

def inverse_quadratic(r: Union[np.ndarray, torch.Tensor]):
    return 1. / (1. + r ** 2)

def multiquadric(r: Union[np.ndarray, torch.Tensor]):
    return (1. + r ** 2) ** 0.5

def inverse_multiquadric(r: Union[np.ndarray, torch.Tensor]):
    return 1. / ((1. + r ** 2) ** 0.5)

funcs = {
    'linear': linear,
    'gaussian': gaussian,
    'quadratic': quadratic,
    'inverse_quadratic': inverse_quadratic,
    'multiquadric': multiquadric,
    'inverse_multiquadric': inverse_multiquadric,
}

