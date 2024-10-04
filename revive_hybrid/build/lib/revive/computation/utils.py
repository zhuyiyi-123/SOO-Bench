import torch
import numpy as np
from numbers import Number
from torch.functional import F
from typing import Callable, Tuple

from revive.data.batch import Batch
from revive.computation.dists import ReviveDistribution


def soft_clamp(x : torch.Tensor, _min=None, _max=None) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

def maintain_gradient_hard_clamp(x : torch.Tensor, _min=None, _max=None) -> torch.Tensor:
    # clamp tensor values with hard constrain while mataining the gradient
    output = torch.clamp(x, _min, _max)
    left_mask = torch.zeros(x.shape, dtype=bool) if _min is None else x < _min
    right_mask = torch.zeros(x.shape, dtype=bool) if _max is None else x > _max
    mask = torch.logical_or(left_mask, right_mask).to(output)
    output = output + mask * (x - x.detach())
    return output

def safe_atanh(x):
    return torch.atanh(maintain_gradient_hard_clamp(x, -0.999, 0.999))

def get_input_from_names(batch : Batch, names : list):
    input = []
    for name in names:
        input.append(batch[name])
    return torch.cat(input, dim=-1)

def get_input_from_graph(graph, 
                         output_name : str, 
                         batch_data : Batch):
    input_names = graph[output_name]
    inputs = []
    for input_name in input_names:
        inputs.append(batch_data[input_name])
    return torch.cat(inputs, dim=-1)

def get_sample_function(deterministic : bool) -> Callable[[ReviveDistribution], torch.Tensor]:
    if deterministic:
        sample_fn = lambda dist: dist.mode
    else:
        sample_fn = lambda dist: dist.rsample()
    return sample_fn
    
def to_numpy(x : Tuple[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return x

def to_torch(x : Tuple[np.ndarray, torch.Tensor], dtype=torch.float32, device : str = "cpu") -> torch.Tensor:
    """Return an object without torch.Tensor."""

    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        x = to_torch(np.asanyarray(x), dtype, device)
    else:  # fallback
        x = np.asanyarray(x)
        if issubclass(x.dtype.type, (np.bool_, np.number)):
            x = torch.from_numpy(x).to(device)
            if dtype is not None:
                x = x.type(dtype)
        else:
            raise TypeError(f"object {x} cannot be converted to torch.")
    return x
