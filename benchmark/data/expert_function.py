import torch
from typing import Dict

def reward_node_function(inputs : Dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs['next_obs'][...,0:1] - inputs['obs'][...,0:1]