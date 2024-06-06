import torch
from collections import OrderedDict
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]):
    return -data['rew'][..., 0]