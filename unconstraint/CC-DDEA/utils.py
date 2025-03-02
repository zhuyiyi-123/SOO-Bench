from typing import Tuple, Union, List
import numpy as np
import torch
from torch.utils.data import Dataset

def toTensor(*data) -> Union[torch.Tensor, List[torch.Tensor]]:
    ret = []
    for val in data:
        if isinstance(val, torch.Tensor):
            pass
        elif isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        else:
            val = torch.tensor(val)
        ret.append(val)
    if len(ret) == 0:
        return None
    elif len(ret) == 1:
        return ret[0]
    else:
        return ret


def split_dataset(dataset: Dataset, train_rate: float = 0.8):
    n = len(dataset)
    n_train = int(train_rate * n)
    n_val = n - n_train
    return torch.utils.data.random_split(
        dataset,
        [n_train, n_val]
    )

def select_m_different_indexes_randomly_np(
    n: int,
    m: int,
    times: int
) -> np.ndarray:
    '''
    Select m different indexes from [0, n), and it will be repeated `times` times.

    Args:
        n: length of indexes
        m: the number of index to be picked at one time, 1 <= m <= n
        times: the number of pick
    '''
    idxes = np.arange(n)
    ret = []
    for _ in range(times):
        np.random.shuffle(idxes)
        ret.append(idxes[0: m].copy())
    return np.array(ret)

def normal(data: Union[torch.Tensor, np.ndarray], lower_bound=None, upper_bound=None):
    d = data.ndim
    if d > 2:
        raise Exception('Error dimension')
    
    if d == 1:
        toFlatten = True
        data = data.reshape(-1, 1)
    else:
        toFlatten = False

    if lower_bound == None:
        if isinstance(data, torch.Tensor):
            lower_bound = torch.min(data, dim=0)[0]
        elif isinstance(data, np.ndarray):
            lower_bound = np.min(data, axis=0)
    if upper_bound == None:
        if isinstance(data, torch.Tensor):
            upper_bound = torch.max(data, dim=0)[0]
        elif isinstance(data, np.ndarray):
            upper_bound = np.max(data, axis=0)
    
    data = ((data - lower_bound) / (upper_bound - lower_bound) - 0.5) * 2
    if toFlatten:
        data = data.reshape(-1)
    return data, lower_bound, upper_bound

def denormal(data, lower_bound, upper_bound):
    return (data * 2 + 0.5) * (upper_bound - lower_bound) + lower_bound
