from typing import List
import numpy as np

def get_partial_decision(d: int, n: int) -> np.ndarray:
    '''

    Args:
        d: the number of dimension in total
        n: the number of dimension to be picked
    '''
    i = np.random.permutation(np.arange(d))
    ret = np.zeros([d], dtype=bool)
    ret[i[:n]] = True
    return ret

def get_group_rules_by_given(ds: List[int], indexes: np.ndarray) -> List[np.ndarray]:
    '''Get group rules by split the indexes sequentially

    Attension: if d is not divided by n_group, the last group will be the large one.
    Args:
        ds: the dimension of each group
        indexes: the index array
    '''
    if np.sum(ds) != len(indexes):
        return None
    ret = [np.zeros([len(indexes)], dtype=bool) for _ in range(len(ds))]
    idx = 0
    for i, rule in enumerate(ret):
        rule[indexes[idx: idx + ds[i]]] = True
        idx += ds[i]
    return ret

def get_group_rules_evenly(n_group: int, indexes: np.ndarray) -> List[np.ndarray]:
    '''Get group rules by split the indexes sequentially

    Attension: if d is not divided by n_group, the last group will be the large one.
    Args:
        n_group: the number of group
        indexes: the index array
    '''
    d = len(indexes)
    size = d // n_group
    ds = [size] * n_group
    if d % n_group != 0:
        ds[-1] += d % n_group
    return get_group_rules_by_given(ds, indexes)

def get_group_rules_sequentially(d: int, n_group: int) -> List[np.ndarray]:
    '''Get group rules sequentially

    Attension: if d is not divided by n_group, the last group will be the large one.
    Args:
        d: dimension
        n_group: the number of group
    '''
    return get_group_rules_evenly(n_group, np.arange(d))


def get_group_rules_randomly(d: int, n_group: int) -> List[np.ndarray]:
    '''Get group rules sequentially

    Attension: if d is not divided by n_group, the last group will be the large one.
    Args:
        d: dimension
        n_group: the number of group
    '''
    i = np.random.permutation(np.arange(d))
    return get_group_rules_evenly(n_group, i)

def get_overlap_group_rules_randomly(d_min: int, d_max: int, d: int, n_group: int) -> List[np.ndarray]:
    d_min, d_max = min(d_min, d), min(d_max, d)
    if d_min > d_max or d_max * n_group < d:
        return None
    
    seq = []
    for _ in range(n_group):
        indexes = np.arange(d)
        np.random.shuffle(indexes)
        seq.append(indexes)
    seq = np.hstack(seq)

    ret = [np.zeros(d, dtype=bool) for _ in range(n_group)]
    idx = 0
    for rule in ret:
        rule[seq[idx : idx + d_min]] = True
        idx += d_min
    while True:
        for rule in ret:
            diff = d_max - np.count_nonzero(rule)
            if diff == 0:
                continue
            cnt = np.random.randint(diff) + 1
            rule[seq[idx : idx + cnt]] = True
            idx += cnt
        if idx >= d:
            break
    return ret
