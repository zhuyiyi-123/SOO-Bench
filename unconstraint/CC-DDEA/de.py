from typing import Callable, Union
import numpy as np
try:
    import torch
except ModuleNotFoundError as e:
    pass
from sampling import sample_within_bounds
import utils

def de_best_1_bin_np(
    pop: np.ndarray,
    lower_bound: Union[int, float, np.ndarray],
    upper_bound: Union[int, float, np.ndarray],
    f: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    F: float = 0.5,
    c: float = 0.9,
):
    n, d = pop.shape

    y_pop = f(pop).reshape(-1)
    for _ in range(max_iter):
        r1 = np.argmin(y_pop)
        x_best = np.copy(pop[r1]).reshape()

        idxes = utils.select_m_different_indexes_randomly_np(n, 2, n)
        r2, r3 = idxes[:, 0], idxes[:, 1]

        # variant vector
        v = pop[r1] + F * (pop[r2] - pop[r3])
        j_r = np.random.randint(0, d, size=[n, d])
        r = np.random.rand(n, d)
        # children
        u = np.where((r < c) | (j_r == np.arange(d)), v, pop)
        # using the random strategy to fix out of range
        u = np.where(
            (lower_bound <= u) & (u <= upper_bound),
            u,
            sampling.rs_np(n, d, lower_bound, upper_bound),
        )
        # using the bound strategy to fix out of range
        # u = np.clamp(u, lower_bound, upper_bound)
        # update the population
        y_u = f(u).reshape(-1)
        res = (y_u < y_pop).reshape(-1)
        pop = np.where(res.reshape(-1, 1), u, pop)
        y_pop = np.where(res, y_u, y_pop)
    return pop

def de_rand_1_bin_np(
    pop: np.ndarray,
    lower_bound: Union[int, float, np.ndarray],
    upper_bound: Union[int, float, np.ndarray],
    f: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    F: float = 0.5,
    c: float = 0.9,
) -> np.ndarray:
    '''DE/rand/1/bin

    Args:
        pop: population
        f: fitness function
        max_iter: the maximum iteration
        F: step
        c: crossover rate

    Return:
        the final populations

    Formula:
        v_i = x_{r_1} + F(x_{r_2} - x_{r_3})
        u_i =
            \begin{cases}
                v_{ij} & U(0, 1) < c \text{OR} j = J_r \\
                x_{ij}
            \end{cases}
        where J_r is a integer of U(1, d)
    '''
    n, d = pop.shape
    if max_iter <= 0:
        return np.empty([0, d], dtype=pop.dtype)
    
    if isinstance(lower_bound, np.ndarray):
        lower_bound = lower_bound.reshape(1, -1)
    if isinstance(upper_bound, np.ndarray):
        upper_bound = upper_bound.reshape(1, -1)


    y_pop = f(pop).reshape(-1)
    for _ in range(max_iter):
        idxes = utils.select_m_different_indexes_randomly_np(n, 3, n)
        r1, r2, r3 = idxes[:, 0], idxes[:, 1], idxes[:, 2]
        # variant vector
        v = pop[r1] + F * (pop[r2] - pop[r3])
        j_r = np.random.randint(0, d, size=[n, d])
        r = np.random.rand(n, d)
        # children
        u = np.where((r < c) | (j_r == np.arange(d)), v, pop)
        # using the random strategy to fix out of range
        u = np.where(
            (lower_bound <= u) & (u <= upper_bound),
            u,
            sampling.rs_np(n, d, lower_bound, upper_bound),
        )
        # using the bound strategy to fix out of range
        # u = np.clamp(u, lower_bound, upper_bound)
        # update the population
        y_u = f(u).reshape(-1)
        res = (y_u < y_pop).reshape(-1)
        pop = np.where(res.reshape(-1, 1), u, pop)
        y_pop = np.where(res, y_u, y_pop)
    return pop

@torch.no_grad()
def de_rand_1_bin_torch(
    pop: torch.Tensor,
    group_rules: bool, 
    lower_bound: Union[int, float, torch.Tensor],
    upper_bound: Union[int, float, torch.Tensor],
    f: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int,
    F: float = 0.5,
    c: float = 0.9
) -> torch.Tensor:
    n, d = pop.shape
    if max_iter <= 0:
        return torch.empty([0, d], dtype=pop.dtype, device=pop.device)

    if isinstance(lower_bound, torch.Tensor):
        lower_bound = lower_bound.reshape(1, -1)
    if isinstance(upper_bound, torch.Tensor):
        upper_bound = upper_bound.reshape(1, -1)

    y_pop = f(pop).reshape(-1)
    for _ in range(max_iter):
        idxes = torch.from_numpy(utils.select_m_different_indexes_randomly_np(n, 3, n)).to(pop.device)
        r1, r2, r3 = idxes[:, 0], idxes[:, 1], idxes[:, 2]
        # variant vector
        v = pop[r1] + F * (pop[r2] - pop[r3])
        j_r = torch.randint(0, d, size=[n, d], device=pop.device)
        r = torch.rand(n, d, dtype=pop.dtype, device=pop.device)
        # children
        u = torch.where(
            (r < c) | (j_r == torch.arange(d, device=pop.device)),
            v,
            pop
        )
        # using the random strategy to fix out of range
        # for i in range(len(lower_bound)):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        lower_bound_new = torch.tensor([data for data, flag in zip(lower_bound, group_rules) if flag]).to(device)
        upper_bound_new = torch.tensor([data for data, flag in zip(upper_bound, group_rules) if flag]).to(device)
        u = torch.where(
            (lower_bound_new <= u) & (u <= upper_bound_new),
            u,
            sample_within_bounds(lower_bound_new, upper_bound_new, n, d, device=device),
        )
        # using the bound strategy to fix out of range
        # pytorch 1.8.2 has no clamp(min: torch.Torch, max:torch.Tensor)
        # u.clamp_(min=lower_bound, max=upper_bound)
        
        # update the population
        y_u = f(u).reshape(-1)
        res = (y_u < y_pop).reshape(-1)
        pop = torch.where(res.reshape(-1, 1), u, pop)
        y_pop = torch.where(res, y_u, y_pop)
    return pop

def de_rand_1_bin(
    pop: Union[np.ndarray, torch.Tensor],
    group_rules: Union[bool],
    lower_bound: Union[int, float, np.ndarray, torch.Tensor],
    upper_bound: Union[int, float, np.ndarray, torch.Tensor],
    f: Union[Callable[[np.ndarray], np.ndarray], Callable[[torch.Tensor], torch.Tensor]],
    max_iter: int,
    F: float = 0.5,
    c: float = 0.9,
) -> Union[np.ndarray, torch.Tensor]:
    if hasattr(pop, 'numpy'):
        return de_rand_1_bin_torch(pop, group_rules, lower_bound, upper_bound, f, max_iter, F, c)
    else:
        return de_rand_1_bin_np(pop, lower_bound, upper_bound, f, max_iter, F, c)