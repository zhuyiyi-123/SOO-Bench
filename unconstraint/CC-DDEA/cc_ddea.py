import math
import functools
from typing import List, Tuple, Union
import numpy as np
import torch
import group
from hsjl import HSJL
from utils import toTensor
import de

@torch.no_grad()
def split_pop(
    pop: torch.Tensor,
    group_rules
) -> List[torch.Tensor]:
    ''''
    split the population into sub-population according to the group rules

    Return
        The sub-population
    '''
    subpops = []
    for group_rule in group_rules:
        subpops.append(pop[..., group_rule])
    return subpops


def equip_subpops_with_optimizer(
    subpops: List[torch.Tensor],
    lr_individual: float
) -> Tuple[List[Tuple[torch.Tensor]], List[torch.optim.Optimizer]]:
    '''
    This function will split the subpops into individuals
    '''
    # TODO optimizer add_group api
    optimizers = []
    # optimizer = torch.optim.Adam(lr=lr_individual)
    subpops_grad = []
    for subpop in subpops:
        # optimize the individuals independently
        subpop_grad = subpop.split(1, dim=0)
        for individual in subpop_grad:
            individual.requires_grad_(True)
        optimizer = torch.optim.Adam(subpop_grad, lr=lr_individual)
        subpops_grad.append(subpop_grad)
        optimizers.append(optimizer)
    return subpops_grad, optimizers


def unequip_subpops(subpops_grad: List[Tuple[torch.Tensor]]) -> List[torch.Tensor]:
    subpops = []
    for subpop_grad in subpops_grad:
        subpop = torch.cat(subpop_grad, dim=0).detach().requires_grad_(False)
        subpops.append(subpop)
    return subpops


@torch.no_grad()
def merge_subpops(
    subpops: List[torch.Tensor],
    group_rules,
) -> torch.Tensor:
    d = functools.reduce(lambda result, subpop: result +
                         subpop.shape[-1], subpops, 0)

    pop_new = torch.empty([len(subpops[0]), d],
                          dtype=subpops[0].dtype, device=subpops[0].device)
    for group_id, group_rule in enumerate(group_rules):
        pop_new[..., group_rule] = subpops[group_id]
    return pop_new


def optimize_sub(
    subpops_grad: List[Tuple[torch.Tensor]],
    optimizers: List[torch.optim.Optimizer],
    iter_max: int,
    surrogate: torch.nn.Module,
) -> List[Tuple[torch.Tensor]]:
    # change to evaluate mode
    surrogate.eval()

    for group_id, subpop in enumerate(subpops_grad):
        # for each sub-population
        optimizer = optimizers[group_id]

        for _ in range(iter_max):
            optimizer.zero_grad()

            # optimize the individuals independently
            surrogate.predict(torch.cat(subpop, dim=0),
                              group_id).sum().backward()

            optimizer.step()

    return subpops_grad


@torch.no_grad()
def sort_subpops(
    subpops: List[torch.Tensor],
    surrogate: torch.nn.Module,
) -> List[torch.Tensor]:
    ret = []
    for group_id, subpop in enumerate(subpops):
        y_subpop = surrogate.predict(subpop, group_id).reshape(-1)
        idxes = torch.argsort(y_subpop)
        subpop = subpop[idxes, ...]
        ret.append(subpop)
    return ret

@torch.no_grad()
def ea_operator(
    subpops: List[torch.Tensor],
    group_rules: List[bool],
    surrogate: torch.nn.Module,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    n_explore_ea: int,
    F: float,
    c: float,
):
    # DE operation
    i = -1
    for group_id, subpop in enumerate(subpops):
        i = i + 1
        new_subpop = de.de_rand_1_bin(
            pop=subpop.clone(),
            group_rules=group_rules[i],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            f=functools.partial(surrogate.predict, group_id=group_id),
            max_iter=n_explore_ea,
            F=F,
            c=c,
        )
        subpops[group_id] = torch.cat([subpop, new_subpop], dim=0)
    return subpops

@torch.no_grad()
def produce_offspring(
    # pop: torch.Tensor,
    subpops: List[torch.Tensor],
    surrogate: torch.nn.Module,
    n_top: int,
    n_random_children: int,

) -> torch.Tensor:
    '''
    Args:
        - n_random_children: The number of the children generated by randomly assembling the good sub-individuals
        - n_top: only select among the top `n_top` sub-individuals
    '''
    d = 0
    for subpop in subpops:
        d += subpop.shape[-1]

    
    if len(subpops) > 1:
        subpops = sort_subpops(
            subpops=subpops,
            surrogate=surrogate,
        )
        # 1. merge the sub-population in order
        children1 = merge_subpops(
            subpops=subpops,
            group_rules=surrogate.group_rules
        )
        # 2. merge the sub-population randomly in the top individuals
        children2 = torch.empty([n_random_children, d],
                                dtype=subpops[0].dtype, device=subpops[0].device)
        for group_id, group_rule in enumerate(surrogate.group_rules):
            choices = (torch.rand(n_random_children) * n_top).type(torch.long)
            children2[..., group_rule] = subpops[group_id][choices]
        # 3. merge the children1 and children2 int children
        children = torch.cat([children1, children2], dim=0)
    else:
        children = subpops[0]
    return children


@torch.no_grad()
def update_pop(
    pop: torch.Tensor,
    offsprings: torch.Tensor,
    surrogate: torch.nn.Module,
) -> torch.Tensor:
    pop_size, _ = pop.shape
    pop_new = torch.cat([pop, offsprings], dim=0)
    y_pop_new = surrogate(pop_new).reshape(-1)
    idxes = torch.argsort(y_pop_new)[:pop_size]
    ret = pop_new[idxes, ...]
    return ret


@torch.no_grad()
def find_the_best_individual(
    subpops: List[torch.Tensor],
    surrogate: torch.nn.Module
) -> torch.Tensor:
    pop_ = merge_subpops(
        subpops=subpops,
        group_rules=surrogate.group_rules,
    )
    idx = torch.argmin(surrogate(pop_).reshape(-1))
    return pop_[idx]


def run_groupy(
    pop: torch.Tensor,
    iter_max: int,
    iter_sub_max: int,
    samples: Tuple[torch.Tensor, torch.Tensor],
    lower_bound,
    upper_bound,
    n_group_init: int,
    group_update_gap: int,
    gpu_device: torch.device,
    lr_individual: float,
    surrogate_update_gap: int,
    n_top_rate: float,
    n_random_children_rate: float,
    F: float,
    c: float,
    xl: torch.Tensor,
    xu: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    '''
    Return:
        1. The final population
        2. The best individuals in each generation
    '''
    pop_size, d = pop.shape
    # dataset
    isFloat64 = (samples[0].dtype == torch.float64)
    n_group = n_group_init + 1
    n_group = min(n_group, d)
    i_surrogate = 0
    surrogate = None
    # the best individual in each generation
    best_individuals = []
    # best_ind = []
    save_xs = []
    for gen in range(iter_max):
        # update the n_group
        if gen % group_update_gap == 0 and n_group > 1:
            n_group -= 1
            i_surrogate = 0
        if i_surrogate % surrogate_update_gap == 0 and (surrogate == None or surrogate.n_group() != 1):
            i_surrogate = 0
            # train the surrogate model
            group_rules = group.get_group_rules_randomly(d, n_group)
            group_rules = toTensor(*group_rules)
            if not isinstance(group_rules, List):
                group_rules = [group_rules]
            n_centers = [int(math.sqrt(len(samples[0])))] * len(group_rules)
            if isFloat64:
                surrogate = HSJL(group_rules, n_centers).double().to(gpu_device)
            else:
                surrogate = HSJL(group_rules, n_centers).to(gpu_device)
            # using gpu device
            for i, group_rule in enumerate(group_rules):
                group_rules[i] = group_rule.to(gpu_device)
            # split the dataset again, give each sample a chance to be a training sample
            surrogate.pretrain(samples[0], samples[1])
            # Ban the gradient attribute of surrogate to save GPU memory
            surrogate.set_grad(False)
            
        subpops = ea_operator(
            subpops=split_pop(pop, group_rules),
            group_rules=group_rules,
            surrogate=surrogate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_explore_ea=iter_sub_max,
            F=F,
            c=c
        )

        subpops_grad, optimizers = equip_subpops_with_optimizer(
            subpops=subpops,
            lr_individual=lr_individual
        )

        subpops_grad = optimize_sub(
            subpops_grad=subpops_grad,
            optimizers=optimizers,
            iter_max=iter_sub_max,
            surrogate=surrogate,
        )

        subpops = unequip_subpops(subpops_grad)
        
        # merge the sub-population into the population
        children = produce_offspring(
            # pop=pop,
            subpops=subpops,
            surrogate=surrogate,
            n_top=int(n_top_rate * pop_size),
            n_random_children=int(n_random_children_rate * pop_size),
        )
        # update the population
        pop = update_pop(
            pop=pop,
            offsprings=children,
            surrogate=surrogate,
        )
        pop = torch.clamp(pop, xl, xu)

        best_individual = pop[0]
        # y, _ = task.predict(pop.detach().cpu())
        # best = min(y)
        # # print("best111:", best)
        # best_ind.append(-best)
        save_xs.append(np.array(pop.detach().cpu().numpy()))
        i_surrogate += 1

        # record the best individual
        best_individuals.append(best_individual)

    return pop, best_individuals, save_xs


def run(
    pop: Union[np.ndarray, torch.Tensor],
    iter_max: int,
    iter_sub_max: int,
    samples: Tuple[Union[np.ndarray, torch.Tensor],
                   Union[np.ndarray, torch.Tensor]],
    lower_bound,
    upper_bound,
    n_group_init: int,
    group_update_gap: int,
    gpu_device: torch.device,
    lr_individual: float,
    surrogate_update_gap: int,
    n_top_rate: float,
    n_random_children_rate: float,
    xl: torch.Tensor,
    xu: torch.Tensor,
    F: float = 0.5,
    c: float = 0.5
    
) -> torch.Tensor:
    '''
    enter function
    '''
    # type convertion
    pop = toTensor(pop)
    original_device = pop.device
    # dataset
    x, y = toTensor(*samples)
    # using gpu device
    pop = pop.to(gpu_device)
    x, y = x.to(gpu_device), y.to(gpu_device)

    pop, best_individuals, save_xs = run_groupy(
        pop=pop,
        iter_max=iter_max,
        iter_sub_max=iter_sub_max,
        samples=(x, y),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_group_init=n_group_init,
        group_update_gap=group_update_gap,
        gpu_device=gpu_device,
        lr_individual=lr_individual,
        surrogate_update_gap=surrogate_update_gap,
        n_top_rate=n_top_rate,
        n_random_children_rate=n_random_children_rate,
        F=F,
        c=c,
        xl=xl,
        xu=xu
    )

    return torch.stack(best_individuals, dim=0).to(original_device), save_xs