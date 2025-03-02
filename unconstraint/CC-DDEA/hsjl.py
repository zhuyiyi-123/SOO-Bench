# -*- coding: utf-8 -*-
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
import rbfn_torch


class HSJL(nn.Module):
    def __init__(self, group_rules: List[torch.Tensor], n_centers:List[int], basis_func='gaussian'):
        super(HSJL, self).__init__()
        n = len(group_rules)
        self.group_rules = nn.ParameterList([nn.Parameter(group_rule, requires_grad=False) for group_rule in group_rules])

        self.n_centers = n_centers
        self.basis_func = basis_func
        self.__create_base_model()
        '''
        e^w ~ N(\frac{1}{n}, \sigma), \sigma =\sigma=(\frac{1}{n} - 0) / 3 = \frac{1}{3n}
        '''
        e_weight = torch.Tensor(1, n).normal_(1.0 / n, 1.0 / n / 3)
        self.weight = nn.Parameter(torch.log(e_weight))

        self.bias = nn.Parameter(torch.Tensor(1))
        # the initializion code comes from torch.nn.Linear
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # self.activate_func = torch.nn.LeakyReLU()
    def n_group(self) -> int:
        return len(self.n_centers)

    def __create_base_model(self):
        for i, group_rule in enumerate(self.group_rules):
            d = int(torch.count_nonzero(group_rule).item())

            self.add_module(f'base_model_{i}', rbfn_torch.RBFN(
                d, self.n_centers[i], basis_func=self.basis_func))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = []
        for i, group_rule in enumerate(self.group_rules):
            base_model = getattr(self, f'base_model_{i}')
            ys.append(base_model(x[..., group_rule]))
        ys = torch.cat(ys, dim=-1)
        # activate function
        # ys = self.activate_func(ys)
        return ys @ self.weight.T.exp() + self.bias

    def predict(self, x: torch.Tensor, group_id: int = -1) -> torch.Tensor:
        '''
        Args:
            x: the input vector, some or all dimensions
            group_id: 0 means the 1st base model, -1 means all the base models
        '''
        if group_id == -1:
            return self.__call__(x)
        # TODO use self.named_children instead of getattr
        base_model = getattr(self, f'base_model_{group_id}')
        group_rule = self.group_rules[group_id]
        if x.shape[-1] != torch.count_nonzero(group_rule).item():
            # x contains all the dimension
            return base_model(x[..., group_rule])
            # return self.activate_func(base_model(x[..., group_rule]))
        return base_model(x)
        # return self.activate_func(base_model(x))

    @torch.no_grad()
    def pretrain(self, x: torch.Tensor, y: torch.Tensor):
        device = self.weight.device
        x, y = x.to(device), y.to(device)
        w = []
        for group_id, group_rule in enumerate(self.group_rules):
            base_model = getattr(self, f'base_model_{group_id}')
            x_part = x[..., group_rule]
            base_model.pretrain(x_part, y)
            y_pred = base_model(x_part).reshape(-1)
            # attention
            # y_pred = self.activate_func(y_pred)
            error = torch.mean((y.reshape(-1) - y_pred) ** 2)
            w.append((1. / error).item())

        w = np.log(w / np.sum(w))
        self.weight[:] = torch.from_numpy(w)
        self.bias[:] = torch.tensor(0.)

    def set_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
