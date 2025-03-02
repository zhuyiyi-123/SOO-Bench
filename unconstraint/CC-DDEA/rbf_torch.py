import functools
import math
from typing import Callable, Union
import numpy as np
import torch
import torch.nn as nn
import rbf


class RBF(nn.Module):
    '''radial basis function layer(RBF in short)
    A RBF is defined by the following elements:

        1. A norm function `norm_func`
        2. A basis function `basis_func`
        3. A positive shape parameter epsilon
        4. The number of kernels N, and their relative centers c_i

    For more information, see:

        1. https://en.wikipedia.org/wiki/Radial_basis_function
        2. https://en.wikipedia.org/wiki/Radial_basis_function_network
        3. https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
        4. https://github.com/rssalessio/PytorchRBFLayer
    '''

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 norm_func: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
                     torch.linalg.norm, ord=2, axis=-1),
                 basis_func: Union[Callable[[torch.Tensor],
                                            torch.Tensor], str] = 'gaussian',
                 ):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if callable(norm_func):
            self.norm_func = norm_func
        else:
            raise ValueError('RBF layer: norm_func must be a callable object')
        if callable(basis_func):
            self.basis_func = basis_func
        elif isinstance(basis_func, str):
            if basis_func in rbf.funcs:
                self.basis_func = rbf.funcs[basis_func]
            else:
                raise ValueError(
                    f'RBF layer: No builtin basis function named {basis_func}')
        else:
            raise ValueError('RBF layer: error type of basis_func')

        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        # the Logarithm of shape parameter, using e^{log_sigmas} to ensure that shape parameter is positive
        self.log_sigmas = nn.Parameter(torch.Tensor(self.out_features))

        self.init_parameters()

    def init_parameters(self):
        '''initialize the shape parameter and centers parameters
        '''
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.unsqueeze(1)
        r = self.norm_func(x - self.centers)
        eps_r = self.log_sigmas.exp() * r
        return self.basis_func(eps_r)
