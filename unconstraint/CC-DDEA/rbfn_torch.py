import functools
from typing import Callable, Union
import numpy as np
import torch
from torch import nn
from rbf_torch import RBF
import rbfn_np


class RBFN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 norm_func: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
                     torch.linalg.norm, ord=2, axis=-1),
                 basis_func: Union[Callable[[torch.Tensor],
                                            torch.Tensor], str] = 'gaussian',
                 ):
        super(RBFN, self).__init__()
        self.hidden_features = hidden_features
        self.norm_func = norm_func
        self.basis_func = basis_func
        self.rbf = RBF(
            in_features,
            hidden_features,
            norm_func,
            basis_func,
        )
        self.linear = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rbf(x)
        return self.linear(x)

    @torch.no_grad()
    def pretrain(self, x: torch.Tensor, y: torch.Tensor):
        # if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        # if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
        rbfn = rbfn_np.RBFN(self.hidden_features, basis_func=self.basis_func)
        rbfn.fit(x, y)
        
        self.rbf.centers[:] = torch.tensor(rbfn.centers)
        self.rbf.log_sigmas[:] = torch.log(torch.tensor(rbfn.sigmas))
        self.linear.weight[:] = torch.tensor(rbfn.weight).T
        self.linear.bias[:] = torch.tensor(rbfn.bias)
        # return rbfn

    def set_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
