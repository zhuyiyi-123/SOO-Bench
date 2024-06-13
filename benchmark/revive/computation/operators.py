''''''
"""
    POLIXIR REVIVE, copyright (C) 2021-2022 Polixir Technologies Co., Ltd., is 
    distributed under the GNU Lesser General Public License (GNU LGPL). 
    POLIXIR REVIVE is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
"""
import torch


def cat(*args):    
    return torch.cat(args,axis=-1)

def sum(*args):
    if len(args) == 1:
        return torch.sum(args[0], axis=-1, keepdim=True)
    elif len(args) > 1:
        _res = 0
        for arg in args:
            _res += arg
        return _res
    else:
        raise NotImplementedError
    
def sub(arg1,arg2):
    return arg1 - arg2
    
def mul(*args):
    _res = 1
    if len(args) == 1:
        for index in range(args[0].shape[-1]):
            _res *= args[0][...,index:index+1]
        return _res
    elif len(args) > 1:
        for arg in args:
            _res *= arg
        return _res
    else:
        raise NotImplementedError
        
def div(arg1, arg2):
    return arg1 / (arg2 + 1e-8) 
    
def mean(*args):
    if len(args) == 1:
        return torch.mean(args[0], dim=-1, keepdim=True)
    return torch.mean(torch.cat([arg.unsqueeze(-1) for arg in args], axis=-1), axis=-1)
    
def min(*args):
    if len(args) == 1:
        return torch.min(args[0], dim=-1, keepdim=True)[0]
    
    return torch.min(torch.cat([arg.unsqueeze(-1) for arg in args], axis=-1), axis=-1)[0]

def max(*args):
    if len(args) == 1:
        return torch.max(args[0], dim=-1, keepdim=True)[0]
    
    return torch.max(torch.cat([arg.unsqueeze(-1) for arg in args], axis=-1), axis=-1)[0]

def abs(arg):    
    return torch.abs(arg)

def clip(arg,min_v=None,max_v=None):
    return torch.clip(arg,min_v,max_v)

def exp(arg):
    return torch.exp(arg)

def log(arg):
    return torch.log(arg)
    
"""
def log(arg):
    assert len(args) == 1
    
    return torch.log(arg)



def floor(arg, base):
    pass
    
def fmod(arg):
    pass
"""

__all__ = [
    "sum",
    "cat",
    "sub",
    "mul",
    "div",
    "min",
    "mean",
    "max",
    "abs",
    "clip",
    "exp",
    "log",
]