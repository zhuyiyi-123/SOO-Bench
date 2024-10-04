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
