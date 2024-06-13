import numpy as np
import os, sys
from collections import Counter
# from main import coms_cleaned
import torch
from workshop.Benchmark_new.benchmark.Taskdata import OfflineTask

def sample_dis(task, benchmark, num, low, high):
    x = np.array(task.sample_x(num=num))
    y = np.array(task.sample_y()[0])

    lower_bound = np.percentile(y, low)  
    upper_bound = np.percentile(y, high) 
    print(lower_bound, upper_bound)

    filtered_indices = (y <= upper_bound) & (y >= lower_bound)
    x = x[filtered_indices]
    y = y[filtered_indices]
    
    x = torch.tensor(np.array(x).astype(np.float32).reshape(len(x), len(x[0])))
    y = torch.tensor(-np.array(y).astype(np.float32).reshape(len(x),1))
    return x, y


