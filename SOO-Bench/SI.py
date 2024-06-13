import numpy as np
import pickle
import os, sys
from workshop.Benchmark_new.benchmark.Taskdata import OfflineTask

path='/root/workshop/Benchmark_new/con_results_last/CArcoo-gtopx_data-1-0-12000-0-60.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径

f=open(path,'rb')
data=pickle.load(f)


opt_step = 101
curve = data[:opt_step]

r = np.array([i for i in range(opt_step)])
oi_a = curve[0]
S_d = np.trapz(np.ones_like(curve) * oi_a, r)
si_a = np.max(curve)
S_O = np.trapz(np.ones_like(curve) * si_a, r)
S = np.trapz(curve, r)
SI = (S-S_d) / (S_O - S_d)
SI = np.around(SI, 3)
print(SI, S, S_d, S_O)