import numpy as np
import pickle
import os, sys

import csv
import pandas as pd


import pickle
def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res

def getInfo(path,dataset, benchmark,seed,num):
    f=open(path,'rb')
    data=pickle.load(f)

    is_pointlist = (len(np.shape(data[0])) >=1)
    if is_pointlist:
        print(f'[INFO] Data in "{path}" is a pointlist. Resolving...')
        from soo_bench.Taskdata import OfflineTask
        task = OfflineTask(dataset,benchmark, seed)
        data = task.predict(data)[0]
        data = data.reshape(np.size(data))
        offlineBest = np.max(task.y)
        data = list(data)
        data = [offlineBest] + data
        
    else:
        tmp = []
        for item in data:
            try:
                tmp.append(item[0])
            except:
                tmp.append(item)
        data = np.array(tmp)
        
    
    # calculate SI
    opt_step = len(data)
    curve = data[:opt_step]
    curve = [item[0] if isinstance(item, list) else item for item in curve]

    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    SI = (S-S_d) / (S_O - S_d)
    SI = np.around(SI, 3)
    
    num = len(data)
    if not is_pointlist:
        offlineBest = data[0]
    algorithmBest = data[-1]
    return num, offlineBest, algorithmBest, SI



def main(rootpath, save_to):
    savedata = [] 
    for filename in os.listdir(rootpath):
        algorithm, dataset, benchmark, seed, num, low, high = filename.split('-')
        high = high.rstrip('.pkl')
        benchmark = int(benchmark)
        seed = int(seed)
        num = int(num)
        low = int(low)
        high = int(high)
        item = {
            'algorithm': algorithm,
            'dataset': dataset,
            'benchmark':benchmark,
            'seed':seed,
            'num':num,
            'low':low,
            'high':high,
            'offlineBest': '?',
            'algorithmBest': '?',
            'SI': '',
            }
        try:
            num, offlineBest, algorithmBest, SI = getInfo(rootpath + '/' + filename, dataset, benchmark,seed,num)
            item['num'] = num
            item['offlineBest'] = offlineBest
            item['algorithmBest'] = algorithmBest
            item['SI'] = SI
        except Exception as exc:
            print(f'[WARNNING] skipping file {filename} with following Excetion:')
            print(f'{exc.__class__.__name__}: {exc}')
            pass

        savedata.append(item)
    
    if len(savedata) == 0:
        print('[WARNNING] result not found at {}'.format(rootpath))
        return

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=savedata[0].keys())
        writer.writeheader()
        for row in savedata:
            writer.writerow(row)
    print(f'[DONE] result saved to {save_to}')

rootpath = os.path.abspath(os.path.join(__file__, '..', '..', 'results','unconstraint'))
save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary', 'results_unconstraint.csv'))
main(rootpath, save_to)

rootpath = os.path.abspath(os.path.join(__file__, '..', '..', 'results','constraint'))
save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary', 'results_constraint.csv'))
main(rootpath,save_to)
