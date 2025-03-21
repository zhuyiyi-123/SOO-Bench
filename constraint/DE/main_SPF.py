import random
import numpy as np
import pandas as pds
import time
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RBFN import RBFN
from Latin import latin
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()

from soo_bench.Taskdata import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from par import get_parser

import warnings
warnings.filterwarnings("ignore")

class DE():
    def __init__(self, max_iter, dimension, ub, lb, pop_size, F, CR):
        self.pop_size = pop_size 
        self.chrom_length = dimension  
        self.F = F  
        self.CR = CR  
        self.max_value = ub
        self.min_value = lb
        self.max_iter = max_iter
        self.popfirst = []
        self.fitfirst = []
        self.pop = np.zeros((self.pop_size, self.chrom_length))
        self.init_Population()

    def init_Population(self):
        self.pop = latin(self.pop_size, self.chrom_length, self.min_value, self.max_value)

    def mutation(self, iter):
        F = self.F
        for seq in range(self.pop_size):
            p1, p2 = random.sample(list(self.pop), 2)
            while (self.pop[seq] == p1).all() or (self.pop[seq] == p2).all():
                p1, p2 = random.sample(list(self.pop), 2)
            newp = self.pop[seq] + F * (p2 - p1) 
            newp = np.max(np.vstack((newp, [self.min_value]*self.chrom_length)), 0)
            newp = np.min(np.vstack((newp, [self.max_value]*self.chrom_length)), 0)
            self.pop = np.row_stack((self.pop, newp))

    def crossover(self):
        for seq in range(self.pop_size):
            newp = np.zeros(self.chrom_length)
            jrand = np.random.randint(0,self.chrom_length)
            for i in range(self.chrom_length):
                if random.random() <= self.CR or i == jrand:
                    newp[i] = self.pop[seq+self.pop_size][i]
                else:newp[i] = self.pop[seq][i]
            newp = np.max(np.vstack((newp, [self.min_value]*self.chrom_length)), 0)
            newp = np.min(np.vstack((newp, [self.max_value]*self.chrom_length)), 0)
            self.pop[seq + self.pop_size] = newp  

    def selection(self, fit_value, fit_cons, cons_num, m):
        newpop = np.zeros((self.pop_size, self.chrom_length))
        for i in range(cons_num):
            fit_cons[i] = fit_cons[i]/abs(min(fit_cons[i]))
        fitnessi2 = 0
        fitnessj2 = 0
        for i in range(self.pop_size):
            for j in range(cons_num):
                if fit_cons[j][i] >= 0:
                    fitnessi1 = fit_value[i]
                else:fitnessi1 = False
                fitnessi2 += np.square(min(0, fit_cons[j][i]))
                if fit_cons[j][i + self.pop_size] >= 0:
                    fitnessj1 = fit_value[i + self.pop_size]
                else:fitnessj1 = False
                fitnessj2 += np.square(min(0, fit_cons[j][i + self.pop_size]))

                if fitnessi1 != False and fitnessj1 != False:
                    if fit_value[i] < fit_value[i + self.pop_size]:  # mimimum<, maximum>
                        newpop[i] = self.pop[i]
                    else: newpop[i] = self.pop[i + self.pop_size]
                elif fitnessi1 == False and fitnessj1 != False:
                    newpop[i] = self.pop[i + self.pop_size]
                elif fitnessi1 != False and fitnessj1 == False:
                    newpop[i] = self.pop[i]
                elif fitnessi2 < fitnessj2:  #  mimimum<, maximum>
                    newpop[i] = self.pop[i]
                else: newpop[i] = self.pop[i + self.pop_size]
                
        cons_new = [None] * cons_num
        for i in range(cons_num):
            cons_new[i] = SurrogateObj.predict(de.pop, dataLibrary, m[i])

        popa = []
        for i in range(len(newpop)):
                if self.judge(cons_new, cons_num, i):
                    popa.append(newpop[i])
        if len(popa)>0:
            fitnew = SurrogateObj.predict(popa, dataLibrary, m0)
        else:
            popa = newpop
            fitnew = np.zeros((len(newpop)))
            for j in range(len(newpop)):
                for o in range(cons_num):
                    fitnew[j] = np.square(min(0, cons_new[o][j])) + np.square(min(0, cons_new[o][j]))
        rank = np.argsort(fitnew, axis=0)
        self.fitfirst.append(SurrogateObj.predict(np.array(popa), dataLibrary, m0)[rank[-1]])  #  mimimum0, maximum-1
        self.popfirst.append(popa[rank[-1]])
        firstpop = popa[rank[-1]]
        self.pop = newpop
        # return firstpop, task.predict(np.array([firstpop]))
    def judge(self, v, cons_num, o):
        w = 0
        for i in range(cons_num):
            if v[i][o] < 0:
                w = w + 1
        if w == 0:
            return True
        else: return False

class SurrogateObj():
    def predict(arr, dataLibrary, m):
        '''
        :return: The value of the objective function predicted by the surrogate model
        '''
        return m.predict(arr)

if __name__ == '__main__':
    args = get_parser()
    save_name = 'DESPF' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(current_dir, 'results','constraint', save_name +'.savedata.pkl')):
        print('Already exists:', save_name +'.savedata.pkl')
        exit()

    tt0 = time.time()
    def set_seed(seed):
        import random
        import torch
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(args.seed)
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark,args.seed)
    num = args.num*task.var_num
    if args.sample_method == 'sample_bound':
        task.sample_bound(num, args.low, args.high)
    elif args.sample_method == 'sample_limit':
        task.sample_limit(num, args.low, args.high)
    x = task.x
    dataLibrary = x
    y, cons = task.y.reshape(len(x),), task.cons
    dimension = task.var_num
    cons_num = task.con_num
    Eva_num = 3
    if args.change_optimization_step > 0:
        Eva_num = args.change_optimization_step
        
    m0 = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(len(x))), kernel='gaussian')
    m0.fit(x, y)
    m = [None] * cons_num
    for i in range(cons_num):
        m[i] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(len(x))), kernel='gaussian')
        m[i].fit(x, cons[:, i])

    max_iter = Eva_num
    de = DE(max_iter=max_iter,dimension=dimension,ub=task.xu,lb=task.xl,pop_size=128,F=0.5, CR=0.8)
    de.init_Population()
    EvaList = []
    save_xs = []
    for iter in range(max_iter):
        de.mutation(iter)
        de.crossover()
        fit_value = SurrogateObj.predict(de.pop, dataLibrary, m0)
        fit_cons = [None] * cons_num
        for i in range(cons_num):
            fit_cons[i] = SurrogateObj.predict(de.pop, dataLibrary, m[i])
        de.selection(fit_value, fit_cons, cons_num, m)  
        save_xs.append(np.array(de.pop))

        if Eva_num <= 0:
            break
    tt1 = time.time()
    save_ys = []
    save_cons = []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)

    print('Optimum:', save_xs[-1][0])
    print('True fitness:', save_ys[-1][0], 'Execute time:', tt1 - tt0)
    
    
    def save_data(algorithm_name, args, offline_x, offline_y, offline_cons, save_xs, save_ys, save_cons,other = None, is_constraint=True, level=2):
        save_name = algorithm_name + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(level):
            current_dir = os.path.dirname(current_dir)
        
        if is_constraint:
            os.makedirs(os.path.join(current_dir, 'results','constraint'), exist_ok=True)
            path = os.path.join(current_dir, 'results','constraint', save_name +'.savedata.pkl')
        else:
            os.makedirs(os.path.join(current_dir, 'results','unconstraint'), exist_ok=True)
            path = os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')

        print('saving to: ', path)
        
        offline_x = np.array(offline_x)
        offline_y = np.array(offline_y).flatten()
        if is_constraint:
            offline_cons = np.array(offline_cons)
        else:
            offline_cons = None
        for i in range(len(save_xs)):
            save_xs[i] = np.array(save_xs[i])
            save_ys[i] = np.array(save_ys[i]).flatten()
            if is_constraint:
                save_cons[i] = np.array(save_cons[i])
        if not is_constraint:
            save_cons = None

        save_data = {
            'algorithm_name': algorithm_name,
            'constraint': is_constraint,
            'args': args,
            'offline_x': offline_x,
            'offline_y': offline_y,
            'offline_cons': offline_cons,
            'xs':save_xs,
            'ys':save_ys,
            'cons':save_cons,
            'other': other
        }
        with open (path, 'wb') as f:
            import pickle
            pickle.dump(save_data, f)
    

    save_data('DESPF', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, other=None, is_constraint=True, level=2)
