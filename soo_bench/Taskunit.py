import numpy.random
import os
from ctypes import *
import random
import numpy as np
import torch
import pickle
from soo_bench.protein.tf_bind_8_oracle import TFBind8Oracle
from soo_bench.protein.tf_bind_8_dataset import TFBind8Dataset
from soo_bench.protein.tf_bind_10_oracle import TFBind10Oracle
from soo_bench.protein.tf_bind_10_dataset import TFBind10Dataset
try:
    from soo_bench.hybrid_oracle import SearchParams, ISearchParams
except ImportError:
    print('warnning: hybrid_data is not availible')

class TaskUnit:
    TASK_NAME = ''
    def __init__(self, benchmark, seed = None, use_cache=True, cache_path=None,*args, **kwargs):
        self.task = self.TASK_NAME
        self.benchmark = benchmark
        self.seed = seed
        self.use_cache = use_cache
        self.cache_path = cache_path
        
        
        self.init(benchmark, seed, use_cache, *args, **kwargs)
        if not hasattr(self, 'seed'):
            raise Exception("self.seed is not set yet. it must be set before calling self.init()")
        if not hasattr(self, 'obj_num'):
            raise Exception("self.obj_num is not set yet. it must be set before calling self.init()")
        if not hasattr(self, 'var_num'):
            raise Exception("self.var_num is not set yet. it must be set before calling self.init()")
        if not hasattr(self, 'con_num'):
            raise Exception("self.con_num is not set yet. it must be set before calling self.init()")
        if not hasattr(self, 'xl'):
            raise Exception("self.xl is not set yet. it must be set before calling self.init()")
        if not hasattr(self, 'xu'):
            raise Exception("self.xu is not set yet. it must be set before calling self.init()")
        
        self.x_values = []
        self.f = [0.0] * self.obj_num
        self.g = [0.0] * self.con_num
        self.x = None
        self.y = None
    
    def init(self,benchmark, seed, use_cache, *args, **kwargs):
        '''This function must be called at the end of __init__()'''
        raise NotImplementedError


    
    def is_constraint(self):
        if self.con_num == 0:
            return False
        else:
            return True


    def predict(self, x):
        raise NotImplementedError


    
    def filter_useful(self, dataset, cons = None):
        if self.is_constraint() == False:
            return dataset, []
        
        if cons is None:
            y, cons = self.predict(dataset)
        useful = []
        useless = []
        for i in range(len(dataset)):
            if np.all(cons[i] >= 0):
                useful.append(dataset[i])
            else:
                useless.append(dataset[i])
        return useful, useless
        
        
    def sample_x_ignore_constraints(self, num=2):
        self.x = []
        self.num = num
        for _ in range(num):
            for lower, upper in zip(self.xl, self.xu):
                x = random.uniform(lower, upper)
                self.x_values.append(x)
            self.x.append(self.x_values)
            self.x_values = []
        return self.x

    def sample_x_using_cache(self,  num=2 , rate_satisfying_constraints=0.4, maxtry=10000000, seek_size=512, print_info=True):
        cache_name = f'{self.task}_b{self.benchmark}_n{num}_r{round(rate_satisfying_constraints,4)}_seed{self.seed}.cache.pkl'
        path = os.path.join(self.cache_path, cache_name)
        if self.use_cache and os.path.exists(path):
            print(f'using cache:{path}')
            with open(path, 'rb') as f:
                self.x, self.y, self.cons = pickle.load(f)
            return self.x
        
        result = self.sample_x(num,rate_satisfying_constraints,maxtry,seek_size,print_info)
        
        if self.use_cache and not os.path.exists(path):
            with open(path,'wb') as f:
                pickle.dump((self.x, self.y, self.cons),f)
        
        return result
        
    def sample_x(self, num=2 , rate_satisfying_constraints=0.4, maxtry=10000000, seek_size=512, print_info=True):
        assert(0 <= rate_satisfying_constraints <= 1)
        assert(maxtry >= 0)
        
        if self.seed is not None:
            random.seed(self.seed)
            numpy.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        
        if self.is_constraint() == False: 
            x = self.sample_x_ignore_constraints(num)
            x = np.array(x)
            x = x.reshape(x.shape[0], -1)
            self.x = x
            if self.is_constraint():
                self.y, self.cons = self.predict(self.x)
            else:
                self.y, self.cons = self.predict(self.x)
                self.cons = np.zeros_like(self.y)
            return x
        else: 
            
            final_useful = []
            final_useless = []
            expected_n = int(num * rate_satisfying_constraints)
        
            
            for turn_id in range(maxtry + 1):
                if turn_id == maxtry:
                    raise Exception('Failed when generating constraint dataset. Max try excessed.')
                

                x = self.sample_x_ignore_constraints(seek_size)
                x = np.array(x)
                x = x.reshape(x.shape[0], -1)
                self.x = x
                useful, useless = self.filter_useful(x)
                

                final_useful += useful
                final_useless += useless[: max(0, num - len(final_useless))]
                if print_info:
                    print('now generating constraint dataset: ', len(final_useful), '/', expected_n, 'turn_id: ', turn_id, '/', maxtry, end='\r')
                
                if len(final_useful) >= expected_n:
                    
                    break
                
            if print_info:
                print()
            final_useful = final_useful[:expected_n]
            final_dataset = final_useful + final_useless[:num - len(final_useful)]
            self.detail_useful_data_num = len(final_useful)
            
            self.x = np.array(final_dataset)
            
            # sample_y
            if self.is_constraint():
                self.y, self.cons = self.predict(self.x)
            else:
                self.y, self.cons = self.predict(self.x)
                self.cons = np.zeros_like(self.y)
            return self.x
            
        
        
    def sample_y(self):
        return self.y

    def sample_bound(self, num=0, low=0, high=100, rate_satisfying_constraints=0.4, *args, **kwargs):
        if num == 0:
            dim = self.var_num
            num = dim * 1000
        self.sample_x_using_cache(num,rate_satisfying_constraints, *args,**kwargs)
        self.sample_y()
        lower_bound = np.percentile(self.y, low)
        upper_bound = np.percentile(self.y, high)
        filtered_indices = (self.y <= upper_bound) & (self.y >= lower_bound)
        filtered_indices = filtered_indices.reshape(-1)
        self.x = np.array(self.x).astype(np.float64)
        self.y = np.array(self.y).astype(np.float64)
        self.x = self.x[filtered_indices]
        self.y = self.y[filtered_indices]
        if self.is_constraint():
            self.cons = self.cons[filtered_indices]
        self.x = self.x.astype(np.float64)
        self.y = self.y.astype(np.float64).reshape(len(self.x), 1)
    
    def read(self, x, y):
        self.x = x
        self.y = y
        

    def dataset(self, num):
        return self.x(num), self.y()

    def normalize_x(self, x):
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        z_score_x = (x - mean_x) / std_x
        return z_score_x

    def map_normalize_x(self):
        self._mean_x = np.mean(self.x, axis=0)
        self._std_x = np.mean(self.x, axis=0)
        self.x = (self.x - self._mean_x) / self._std_x

    def normalize_y(self, y):
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
        z_score_y = (y - mean_y) / std_y
        return z_score_y

    def map_normalize_y(self):
        self._mean_y = np.mean(self.y, axis=0)
        self._std_y = np.mean(self.y, axis=0)
        self.y = (self.y - self._mean_y) / self._std_y

    def normalize_cons(self, cons):
        mean_cons = np.mean(cons, axis=0)
        std_cons = np.std(cons, axis=0)
        z_score_cons = (cons - mean_cons) / std_cons
        return z_score_cons

    def map_normalize_cons(self):
        self._mean_cons = np.mean(self.cons, axis=0)
        self._std_cons = np.std(self.cons, axis=0)
        z_score_cons = (self.cons - self._mean_cons) / self._std_cons
        return z_score_cons

    def denormalize_x(self, x):
        origin_x = x * self._std_x + self._mean_x
        return origin_x

    def denormalize_y(self, y):
        origin_y = y * self._std_y + self._mean_y
        return origin_y

    def denormalize_cons(self, cons):
        origin_cons = cons * self._std_cons + self._mean_cons
        return origin_cons

    def ood_x(self, num=10, rate=0.6):
        ood_data = []
        gap = np.random.uniform(low=0, high=0, size=len(self.xl))
        for j in range(len(self.xl)):
            gap[j] = self.xu[j] - self.xl[j]
        for _ in range(num):
            x = np.random.uniform(self.xl, self.xu)
            i = -1
            while i < len(x) - 1:
                i = i + 1
                while x[i] > gap[i] * rate:
                    x = np.random.uniform(self.xl, self.xu)
                    i = 0
            ood_data.append(x)
        return ood_data

    def __str__(self):
        return f'TaskUnit(task={self.TASK_NAME},benchmark={self.benchmark}, seed={self.seed})'
    

class Task_GTOPX(TaskUnit):
    TASK_NAME = 'gtopx_data'
    def init(self, benchmark, seed=None, *args, **kwargs):

        if benchmark == 1:
            """"benchmark name is cassini1"""
            self.obj_num = 1
            self.var_num = 6
            self.con_num = 4
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0]  # lower bounds
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0]  # upper bounds
        elif benchmark == 2:
            """"benchmark name is cassini2"""
            self.obj_num = 1
            self.var_num = 22
            self.con_num = 0
            self.xl = [-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05,
                       1.05, 1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi]
            self.xu = [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5,
                       291.0, np.pi, np.pi, np.pi, np.pi]
        elif benchmark == 3:
            """"benchmark name is Messenger(reduced)"""
            self.obj_num = 1
            self.var_num = 18
            self.con_num = 0
            self.xl = [1000.0, 1.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.1, -np.pi,
                       -np.pi, -np.pi]
            self.xu = [4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99, 0.99, 0.99, 6.0, 6.0, 6.0, np.pi,
                       np.pi, np.pi]
        elif benchmark == 4:
            """"benchmark name is Messenger(full)"""
            self.obj_num = 1
            self.var_num = 26
            self.con_num = 0
            self.xl = [1900.0, 2.5, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                       0.01,
                       1.1, 1.1, 1.05, 1.05, 1.05, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
            self.xu = [2300.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 600.0, 0.99, 0.99, 0.99, 0.99, 0.99,
                       0.99,
                       6.0, 6.0, 6.0, 6.0, 6.0, np.pi, np.pi, np.pi, np.pi, np.pi]
        elif benchmark == 5:
            """"benchmark name is GTOC1"""
            self.obj_num = 1
            self.var_num = 8
            self.con_num = 6
            self.xl = [3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0]
            self.xu = [10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0]
        elif benchmark == 6:
            """"benchmark name is Rosetta"""
            self.obj_num = 1
            self.var_num = 22
            self.con_num = 0
            self.xl = [1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.06,
                       1.05, 1.05, 1.05, -np.pi, -np.pi, -np.pi, -np.pi]
            self.xu = [1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9, 0.9, 0.9, 0.9, 0.9, 9.0, 9.0,
                       9.0, 9.0, np.pi, np.pi, np.pi, np.pi]
        elif benchmark == 7:
            """"benchmark name is Cassini1-MINLP"""
            self.obj_num = 1
            self.var_num = 10
            self.con_num = 4
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0, 1.0, 1.0, 1.0, 1.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0, 9.0, 9.0, 9.0, 9.0]
        else:
            raise Exception(f"benchmark {benchmark} is not found in task {self.__class__.__name__}")
        
    
    def predict(self, x):
        if os.name == "posix":
            lib_name = "gtopx.so"  # Linux//Mac/Cygwin
        else:
            lib_name = "gtopx.dll"  # Windows

        benchmark = self.benchmark
        if benchmark == 7:
            benchmark = 8
        lib_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib_name
        CLIB = CDLL(lib_path)
        f_ = (c_double * self.obj_num)()
        benchmark_ = c_long(benchmark)
        value = []
        constraint = []

        for j in range(0, len(x)):
            x_ = (c_double * self.var_num)()
            for i in range(0, self.var_num):
                x_[i] = c_double(x[j][i])
            if self.con_num > 0:
                g_ = (c_double * self.con_num)()
            if self.con_num == 0:
                g_ = (c_double * 1)()
            CLIB.gtopx(benchmark_, f_, g_, x_)
            for i in range(0, self.obj_num):
                self.f[i] = f_[i]
            for i in range(0, self.con_num):
                self.g[i] = g_[i]
            value.append(self.f[0])
            constraint.append(self.g[:])
        value = -np.array(value)
        value[np.isnan(value)] = np.max(self.y)

        return value, np.array(constraint)
    
class Task_CEC(TaskUnit):
    TASK_NAME = 'cec_data'
    def init(self, benchmark, seed=None, *args, **kwargs):
        # Optimal operation of alkylation unit
        if benchmark == 1:
            self.obj_num = 1
            self.var_num = 7
            self.con_num = 14
            self.xl = [1000.0, 0.0, 2000.0, 0.0, 0.0, 0.0, 0.0]
            self.xu = [2000.0, 100.0, 4000.0, 100.0, 100.0, 20.0, 200.0]
        # Process flow sheeting problem
        elif benchmark == 2:
            self.obj_num = 1
            self.var_num = 3
            self.con_num = 3 
            self.xl = [0.2, -2.22554, 0.0]
            self.xu = [1, -1, 1]
        # Process synthesis problem
        elif benchmark == 3:
            self.obj_num = 1
            self.var_num = 2
            self.con_num = 2
            self.xl = [0.0, 0.0]
            self.xu = [1.6, 1.0]
        # Three-bar truss design problem
        elif benchmark == 4:
            self.obj_num = 1
            self.var_num = 2
            self.con_num = 3
            self.xl = [0.0, 0.0]
            self.xu = [1.0, 1.0]
        # Welded beam design
        elif benchmark == 5:
            self.obj_num = 1
            self.var_num = 4
            self.con_num = 5
            self.xl = [0.125, 0.1, 0.1, 0.1]
            self.xu = [2, 10, 10, 2]
        else:
            raise Exception(f"benchmark {benchmark} is not found in task {self.__class__.__name__}")
    
    def predict(self, x):
        if self.benchmark == 1:
            f = 1.715 * x[:, 0] + 0.035 * x[:, 0] * x[:, 5] + 4.0565 * x[:, 2] + 10.0 * x[:, 1] - 0.063 * x[:, 2] * x[:,
                                                                                                                    4]
            g = np.zeros((len(x), 14))
            g[:, 0] = -(0.0059553571 * x[:, 5] * x[:, 5] * x[:, 0] + 0.88392857 * x[:, 2] - 0.1175625 * x[:, 5] * x[:, 0]
                       - x[:, 0])
            g[:, 1] = -(1.1088 * x[:, 0] + 0.1303533 * x[:, 0] * x[:, 5] - 0.0066033 * x[:, 0] * x[:, 5] * x[:, 5] - x[:,
                                                                                                                   2])
            g[:, 2] = -(6.66173269 * x[:, 5] * x[:, 5] + 172.39878 * x[:, 4] - 56.596669 * x[:, 3] - 191.20592 * x[:, 5]
                       - 10000)
            g[:, 3] = -(1.08702 * x[:, 5] + 0.32175 * x[:, 3] - 0.03762 * x[:, 5] * x[:, 5] - x[:, 4] + 56.85075)
            g[:, 4] = -(0.006198 * x[:, 6] * x[:, 3] * x[:, 2] + 2462.3121 * x[:, 1] - 25.125634 * x[:, 1] * x[:, 3]
                       - x[:, 2] * x[:, 3])
            g[:, 5] = -(161.18996 * x[:, 2] * x[:, 3] + 5000.0 * x[:, 1] * x[:, 3] - 489510.0 * x[:, 1] - x[:, 2] * x[:,
                                                                                                                   3]
                       * x[:, 6])
            g[:, 6] = -(0.33 * x[:, 6] - x[:, 4] + 44.333333)
            g[:, 7] = -(0.022556 * x[:, 4] - 0.007595 * x[:, 6] - 1.0)
            g[:, 8] = -(0.00061 * x[:, 2] - 0.0005 * x[:, 0] - 1.0)
            g[:, 9] = -(0.819672 * x[:, 0] - x[:, 2] + 0.819672)
            g[:, 10] = -(24500.0 * x[:, 1] - 250.0 * x[:, 1] * x[:, 3] - x[:, 2] * x[:, 3])
            g[:, 11] = -(1020.4082 * x[:, 3] * x[:, 1] + 1.2244898 * x[:, 2] * x[:, 3] - 100000. * x[:, 1])
            g[:, 12] = -(6.25 * x[:, 0] * x[:, 5] + 6.25 * x[:, 0] - 7.625 * x[:, 2] - 100000)
            g[:, 13] = -(1.22 * x[:, 2] - x[:, 5] * x[:, 0] - x[:, 0] + 1.0)
        elif self.benchmark == 2:
            x[:, 2] = np.round(x[:, 2])
            g = np.zeros((len(x), 3))
            f = -0.7 * x[:, 2] + 5 * (x[:, 0] - 0.5) ** 2 + 0.8
            g[:, 0] = -(-np.exp(x[:, 0] - 0.2) - x[:, 1])
            g[:, 1] = -(x[:, 1] + 1.1 * x[:, 2] + 1)
            g[:, 2] = -(x[:, 0] - x[:, 2] - 0.2)
        elif self.benchmark == 3:
            x = np.array(x)
            x[:, 1] = np.round(x[:, 1])
            f = x[:, 1] + 2 * x[:, 0]
            g = np.zeros((len(x), 2))
            g[:, 0] = -(-x[:, 0] ** 2 - x[:, 1] + 1.25)
            g[:, 1] = -(x[:, 0] + x[:, 1] - 1.6)
        elif self.benchmark == 4:
            f = (2 * np.sqrt(2) * x[:, 0] + x[:, 1]) * 100
            g = np.zeros((len(x), 3))
            g[:, 0] = -(x[:, 1] / (np.sqrt(2) * x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1]) * 2 - 2)
            g[:, 1] = -((np.sqrt(2) * x[:, 0] + x[:, 1]) / (np.sqrt(2) * x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1]) * 2 - 2)
            g[:, 2] = -(1 / (np.sqrt(2) * x[:, 1] + x[:, 0]) * 2 - 2)
        elif self.benchmark == 5:
            f = 1.10471 * x[:, 0] ** 2 * x[:, 1] + 0.04811 * x[:, 2] * x[:, 3] * (14 + x[:, 1])
            P = 6000
            L = 14
            delta_max = 0.25
            E = 30 * 1e6
            G = 12 * 1e6
            T_max = 13600
            sigma_max = 30000
            Pc = 4.013 * E * np.sqrt(x[:, 2] ** 2 * x[:, 3] ** 6 / 30) / L ** 2 * (
                        1 - x[:, 2] / (2 * L) * np.sqrt(E / (4 * G)))
            sigma = 6 * P * L / (x[:, 3] * x[:, 2] ** 2)
            delta = 6 * P * L ** 3 / (E * x[:, 2] ** 2 * x[:, 3])
            J = 2 * (np.sqrt(2) * x[:, 0] * x[:, 1] * (x[:, 1] ** 2 / 4 + (x[:, 0] + x[:, 2]) ** 2 / 4))
            R = np.sqrt(x[:, 1] ** 2 / 4 + (x[:, 0] + x[:, 2]) ** 2 / 4)
            M = P * (L + x[:, 1] / 2)
            ttt = M * R / J
            tt = P / (np.sqrt(2) * x[:, 0] * x[:, 1])
            t = np.sqrt(tt ** 2 + 2 * tt * ttt * x[:, 1] / (2 * R) + ttt ** 2)
            # constraints
            g = np.zeros((x.shape[0], 5))
            g[:, 0] = -(x[:, 0] - x[:, 3])
            g[:, 1] = -(sigma - sigma_max)
            g[:, 2] = -(P - Pc)
            g[:, 3] = -(t - T_max)
            g[:, 4] = -(delta - delta_max)
        return -f, g

class Task_hybrid(TaskUnit):
    TASK_NAME = 'hybrid_data'
    def init(self, benchmark, seed, *args, **kwargs):
        if benchmark == 0:
            self.con_num = 2
        elif benchmark == 1:
            self.con_num = 0
        else:
            raise Exception(f"benchmark {benchmark} is not found in task {self.__class__.__name__}")
        self.obj_num = 1
        self.var_num = 115
        self.xl = []
        self.xu = []
        for i in range(115):
            if i < 7:
                xl = 0
                xu = 100
            elif i < 9:
                xl = 0.2
                xu = 1.5
            elif i < 62:
                xl = 1000
                xu = 3000
            elif i < 115:
                xl = 70
                xu = 150
            self.xl.append(xl)
            self.xu.append(xu)
    
    def predict(self, x):
        if self.benchmark == 1:
            searcher = ISearchParams()
            results, tra_info, results_info = searcher.search(x)
            constraints = 0
        elif self.benchmark == 0:
            searcher = ISearchParams()
            results, constraints, tra_info, results_info = searcher.csearch(x)
            constraints = - np.array(constraints)
        results = -np.array(results)
        return results, constraints
    
class Task_mujoco(TaskUnit):
    TASK_NAME = 'mujoco_data'
    def init(self, benchmark, seed, *args, **kwargs):
        if benchmark == 1:
            from soo_bench.protein.tf_bind_8_oracle import TFBind8Oracle
            from soo_bench.protein.tf_bind_8_dataset import TFBind8Dataset
            
            self.obj_num = 1
            self.var_num = 8
            self.con_num = 0
            self.xl = [0, 0, 0, 0, 0, 0, 0, 0]
            self.xu = [3, 3, 3, 3, 3, 3, 3, 3]
            placeholder_dataset = TFBind8Dataset()
            self.oracle = TFBind8Oracle(placeholder_dataset)
        elif benchmark == 2:
            from soo_bench.protein.tf_bind_10_oracle import TFBind10Oracle
            from soo_bench.protein.tf_bind_10_dataset import TFBind10Dataset
            
            self.obj_num = 1
            self.var_num = 10
            self.con_num = 0
            self.xl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.xu = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            placeholder_dataset = TFBind10Dataset()
            self.oracle = TFBind10Oracle(placeholder_dataset)
    
    def predict(self, x):
        design = x
        if self.benchmark == 1:
            value = self.oracle.predict(design)
        elif self.benchmark == 2:
            value = self.oracle.predict(design)
        constraint = 0
        return value, constraint
    
    def sample_x_ignore_constraints(self, num=2):
        '''
        sample x values from the dataset without considering constraints
        '''
        self.x = []
        self.num = num
        for _ in range(num):
            for lower, upper in zip(self.xl, self.xu):
                x = random.uniform(lower, upper)
                self.x_values.append(x)
            self.x_values = [round(x) for x in self.x_values] # the only different line to the base class TaskUnit, 
            self.x.append(self.x_values)
            self.x_values = []
        self.x = np.array(self.x)
        self.x = self.x.reshape(self.x.shape[0], -1)
        return self.x
    
