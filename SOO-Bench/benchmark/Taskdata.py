import numpy.random

from Task import make
import os
from ctypes import *
import random
import multiprocessing
import numpy as np
import torch
from tf_bind_8_oracle import TFBind8Oracle
from tf_bind_8_dataset import TFBind8Dataset
from tf_bind_10_oracle import TFBind10Oracle
from tf_bind_10_dataset import TFBind10Dataset
from hybrid_oracle import SearchParams, ISearchParams


class OfflineTask:
    def __init__(self, task, benchmark, seed = 1):
        self.task = task
        self.benchmark = benchmark
        self.seed = seed
        self.x_values = []
        self.wrapped_task = make(self.task, self.benchmark)
        self.obj_num = self.wrapped_task.dataset[0]
        self.var_num = self.wrapped_task.dataset[1]
        self.con_num = self.wrapped_task.dataset[2]
        self.xl = self.wrapped_task.dataset[3]
        self.xu = self.wrapped_task.dataset[4]
        self.f = [0.0] * self.obj_num
        self.g = [0.0] * self.con_num
        self.x = None
        self.y = None

    def is_constraint(self):
        if self.con_num == 0:
            return False
        else:
            return True

    def single_evaluate(self, design):
        if self.benchmark == 1:
            placeholder_dataset = TFBind8Dataset()
            oracle = TFBind8Oracle(placeholder_dataset)
            return oracle.predict(design)
        elif self.benchmark == 2:
            placeholder_dataset = TFBind10Dataset()
            oracle = TFBind10Oracle(placeholder_dataset)
            return oracle.predict(design)

    def predict(self, x):
        x = np.array(x)
        if self.task == "gtopx_data":
            value, constraint = self.prediction_gtopx(x)
        elif self.task == "cec_data":
            value, constraint = self.prediction_cec(x)
        elif self.task == "hybrid_data":
            value, constraint = self.prediction_hybrid(x)
        elif self.task == "mujoco_data":
            value, constraint = self.prediction_mujoco(x)
        
        value = np.array(value)
        constraint = np.array(constraint)
        return value, constraint

    def prediction_gtopx(self, x):
        if os.name == "posix":
            lib_name = "gtopx.so"  # Linux//Mac/Cygwin
        else:
            lib_name = "gtopx.dll"  # Windows
        lib_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib_name
        CLIB = CDLL(lib_path)
        f_ = (c_double * self.obj_num)()
        benchmark_ = c_long(self.benchmark)
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

    def prediction_cec(self, x):
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

    def prediction_hybrid(self, x):
        # if self.benchmark == 1:
        #     searcher = SearchParams()
        #     results, tra_info, results_info = searcher.search(**x)
        #     constraints = 0
        # elif self.benchmark == 0:
        #     searcher = SearchParams()
        #     results, constraints, tra_info, results_info = searcher.csearch(**x)
        # return results, constraints
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

    def prediction_mujoco(self, x):
        value = self.single_evaluate(x)
        constraint = 0
        return value, constraint

    
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
        
        
    def _sample_x_ignore_constraints(self, num=2):
        # random.seed(0)
        # numpy.random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        
        
        self.x = []
        self.num = num
        # uniform_dist = torch.distributions.Uniform(torch.tensor(self.wrapped_task.dataset[3]), torch.tensor(self.wrapped_task.dataset[4]))
        # print(uniform_dist.sample((num, )))
        for _ in range(num):
            for lower, upper in zip(self.wrapped_task.dataset[3], self.wrapped_task.dataset[4]):
                x = random.uniform(lower, upper)
                self.x_values.append(x)
                # if self.task == "mujoco_data":
                #     self.xx.append(round(self.x_values))
                #     self.x_values = []
                # else:
            if self.task == "mujoco_data":
                self.x_values = [round(x) for x in self.x_values]
            self.x.append(self.x_values)
            self.x_values = []
        self.x = np.array(self.x)
        self.x = self.x.reshape(self.x.shape[0], -1)
        return self.x

    def sample_x(self, num=2, / , rate_satisfying_constraints=0.4, maxtry=10000000):
        assert(0 <= rate_satisfying_constraints <= 1)
        assert(maxtry >= 0)
        
        if self.seed is not None:
            random.seed(self.seed)
            numpy.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        
        seek_size = 512
        seek_rate = 0
        notfound = True
        
        if self.is_constraint() == False: 
            x = self._sample_x_ignore_constraints(num)
            return x
        else: 
            
            final_useful = []
            final_useless = []
            expected_n = int(num * rate_satisfying_constraints)
            
            # seek_size = max( num, seek_size)
            
            for turn_id in range(maxtry + 1):
                if turn_id == maxtry:
                    raise Exception('Failed when generating constraint dataset. Max try excessed.')
                

                x = self._sample_x_ignore_constraints(seek_size)
                useful, useless = self.filter_useful(x)
                print(len(useful), len(useless))

                seek_rate = len(useful) / seek_size
                if seek_rate>0: notfound = False
                
                if seek_rate == 0:
                    seek_size *= 2
                    if notfound and seek_size > 100000:
                        raise Exception('Failed when generating constraint dataset. Valid solution not found')
                else:
                    seek_size = int((expected_n - len(final_useful) - len(useful)) / seek_rate * 1.2)
                    
                seek_size = min(seek_size, int(1e6))
                

                final_useful += useful
                final_useless += useless[: max(0, num - len(final_useless))]
                
                if len(final_useful) >= expected_n:
                    break
                
            
            final_useful = final_useful[:expected_n]
            final_dataset = final_useful + final_useless[:num - len(final_useful)]
            self.detail_useful_data_num = len(final_useful)
            
            self.x = np.array(final_dataset)
            return self.x
            
        
        
    def sample_y(self):
        if self.is_constraint():
            self.y, self.cons = self.predict(self.x)
        else:
            self.y, self.cons = self.predict(self.x)
            self.cons = np.zeros_like(self.y)
        
        return self.y, self.cons

    def sample_bound(self, num=0, low=0, high=100):
        # if num == 0:
        #     self.sample_x(1)
        #     dim = self.x.shape[1]
        #     num = dim * 1000
        # self.sample_x(num)
        # self.sample_y()
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
