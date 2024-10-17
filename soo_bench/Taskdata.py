import os
from ctypes import *
import numpy as np
from soo_bench.Taskunit import *

REGISTERED_TASK = dict()
def register_task(taskname:str, task):
    assert(type(taskname) is str)
    REGISTERED_TASK[taskname] = task

register_task('gtopx_data', Task_GTOPX)
register_task('cec_data', Task_CEC)
register_task('hybrid_data', Task_hybrid)
register_task('mujoco_data', Task_mujoco)

USE_CACHE = False
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
if not os.path.exists(CACHE_PATH):
    os.mkdir(CACHE_PATH)
def clear_cache():
    for item in os.listdir(CACHE_PATH):
        if not item.endswith('.cache.pkl'):
            continue
        os.remove(os.path.join(CACHE_PATH, item))
def change_cache_path(new_path):
    global CACHE_PATH
    CACHE_PATH = new_path
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)
def set_use_cache(use_cache=True):
    global USE_CACHE
    USE_CACHE = use_cache

class OfflineTask:
    def __init__(self, task, benchmark, seed = 1, use_cache=None, *args, **kwargs):
        if not task in REGISTERED_TASK:
            raise Exception('Task "{}" is not registered.'.format(task))
        
        if use_cache is None: use_cache = USE_CACHE
        self.task = task
        benchmark = int(benchmark)
        seed = int(seed)
        self.taskunit = REGISTERED_TASK[task](benchmark, seed, use_cache=use_cache, cache_path=CACHE_PATH, *args, **kwargs)
        self.taskunit:TaskUnit

    def is_constraint(self):
        return self.taskunit.is_constraint()

    def predict(self, x):
        x = np.array(x)
        if np.size(x) == 0: # updated
            return np.array([]), np.array([]) # updated
    
        if len(np.shape(x)) == 1:
            return self.predict([x])
        value, constraint = self.taskunit.predict(x)

        value = np.array(value)
        value = np.reshape(value, (np.size(value)))
        constraint = np.array(constraint)
        return value, constraint

    
    def filter_useful(self, dataset, cons = None):
        return self.taskunit.filter_useful(dataset, cons)
        
        
    def _sample_x_ignore_constraints(self, num=2):
        return self.taskunit._sample_x_ignore_constraints(num)

    def sample_x(self, num=2 , rate_satisfying_constraints=0.4, maxtry=10000000,seek_size=512, print_info=True):
        return self.taskunit.sample_x(num, rate_satisfying_constraints, maxtry,seek_size, print_info)
            
        
        
    def sample_y(self):
        return self.taskunit.sample_y()

    def sample_bound(self, num=0, low=0, high=100, rate_satisfying_constraints=0.4, *args, **kwargs):
        self.taskunit.sample_bound(num, low, high,rate_satisfying_constraints, *args, **kwargs)
    
    def read(self, x, y):
        return self.taskunit.read(x, y)
        

    def dataset(self, num):
        return self.taskunit.dataset(num)

    def normalize_x(self, x):
        return self.taskunit.normalize_x(x)

    def map_normalize_x(self):
        return self.taskunit.map_normalize_x()

    def normalize_y(self, y):
        return self.taskunit.normalize_y(y)

    def map_normalize_y(self):
        return self.taskunit.map_normalize_y()

    def normalize_cons(self, cons):
        return self.taskunit.normalize_cons(cons)

    def map_normalize_cons(self):
        return self.taskunit.map_normalize_cons()

    def denormalize_x(self, x):
        return self.taskunit.denormalize_x(x)

    def denormalize_y(self, y):
        return self.taskunit.denormalize_y(y)

    def denormalize_cons(self, cons):
        return self.taskunit.denormalize_cons(cons)

    def ood_x(self, num=10, rate=0.6):
        return self.taskunit.ood_x(num, rate)
    
    @property
    def x(self):
        return self.taskunit.x
    @property
    def y(self):
        return self.taskunit.y
    @property
    def cons(self):
        return self.taskunit.cons
    
    @property
    def obj_num(self):
        return self.taskunit.obj_num
    @property
    def var_num(self):
        return self.taskunit.var_num
    @property
    def con_num(self):
        return self.taskunit.con_num
    @property
    def xl(self):
        return self.taskunit.xl
    @property
    def xu(self):
        return self.taskunit.xu
    @property
    def benchmark(self):
        return self.taskunit.benchmark
    @property
    def seed(self):
        return self.taskunit.seed
    
    @property
    def use_cache(self):
        return self.taskunit.use_cache
    
    @property
    def cache_path(self):
        return self.taskunit.cache_path
    
    def __str__(self):
        return str(self.taskunit)
    def __repr__(self):
        return repr(self.taskunit)
    
    
