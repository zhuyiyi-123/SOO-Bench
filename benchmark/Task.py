import numpy as np
import os 
from ctypes import *

class make(object):
    def __init__(self, task, benchmark):
        self.task = task
        self.benchmark = benchmark
        if self.task == "gtopx_data":
            self.dataset = self.gtopx(self.benchmark)
        elif self.task == "cec_data":
            self.dataset = self.cec(self.benchmark)
        elif self.task == "hybrid_data":
            self.dataset = self.hybrid(self.benchmark)
        elif self.task == "mujoco_data":
            self.dataset = self.mujoco(self.benchmark)

    def gtopx(self, benchmark):
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
        return self.obj_num, self.var_num, self.con_num, self.xl, self.xu
    
    def cec(self, benchmark):
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
        return self.obj_num, self.var_num, self.con_num, self.xl, self.xu

    def hybrid(self, benchmark):
        if benchmark == 0:
            self.con_num = 2
        elif benchmark == 1:
            self.con_num = 0
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
        return self.obj_num, self.var_num, self.con_num, self.xl, self.xu

    def mujoco(self, benchmark):
        if benchmark == 1:
            self.obj_num = 1
            self.var_num = 8
            self.con_num = 0
            self.xl = [0, 0, 0, 0, 0, 0, 0, 0]
            self.xu = [3, 3, 3, 3, 3, 3, 3, 3]
        elif benchmark == 2:
            self.obj_num = 1
            self.var_num = 10
            self.con_num = 0
            self.xl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.xu = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        return self.obj_num, self.var_num, self.con_num, self.xl, self.xu

        