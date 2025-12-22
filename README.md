<h1 align="center">SOO-Bench: Benchmarks for Evaluating the Stability of Offline Black-Box Optimization</h1>

<p align="center">
  <strong>Hong Qian</strong>¹, <strong>Yiyi Zhu</strong>¹, <strong>Xiang Shu</strong>¹,
  <strong>Shuo Liu</strong>¹, <strong>Yaolin Wen</strong>¹,<br>
  <strong>Xin An</strong>¹, <strong>Huakang Lu</strong>¹,
  <strong>Aimin Zhou</strong>*¹,
  <strong>Ke Tang</strong>²,
  <strong>Yang Yu</strong>³˒⁴
</p>

<p align="center">
  * Corresponding Author
</p>

<p align="center">
  ¹ <strong>East China Normal University</strong> | ² <strong>Southern University of Science and Technology</strong> | ³ <strong>Nanjing University</strong> | ⁴ <strong>Polixir Technologies, China</strong>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=bqf0aCF3Dd">
    <img src="https://img.shields.io/badge/Paper-ICLR-red" />
  </a>
  <a href="[link]([https://github.com/zhuyiyi-123/SOO-Bench](https://iclr.cc/media/PosterPDFs/ICLR%202025/29089.png?t=1744277452.3747146))">
    <img src="https://img.shields.io/badge/Poster-ICLR-blue" />
  </a>
  <a href="[link](https://github.com/zhuyiyi-123/SOO-Bench)">
    <img src="https://img.shields.io/badge/GitHub-Repo-black" />
  </a>
</p>

---
![](./src/Figure1.png)

🎉 This repository contains the code of "SOO-Bench: Benchmarks for Evaluating the Stability of Offline Black-Box Optimization". Unlike traditional benchmarks that focus primarily on identifying the optimal solution, SOO-Bench incorporates a **stability indicator** specifically designed to assess the stability of algorithms throughout the whole optimization process. Additionally, this suite provides a diverse array of offline optimization tasks and datasets spanning various fields such as satellite technology, materials science, structural mechanics, and automotive manufacturing. Each component is **highly customizable**, allowing adjustments in data sampling distribution, task complexity, constraints, and more.

# :page_facing_up: Offline Black-Box Optimization

Offline optimization aims to make full use of the offline data and finally recommend potential high-quality solutions to be applied online without actively querying the solutions online. Due to the exploration nature of global optimization, optimizers may tend to try unseen out-of distribution solutions for potentially better performance (i.e., find much better solutions than the offline dataset). As the optimization procedure proceeds, the quality of solutions may tend to decline. It is difficult to determine when to stop the optimization procedure and return the final solution without active evaluation feedback. Previous work has suggested that by specifying the number of stopping steps for optimization, good results can be achieved by adjusting the step size or modifying the model for experiments. However, this is often not feasible in real life. Therefore, we hope to maintain a steady improvement throughout the optimization process, or at least maintain stability so that the solutions at any step will not be too bad.

## 📊 Performance

SOO-Bench evaluates algorithms across multiple dimensions including stability (SI) and optimality (FS). FS (final score) measures the function value found in the final optimization step, while SI (stability indicator) measures stability throughout the optimization process. Larger FS and SI values indicate better performance.

### Unconstrained Optimization Results

Overall results in unconstrained scenario. Results are averaged over five times, and "±" indicates the standard deviation.

| Tasks | Cassini2 | | Reduced | | Full | | Rosetta | |
|-------|----------|-|---------|-|------|-|---------|-|
| $f(\mathcal{X}^{*}_{\rm OFF})$ | 196.21 | | 151.68 | | 216.34 | | 112.11 | |
| Metrics | FS ↑ | SI ↑ | FS ↑ | SI ↑ | FS ↑ | SI ↑ | FS ↑ | SI ↑ |
| BO | 95.45±18.43 | 0.69±0.02 | 75.78±29.38 | 0.70±0.03 | 117.39±17.11 | 0.68±0.04 | 58.11±8.25 | 0.63±0.04 |
| CMAES | 196.21±1.18 | 0.00±0.00 | 151.68±0.53 | 0.00±0.00 | 216.34±0.61 | 0.00±0.00 | 112.11±0.38 | 0.00±0.00 |
| CBAS | 196.21±1.18 | 0.00±0.00 | 86.51±3.97 | 0.71±0.06 | 208.80±15.50 | 0.14±0.02 | 87.08±31.05 | 0.34±0.09 |
| TTDEEA | 224.17±53.87 | -2.34±2.58 | 156.92±91.05 | -2.28±2.78 | 260.40±54.89 | -∞ | 148.76±50.67 | -∞ |
| ARCOO | 90.73±10.98 | 0.78±0.05 | 65.88±13.12 | 0.85±0.01 | 102.84±21.76 | 0.79±0.04 | 65.17±13.30 | 0.74±0.08 |
| Tri-mentoring | 129.47±54.75 | -∞ | 140.23±22.88 | -∞ | 176.31±37.30 | 0.86±0.18 | 112.11±0.38 | -∞ |

### Constrained Optimization Results

Overall results in constrained scenario. The symbol "-" means that the algorithm cannot work because of too few solutions that satisfy the constraints.

| Tasks | Cassini1 | | GTOC1 | | Cassini1-MINLP | |
|-------|----------|-|-------|-|----------------|-|
| $f(\mathcal{X}^{*}_{\rm OFF})$ | 75.30 | | 5.41 | | 346.20 | |
| Metrics | FS ↑ | SI ↑ | FS ↑ | SI ↑ | FS ↑ | SI ↑ |
| CARCOO | 72.97±5.64 | 0.34±0.04 | 1.08±5.85 | -∞ | 189.14±118.59 | 0.64±0.06 |
| CCOMs | 68.61±108.20 | -0.94±0.55 | - | -∞ | 307.81±19.07 | 0.99±0.00 |
| DDEA-PF | 282.57±35.59 | -∞ | 0.00±0.00 | -∞ | 953.59±27.33 | -∞ |
| DDEA-SPF | 282.57±35.59 | -∞ | 0.00±0.00 | -∞ | 953.59±27.33 | -∞ |

## :wrench: Installation

SOO-Bench can be installed with the complete set of benchmarks via our pip package. For a stable installation and usage, we suggest that you use `CUDA version 11.7`  or higher, and python > 3.8. 

Firstly, SOO-Bench can be installed using anaconda.


## 🛠️ Installation

### Prerequisites

* Python 3.8+
* CUDA 11.8 or higher (recommended)
* Conda package manager

Tested on:

* Ubuntu 20.04, 1 × 2080Ti, CUDA 11.8
* Ubuntu 20.04, 1 × 3080Ti, CUDA 11.8
* Ubuntu 22.04, 1 × 3090, CUDA 12.0

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/zhuyiyi-123/SOO-Bench.git
cd SOO-Bench

# Create a conda environment
conda create -n soo-bench python=3.8 -y
conda activate soo-bench


# Install SOO-Bench
pip install -e .

# Install dependencies of baselines

pip install -r requirements.txt
```

### Test Installation

```python
from soo_bench.Taskdata import OfflineTask, REGISTERED_TASK

print(REGISTERED_TASK.keys())
# dict_keys(['gtopx_data', 'cec_data', 'mujoco_data'])

# Test GTOPX benchmark
task = OfflineTask('gtopx_data', benchmark=2, seed=1)

num = 2
task.sample_x(num)
task.sample_y()
x = task.x
y = task.y
print(x)
print(y)
```

## 🔍 File System

```bash
.
├── README.md
├── requirements.txt
├── setup.py
├── install.sh
├── run.sh
├── run_*.sh 
├── src/
│   ├── Figure1.png
│   └── Figure2.png
├── soo_bench/
│   ├── __init__.py
│   ├── Taskdata.py
│   ├── Taskunit.py
│   ├── utils.py
│   ├── gtopx.dll
│   ├── gtopx.so
│   ├── protein/ 
│   └── design_bench_data/ 
├── unconstraint/ (unconstrained optimization algorithms)
│   ├── CC-DDEA/
│   ├── ARCOO/
│   ├── CMAES/
│   ├── Tri-mentoring/
│   └── CBAS/
├── constraint/ (constrained optimization algorithms)
├── scripts/
│   └── result_gather.py
└── paper/
    └── SOO_Bench.pdf
```

- `soo_bench/`: Core package containing task data handling and algorithm implementations
- `src/`: Contains figures and visualizations used in documentation
- `unconstraint/`: Directory containing unconstrained optimization algorithms
- `constraint/`: Directory containing constrained optimization algorithms
- `scripts/`: Utility scripts for result processing
- `run.sh`: Main script to run all baseline algorithms
- `run_*.sh`: Scripts for specific distribution settings

## 🚦 Usage

### 1. Generate Data

```python
from soo_bench.Taskdata import OfflineTask, REGISTERED_TASK

task = OfflineTask(task='gtopx_data', benchmark=1, seed=1)
print('Available tasks:', REGISTERED_TASK.keys())

# Method 1 (Recommended)
n = 10
task.sample_bound(n)
x = task.x
y = task.y
print(x, y)

# Method 2
n = 10
task.sample_x(n)
task.sample_y()
x = task.x
y = task.y
print(x, y)
```

### 2. Use Oracle Function

```python
from soo_bench.Taskdata import OfflineTask
import numpy as np

task = OfflineTask(task='gtopx_data', benchmark=1, seed=1)
x = task.sample_x()
y = task.sample_y()

ndim = task.var_num  # dimension of x
xx = np.array([.0 for i in range(ndim)])
yy, cons = task.predict(xx)  # returns oracle function value and constraint values
print(yy, cons)
```

### 3. Add Custom Dataset

```python
import numpy as np
from soo_bench.Taskdata import OfflineTask, register_task
from soo_bench.Taskunit import TaskUnit

# Define custom dataset
class Task_simple(TaskUnit):
    TASK_NAME = 'simple_data'

    def init(self, benchmark, seed, *args, **kwargs):
        if benchmark == 1: 
            self.obj_num = 1
            self.var_num = 2
            self.con_num = 0
            self.xl = [-10, -10]
            self.xu = [10, 10]
        elif benchmark == 2:
            self.obj_num = 1
            self.var_num = 3
            self.con_num = 1
            self.xl = [-10, -10, -10]
            self.xu = [10, 20, 30]
        else:
            raise Exception(f"benchmark {benchmark} not found")
    
    def predict(self, x: np.ndarray):
        if self.benchmark == 1:
            result = -x[:,0]**2 - x[:,1]**2
            return result, 0
        elif self.benchmark == 2:
            result = -x[:,0]**2 - x[:,1]**2 - x[:,2]**2
            g = np.zeros((len(x), 1))
            g[:,0] = x[:,0] + x[:,1]**2
            return result, g

# Register and use custom dataset
register_task('simple_data', Task_simple)
task = OfflineTask('simple_data', benchmark=1, seed=1)
task.sample_bound(10)
print(task.x)
print(task.y)
```

### 4. Use Cache for Acceleration

```python
from soo_bench.Taskdata import set_use_cache, change_cache_path, clear_cache

set_use_cache(True)  # Enable cache for acceleration
# change_cache_path('./your_cache_path')  # Custom cache path
clear_cache()  # Clear previous cache

task = OfflineTask('gtopx_data', benchmark=2, seed=1)
task.sample_bound(10)  # Uses cache for acceleration
print(task.x)
print(task.y)
```

### 5. Run Benchmark Algorithms

```bash
# Run all algorithms
./run.sh
```

Results will be saved in:
```bash
ls ./results_y_ood/summary/details/
ls ./results_x_ood/summary/details/
ls ./results_different_n/summary/details/
```

## 🎨 Customization

You can perform advanced customization on several sections, as follows:

![](./src/Figure2.png)

### Data Missingness Control

Adjust the difficulty by controlling data missingness:

```python
# low, high controls missingness degree. e.g. low=0, high=60 indicates that samples with the 40% largest objective value are pruned.
task.sample_bound(n, low=0, high=60)
```

### Custom Data Distribution

Override sampling methods for custom distributions:

```python
import numpy as np
import random
from soo_bench.Taskdata import OfflineTask, register_task
from soo_bench.Taskunit import Task_GTOPX

class Task_GTOPX_user(Task_GTOPX):
    TASK_NAME = 'user_gtopx_data'
    
    def sample_x_ignore_constraints(self, num=2):
        result = []
        for _ in range(num):
            x_values = [random.uniform(lower, upper) for lower, upper in zip(self.xl, self.xu)]
            result.append(x_values)
        return result

register_task('user_gtopx_data', Task_GTOPX_user)
task = OfflineTask('user_gtopx_data', benchmark=3, seed=1)
task.sample_bound(10)
print(task.x)
```

## 📝 Citation
```
@inproceedings{qian2025soobench,
 author = {Hong Qian and Yiyi Zhu and Xiang Shu and Shuo Liu and Yaolin Wen and Xin An and Huakang Lu and Aimin Zhou and Ke Tang and Yang Yu},
 booktitle = {Proceedings of the 13th International Conference on Learning Representations},
 title = {SOO-Bench: Benchmarks for Evaluating the Stability of Offline Black-Box Optimization},
 year = {2025},
 address = {Singapore}
}
```

