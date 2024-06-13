# SOO-Bench

# SOO-Bench: Benchmarks for Evaluating the Stability of Offline Black-Box Optimization

SOO-Bench is introduced to address the limitations of existing benchmarks in assessing the stability of offline black-box optimization. It specifically targets real-world scenarios where direct evaluation of objective functions is infeasible, risky, or costly, such as in designing molecular formulas or mechanical structures. Unlike traditional optimization benchmarks that focus solely on achieving the optimal solution, SOO-Bench aims to reliably surpass performance metrics of offline datasets during the optimization process. This suite provides diverse offline optimization tasks and datasets in fields like satellite technology, materials science, structural mechanics, and automotive manufacturing, catering to different data distributions.

# Installation
### Install benchmark
```bash
git clone git@github.com:zhuyiyi-123/SOO-Bench.git
cd ./SOO-Bench
conda create -f Soo-bench/environment.yml # create a new conda environment
```
#### Modify torch setting
##### View the location of ``torch`` in the current SOO-Bench environment by

```bash
conda activate revive
python 
```


```python
import torch
torch.__file__
```


​    

##### Then modify the default parameters in ``serialization.py``

`` cd xxx/xxx/envs/revive/Lib/site-packages/torch/``

``def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):`` -> 
``def load(f, map_location="cpu", pickle_module=pickle, **pickle_load_args):``

## Usage

### Example
```python
import numpy as np
from benchmark.Taskdata import OfflineTask

def your_algorithm(x,y):
    'repace this to your real algorithm function'
    return [x[0]]

taskname = 'gtopx_data' # explained later
benchmarkid = 1 # explained later
seed = 0

task = OfflineTask(task = taskname,  benchmark=benchmarkid, seed = seed)

# generate sample list 'x'
task.sample_bound(num=10000, low=0, high=100)
x = task.x
y, cons = task.y, task.cons # generate score list 'y' and constraint function value list 'cons' corresponding to x


# filter the samples that violate the constraints.
if task.is_constraint():
    # a valid samples is defined that every constraint function value is non negtive. 
    indx = []
    for con in cons:
        indx.append(np.all(con>=0))
    x = x[indx]
    y = y[indx]
    cons = cons[indx]

x_after = your_algorithm(x, y)
# Apply your optimization algorithm to the dataset.
# The 'your_algorithm' is a placeholder for the actual algorithm you would use.

score, cons = task.predict(x_after)
print(score)
```
## Running Algorithms
```python
./run.sh
```

## License


## Contact us


