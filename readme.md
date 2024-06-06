# SOO-Bench

A benchmark suit to verify the performance of your algorithms for solving Offline MBO problem.

# Installation
### Install benchmark
```bash
git clone https://github.com/zhuyiyi-123/SOO-Bench
cd ./SOO-Bench

conda create -n SOO_Bench python=3.7 # create a new conda environment
pip install -m requirements.txt 
```
### Install revive_hybrid
#### Install revive
    pip install -e ./revive_hybrid
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
x = task.sample_x(num=10000)
y, cons = task.sample_y() # generate score list 'y' and constraint function value list 'cons' corresponding to x

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

### The Parameter ``taskname`` and ``benchmark``
You can find more information about these tasks in our paper .# 待补充,论文链接
#### ``taskname = 'gtopx_data'; benchmark = 1`` 
Constrained task gtopx 1 Cassini 1.

Number of data dimension ( len(task[0]) ) = 6.

Number of constraints functions( len(cons[0]) ) = 4

#### ``taskname = 'gtopx_data'; benchmark = 2`` 
Unconstrained task gtopx 2 Cassini 2.

Number of data dimension ( len(task[0]) ) = 22.

Number of constraints functions( len(cons[0]) ) = 0

#### ``taskname = 'gtopx_data'; benchmark = 3`` 
Unconstrained task gtopx 3 Messenger (reduced).

Number of data dimension ( len(task[0]) ) = 18.

Number of constraints functions( len(cons[0]) ) = 0

#### ``taskname = 'gtopx_data'; benchmark = 4`` 
Unconstrained task gtopx 4 Messenger (full).

Number of data dimension ( len(task[0]) ) = 26.

Number of constraints functions( len(cons[0]) ) = 0


#### ``taskname = 'gtopx_data'; benchmark = 5`` 
Constrained task gtopx 5 GTOC 1.

Number of data dimension ( len(task[0]) ) = 8.

Number of constraints functions( len(cons[0]) ) = 6


#### ``taskname = 'gtopx_data'; benchmark = 6`` 
Unconstrained task gtopx 6 Rosetta.

Number of data dimension ( len(task[0]) ) = 22.

Number of constraints functions( len(cons[0]) ) = 0


#### ``taskname = 'gtopx_data'; benchmark = 7`` 
Constrained task gtopx 7 Cassini 1-MINLP.

Number of data dimension ( len(task[0]) ) = 10.

Number of constraints functions( len(cons[0]) ) = 4

#### ``taskname = 'cec_data'; benchmark = 1`` 
Constrained task cec 1 Optimal operation.

Number of data dimension ( len(task[0]) ) = 7.

Number of constraints functions( len(cons[0]) ) = 14

#### ``taskname = 'cec_data'; benchmark = 2`` 
Constrained task cec 2 Process flow.

Number of data dimension ( len(task[0]) ) = 3.

Number of constraints functions( len(cons[0]) ) = 3


#### ``taskname = 'cec_data'; benchmark = 3`` 
Constrained task cec 3 Process synthesis.

Number of data dimension ( len(task[0]) ) = 2.

Number of constraints functions( len(cons[0]) ) = 2


#### ``taskname = 'cec_data'; benchmark = 4`` 
Constrained task cec 4 Three-bar.

Number of data dimension ( len(task[0]) ) = 2.

Number of constraints functions( len(cons[0]) ) = 3


#### ``taskname = 'cec_data'; benchmark = 5`` 
Constrained task cec 5 Welded beam.

Number of data dimension ( len(task[0]) ) = 4.

Number of constraints functions( len(cons[0]) ) = 5


#### ``taskname = 'hybrid_data'; benchmark = 0`` 
Constrained task hybrid 0 Constraint task.

Number of data dimension ( len(task[0]) ) = 115.

Number of constraints functions( len(cons[0]) ) = 2

#### ``taskname = 'hybrid_data'; benchmark = 1`` 
Unconstrained task hybrid 1 Unconstraint task.

Number of data dimension ( len(task[0]) ) = 115.

Number of constraints functions( len(cons[0]) ) = 0

#### ``taskname = 'mujoco_data'; benchmark = 1`` 
Unconstrained task protein 1 TF Bind 8.

Number of data dimension ( len(task[0]) ) = 8.

Number of constraints functions( len(cons[0]) ) = 0

#### ``taskname = 'mujoco_data'; benchmark = 2`` 
Unconstrained task protein 2 TF Bind 10.

Number of data dimension ( len(task[0]) ) = 10.

Number of constraints functions( len(cons[0]) ) = 0

## License


## Contact us


