# SOO-Bench

# SOO-Bench

A benchmark suit to verify the performance of your algorithms for solving Offline MBO problem.

# Installation
### Install benchmark
```bash
git clone git@github.com:zhuyiyi-123/SOO-Bench.git
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

## License


## Contact us


