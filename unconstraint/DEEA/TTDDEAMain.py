import numpy as np
import time
from RBFN import RBFN
from GA import GA
import sys, os
from par import get_parser
import os,sys
from soo_bench.Taskdata import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()


def updatemodel(data):
    global numx0, numy0, numx1, numy1, numx2, numy2
    pre0 = model[0].predict(data)
    pre1 = model[1].predict(data)
    pre2 = model[2].predict(data)

    error = abs(pre1-pre2)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx0, data[seq]))
    ytemp = np.append(numy0, (pre1[seq]+pre2[seq])/2)
    model[0].fit(xtemp, ytemp)
    error = abs(pre0-pre2)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx1, data[seq]))
    ytemp = np.append(numy1, (pre0[seq]+pre2[seq])/2)
    model[1].fit(xtemp, ytemp)
    error = abs(pre0-pre1)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx2, data[seq]))
    ytemp = np.append(numy2, (pre0[seq]+pre1[seq])/2)
    model[2].fit(xtemp, ytemp)

def resetmodel(x,y):
    global numx0, numx1, numx2, numy0, numy1, numy2
    shuffledata = np.column_stack((y, x))
    np.random.shuffle(shuffledata)
    newx = shuffledata[:, 1:]
    newy = shuffledata[:, :1]
    numx0 = newx[:traindata, ]
    numy0 = newy[:traindata, ]
    numx1 = newx[traindata:2 * traindata, ]
    numy1 = newy[traindata:2 * traindata, ]
    numx2 = newx[datanum - traindata:, ]
    numy2 = newy[datanum - traindata:, ]

    model[0].fit(numx0, numy0)
    model[1].fit(numx1, numy1)
    model[2].fit(numx2, numy2)

if __name__ == '__main__':
    args = get_parser()
    starttime = time.perf_counter()
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, seed=args.seed)
    lower_bound = task.xl
    upper_bound = task.xu
    dimension = task.var_num
    datanum = args.num

    task.sample_bound(args.num, args.low, args.high)
    x = task.x
    y = task.y
    first_score = np.max(y)
    model = [0] * 3
    datanum = len(x)
    traindata = int(datanum / 3)
    model[0] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    model[1] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    model[2] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    resetmodel(x,y)

    max_iter = 5 
    ga = GA(pop_size=100, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)
    ga.init_Population()
    for i in range(max_iter):
        updatemodel(ga.pop)
        ga.crossover(ga.pc)  
        ga.mutation(ga.pm) 
        ga.pop = np.unique(ga.pop, axis=0)
        for j in range(0, 3):
            temp = model[j].predict(ga.pop)
            if j == 0:
                fit_value = temp
            else:
                fit_value = fit_value + temp
        fit_value = fit_value.reshape((len(ga.pop), 1))
        ga.selection(fit_value)  
        resetmodel(x,y)

    optimum = ga.first[-1]
    print('optimum', optimum)
    best_score = task.predict([optimum])[0]
    endtime = time.perf_counter()
    print('Optimal solution :', optimum)
    print('best score:', best_score)
    print('Execution Time :', endtime - starttime)

    scores = [first_score]
    scores += task.predict(ga.first)[0].tolist()
    save_name = 'TTDDEA' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    os.makedirs(os.path.join(current_dir, 'results','unconstraint'), exist_ok=True)
    path = os.path.join(current_dir, 'results','unconstraint', save_name +'.pkl')
    print('saving to: ', path)
    with open (path, 'wb') as f:
        import pickle
        pickle.dump(scores, f)