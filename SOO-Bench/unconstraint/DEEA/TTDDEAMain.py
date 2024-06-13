import numpy as np
import time
from RBFN import RBFN
from Latin import latin
import Test_Functions as fun
from GA import GA
import sys, os
from par import get_parser
from test import sample_dis
from workshop.Benchmark_new.benchmark.Taskdata import OfflineTask
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
    # print('model0update')
    error = abs(pre0-pre2)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx1, data[seq]))
    ytemp = np.append(numy1, (pre0[seq]+pre2[seq])/2)
    model[1].fit(xtemp, ytemp)
    # print('model1update')
    error = abs(pre0-pre1)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx2, data[seq]))
    ytemp = np.append(numy2, (pre0[seq]+pre1[seq])/2)
    model[2].fit(xtemp, ytemp)
    # print('model2update')

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
    task = OfflineTask(args.task, args.benchmark)
    lower_bound = task.xl
    upper_bound = task.xu
    dimension = task.var_num
    datanum = args.num

    x, y = sample_dis(task, args.benchmark, args.num, args.low, args.high)
    y = -y
    model = [0] * 3
    datanum = len(x)
    traindata = int(datanum / 3)
    model[0] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    model[1] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    model[2] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    resetmodel(x,y)

    max_iter = 100
    ga = GA(pop_size=100, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)
    ga.init_Population()
    for i in range(max_iter):
        updatemodel(ga.pop)
        print("gaga:", ga.pc)
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
    score = task.prediction(optimum)[0]
    endtime = time.perf_counter()
    print('Optimal solution :', optimum)
    print('best score:', score)
    print('Execution Time :', endtime - starttime)
