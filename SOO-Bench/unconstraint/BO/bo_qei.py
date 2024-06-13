import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')
import transformers

transformers.logging.set_verbosity_error()

from data_ import StaticGraphTask, build_pipeline
from logger import Logger
from trainers import Ensemble, VAETrainer
from nets import ForwardModel, SequentialVAE
from utils import render_video
import tensorflow as tf
import numpy as np
import os
from par import get_parser
import sys
import os
import wandb
from botorch.acquisition.monte_carlo import qExpectedImprovement
from BO import obj_callable, initialize_model, optimize_acqf_and_get_observation
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning

import torch
import time
import warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
from workshop.Benchmark_new.benchmark.Taskdata import OfflineTask
import pickle
def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res

def OfflineTask(task,benchmark,seed):
    root = '/root/workshop/Benchmark_new/cache_uncons/'
    nameli = os.listdir(root)
    num = []
    for name in nameli:
        if name.startswith(f'{task}_id{benchmark}_num') == False: continue
        if name.endswith(f'_seed{seed}.pkl') == False: continue
        # print(name)
        val = name[len(f'{task}_id{benchmark}_num') : - len(f'_seed{seed}.pkl')]
        # print(val)
        num.append(int(val))
    num = num[-1]
    name = default_name(task, benchmark, num,seed)
    
    res =  load(root + name)
    
    def sample_x(n=2, / , rate_satisfying_constraints=0.4, maxtry=10000000000):
        # if(num != n): print(f'warning: task{task}benchmark{benchmark}实际大小为{num}, 你设置的大小为{n}')
        return res.x
    def sample_y():
        return res.y,res.cons
    res.sample_x = sample_x
    res.sample_y = sample_y
    return res

def data_load(args, num):
    # create the training task
    task = OfflineTask(args.task, args.benchmark, seed = args.seed)

    task.sample_bound(num, args.low, args.high)
    x = task.x
    y = task.y
    if True:
        # y = np.array(task.normalize_y(y[0])).astype(np.float32).reshape(num,task.obj_num)
        y = np.array(y).astype(np.float32).reshape(len(x),task.obj_num)
    if True:
        # x = np.array(task.normalize_x(x)).astype(np.float32).reshape(num,task.var_num)
        x = np.array(x).astype(np.float32).reshape(len(x),task.var_num)
    input_shape = x.shape[1:]
    input_size = np.prod(input_shape)
    return x, y, input_shape, input_size, task

def train(args, x, y, input_shape):
    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, bootstraps=args.bootstraps,
        batch_size=args.ensemble_batch_size,
        val_size=args.val_size)
    
    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        input_shape,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        initial_max_std=args.initial_max_std,
        initial_min_std=args.initial_min_std)
        for b in range(args.bootstraps)]
    
    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=args.ensemble_lr)
    
    # train the model for an additional number of epochs
    ensemble.launch(train_data,
                    val_data,
                    args.ensemble_epochs)
    return ensemble

def boqei(args, x, input_size, initial_x, initial_y, input_shape, task, ensemble):
    NOISE_SE = args.bo_noise_se
    train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)
    # define a feasibility-weighted objective for optimization
    obj = GenericMCObjective(obj_callable)
    BATCH_SIZE = args.bo_batch_size
    bounds = torch.tensor([np.min(x, axis=0).reshape([input_size]).tolist(), np.max(x, axis=0).reshape([input_size]).tolist()], device=device, dtype=dtype)
    
    MC_SAMPLES = args.bo_mc_samples
    N_BATCH = args.bo_iterations

    best_observed_ei = []
    scores = [np.max(initial_y)]
    # call helper functions to generate initial training data and initialize model
    train_x_ei = initial_x.numpy().reshape([initial_x.shape[0], input_size])
    train_x_ei = torch.tensor(train_x_ei).to(device, dtype=dtype)

    train_obj_ei = initial_y.numpy().reshape([initial_y.shape[0], 1])
    train_obj_ei = torch.tensor(train_obj_ei).to(device, dtype=dtype)
    
    best_observed_value_ei = train_obj_ei.max().item()
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, train_yvar)
    best_observed_ei.append(best_observed_value_ei)
        
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_ei)

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei, best_f=train_obj_ei.max(), objective=obj)

        # optimize and get new observation
        result = optimize_acqf_and_get_observation(args, qEI, bounds, BATCH_SIZE, NOISE_SE, input_shape, task, ensemble)
        if result is None:
            print("RuntimeError was encountered, most likely a "
                  "'symeig_cpu: the algorithm failed to converge'")
            break
        new_x_ei, new_obj_ei = result
        scores.append(max(new_obj_ei).detach().cpu().numpy().tolist())

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        print(train_obj_ei, new_obj_ei)
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei.unsqueeze(1)])
        print(train_obj_ei)

        # update progress
        best_value_ei = obj(train_x_ei).max().item()
        best_observed_ei.append(best_value_ei)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei, train_obj_ei, train_yvar, model_ei.state_dict())

        t1 = time.time()
        print(f"Batch {iteration:>2}: best_value = "f"({best_value_ei:>4.2f}), "f"time = {t1 - t0:>4.2f}.", end="")
    return train_x_ei, train_obj_ei, scores

def bo_qei(args):
    save_name = 'BO' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    if os.path.exists('/root/workshop/Benchmark_new/results/' + save_name + '.pkl'):
        print('already exist.')
        return
    
    x, y, input_shape, input_size, task = data_load(args, num=args.num)
    ensemble = train(args, x, y, input_shape)

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=args.bo_gp_samples)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    train_x_ei, train_obj_ei, scores = boqei(args, x, input_size, initial_x, initial_y, input_shape, task, ensemble)
    
    if torch.cuda.is_available():
        x_sol = train_x_ei.detach().cpu().numpy()
        y_sol = train_obj_ei.detach().cpu().numpy()

    else:
        x_sol = train_x_ei.detach().numpy()
        y_sol = train_obj_ei.detach().numpy()

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y_sol[:, 0], k=args.solver_samples)[1]
    solution = tf.gather(x_sol, indices, axis=0)
    solution = tf.reshape(solution, [-1, *input_shape])

    # save the current solution to the disk
    if args.do_evaluation:
        # evaluate the found solution and record a video
        score = task.predict(solution)
        print("score:", min(score[0]), max(score[0]))
    save_name = 'BO' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    with open('/root/workshop/Benchmark_new/results/' + save_name + '.pkl', 'wb') as f:
        import pickle
        pickle.dump(scores, f)

if __name__ == "__main__":
    def set_seed(seed):
        import random
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    args = get_parser()
    set_seed(args.seed)
    bo_qei(args)
