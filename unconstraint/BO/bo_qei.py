import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import transformers
transformers.logging.set_verbosity_error()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_ import build_pipeline
from trainers import Ensemble
from nets import ForwardModel
import tensorflow as tf
import numpy as np
from par import get_parser
from botorch.acquisition.monte_carlo import qExpectedImprovement
from BO import obj_callable, initialize_model, optimize_acqf_and_get_observation
from botorch.acquisition.objective import GenericMCObjective
from botorch import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
import tqdm

import torch
import time
import warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
dtype = torch.float32
warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
from soo_bench.Taskdata import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
import pickle
def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res

def data_load(args, num):
    # create the training task
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, seed = args.seed)
    num = args.num*task.var_num

    if args.sample_method == 'sample_bound':
        task.sample_bound(num, args.low, args.high)
    elif args.sample_method == 'sample_limit':
        task.sample_limit(num, args.low, args.high)
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

def boqei(args, x, input_size, initial_x, initial_y, xl, xu ,input_shape, ensemble):
    NOISE_SE = args.bo_noise_se
    train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)
    # define a feasibility-weighted objective for optimization
    obj = GenericMCObjective(obj_callable)
    BATCH_SIZE = args.bo_batch_size
    bounds = torch.tensor([np.min(x, axis=0).reshape([input_size]).tolist(), np.max(x, axis=0).reshape([input_size]).tolist()], device=device, dtype=dtype)
    
    MC_SAMPLES = args.bo_mc_samples
    N_BATCH = args.bo_iterations

    best_observed_ei = []
    # scores = [np.max(initial_y)]
    save_xs = []
    # call helper functions to generate initial training data and initialize model
    train_x_ei = initial_x.numpy().reshape([initial_x.shape[0], input_size])
    train_x_ei = torch.tensor(train_x_ei).to(device, dtype=dtype)

    train_obj_ei = initial_y.numpy().reshape([initial_y.shape[0], 1])
    train_obj_ei = torch.tensor(train_obj_ei).to(device, dtype=dtype)
    
    best_observed_value_ei = train_obj_ei.max().item()
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, train_yvar)
    best_observed_ei.append(best_observed_value_ei)
    xl = torch.Tensor(xl).to(device)
    xu = torch.Tensor(xu).to(device)
        
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in tqdm.tqdm(range(1, N_BATCH + 1)):
        torch.cuda.empty_cache()
        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_ei)

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei, best_f=train_obj_ei.max(), objective=obj)

        # optimize and get new observation
        result = optimize_acqf_and_get_observation(args, qEI, bounds, BATCH_SIZE, NOISE_SE, input_shape, ensemble)
        if result is None:
            print("RuntimeError was encountered, most likely a "
                  "'symeig_cpu: the algorithm failed to converge'")
            break
        new_x_ei, new_obj_ei = result
        new_x_ei = torch.clamp(new_x_ei, xl, xu)
        save_xs.append(new_x_ei.detach().cpu().numpy())
        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

        # update progress
        best_value_ei = obj(train_x_ei).max().item()
        best_observed_ei.append(best_value_ei)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei, train_obj_ei, train_yvar, model_ei.state_dict())
        t1 = time.time()
        # print('iteration: {}, time: {:.2f}s'.format(iteration, t1-t0))
    return train_x_ei, train_obj_ei, save_xs

def bo_qei(args):
    x, y, input_shape, input_size, task = data_load(args, num=args.num)
    ensemble = train(args, x, y, input_shape)

    # select the top 1 initial designs from the dataset
    k = min(args.bo_gp_samples, y.shape[0])
    indices = tf.math.top_k(y[:, 0], k=k)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    train_x_ei, train_obj_ei, save_xs = boqei(args, x, input_size, initial_x, initial_y,task.xl, task.xu, input_shape, ensemble)

    x_sol = train_x_ei.detach().cpu().numpy()
    y_sol = train_obj_ei.detach().cpu().numpy()

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y_sol[:, 0], k=args.solver_samples)[1]
    solution = tf.gather(x_sol, indices, axis=0)
    solution = tf.reshape(solution, [-1, *input_shape])

    
    save_ys = []
    save_cons = []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)
    print('score:', np.max(save_ys, axis=1))
    # save results
    def save_data(algorithm_name, args, offline_x, offline_y, offline_cons, save_xs, save_ys, save_cons,other=None,is_constraint=False, level=2):
        save_name = algorithm_name + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(level):
            current_dir = os.path.dirname(current_dir)
        
        if is_constraint:
            os.makedirs(os.path.join(current_dir, 'results','constraint'), exist_ok=True)
            path = os.path.join(current_dir, 'results','constraint', save_name +'.savedata.pkl')
        else:
            os.makedirs(os.path.join(current_dir, 'results','unconstraint'), exist_ok=True)
            path = os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')

        print('saving to: ', path)
        
        offline_x = np.array(offline_x)
        offline_y = np.array(offline_y).flatten()
        if is_constraint:
            offline_cons = np.array(offline_cons)
        else:
            offline_cons = None
        for i in range(len(save_xs)):
            save_xs[i] = np.array(save_xs[i])
            save_ys[i] = np.array(save_ys[i]).flatten()
            if is_constraint:
                save_cons[i] = np.array(save_cons[i])
        if not is_constraint:
            save_cons = None

        save_data = {
            'algorithm_name': algorithm_name,
            'constraint': is_constraint,
            'args': args,
            'offline_x': offline_x,
            'offline_y': offline_y,
            'offline_cons': offline_cons,
            'xs':save_xs,
            'ys':save_ys,
            'cons':save_cons,
            'other': other
        }
        with open (path, 'wb') as f:
            import pickle
            pickle.dump(save_data, f)
    

    save_data('BO', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, is_constraint=False, level=2)

if __name__ == "__main__":
    import os
    import time
    time1 = time.time()
    
    def set_seed(seed):
        import random
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    args = get_parser()
    if args.change_optimization_step > 0:
        args.bo_iterations = args.change_optimization_step

    save_name = 'BO' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')):
        print('Already exists:', save_name +'.pkl')
        exit()
    set_seed(args.seed)
    bo_qei(args)
    
    time2 = time.time()
    print('time: ', time2 - time1)
