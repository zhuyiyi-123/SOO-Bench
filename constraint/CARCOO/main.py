import os, sys
import numpy as np
import random
import torch
from core.logger import Logger as Logger
from core.arcoo.nets import DualThreeSurogateModel
from core.arcoo.trainers import DualHeadSurogateTrainer
from core.utils import RiskSuppressionFactor, sample_langevin, neg_cons
from core.data import TaskDataset
from core.arcoo.optimizesa import Optimizer
from soo_bench.Taskdata import *
from soo_bench.algorithm_helper import AlgorithmHelper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from par import get_parser

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stand_scores(data):
    mean = np.mean(data)
    std = np.std(data)
    stand_scores = (data - mean) / std
    return stand_scores

def arcoo(args):
    init_m = 0.02
    Ld_K_max = 64
    opt_config = {'energy_opt':True, 'opt_steps':200}
    if args.change_optimization_step > 0:
        opt_config['opt_steps'] = args.change_optimization_step
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, args.seed)
    num = args.num*task.var_num
    if args.sample_method == 'sample_bound':
        task.sample_bound(num, args.low, args.high)
    elif args.sample_method == 'sample_limit':
        task.sample_limit(num, args.low, args.high)

    # return
    x = torch.tensor(task.x).to(DEVICE)
    y = torch.tensor(task.y).to(DEVICE)
    cons = torch.tensor(task.cons).to(DEVICE)

    # build the dual-head surrogate model
    dhs_model = DualThreeSurogateModel(np.prod(x.shape[1:]), 2048, int(np.prod(y.shape[1:])), cons).to(DEVICE)
    init_m = init_m * np.sqrt(np.prod(x.shape[1:]))
    trainer = DualHeadSurogateTrainer(dhs_model, cons=cons, task = task.task,
                                      dhs_model_prediction_opt=torch.optim.Adam, dhs_model_energy_opt=torch.optim.Adam,
                                      surrogate_lr=0.001, init_m=init_m,
                                      ldk=64)
    # create data loaders
    dataset = TaskDataset(x, y, cons)
    train_dataset_size = int(len(dataset) * (1 - 0.2))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [train_dataset_size,
                                                                (len(dataset) - train_dataset_size)])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    validate_dl = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

    # train the surrogate model
    trainer.launch(train_dl, validate_dl, 64, True)

    # select the top k initial designs from the dataset
    new_x, new_y = neg_cons(x, y, cons, 0,1).feasible()
    indice = torch.topk(new_y[:, 0], 128)[1].unsqueeze(1)
    init_xt = new_x[indice].squeeze(1)
    init_yt = new_y[indice].squeeze(1)

    # get energy scalar
    energy_min = dhs_model(init_xt)[1].mean().detach().cpu().numpy()
    energy_max = dhs_model(sample_langevin(init_xt, dhs_model,
                                            stepsize=init_m,
                                            n_steps=Ld_K_max,
                                            noise=False
                                            ))[1].mean().detach().cpu().numpy()
    uc = RiskSuppressionFactor(energy_min, energy_max, init_m = init_m)
    
    optimizer = Optimizer(opt_config, trainer, init_xt, init_yt, dhs_model=dhs_model)

    # calculate the offline best score
    save_xs= optimizer.optimize(uc)
    
    save_ys, save_cons = [], []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)

    print('Scores: ', np.max(save_ys, axis=1).tolist())
    print('Final Score', np.max(save_ys[-1]))
    
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
    

    save_data('CARCOO', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, is_constraint=True, level=2)

    
    

if __name__ == '__main__':
    args = get_parser()
    save_name = 'CARCOO' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(current_dir, 'results','constraint', save_name +'.savedata.pkl')):
        print('Already exists:', save_name +'.savedata.pkl')
        exit()
    def set_seed(seed):
        import random
        import torch
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(args.seed)

for i in range(1):
    arcoo(args)                                                                                                    
