import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')
import transformers

transformers.logging.set_verbosity_error()

import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/workshop/Benchmark_new/ARCOO')
from core.data_ import TaskDataset
from core.logger import Logger as Logger
from core.utils import RiskSuppressionFactor, sample_langevin
from core.arcoo.trainers import DualHeadSurogateTrainer
from core.arcoo.nets import DualHeadSurogateModel
from core.arcoo.optimizer import Optimizer
import numpy as np
import os
import json
from par import get_parser
import torch
from workshop.Benchmark_new.benchmark.Taskdata import OfflineTask
from workshop.Benchmark_new.test import sample_dis
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

def arcoo(args, task, x, y):
    save_name = 'Arcoo' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    if os.path.exists('/root/workshop/Benchmark_new/results/' + save_name + '.pkl'):
        print('already exist.')
        return
    
    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).cuda()
    
    # build the dual-head surrogate model
    dhs_model = DualHeadSurogateModel(np.prod(x.shape[1:]), args.surrogate_hidden, np.prod(y.shape[1:])).cuda()
    
    init_m = args.init_m * np.sqrt(np.prod(x.shape[1:]))
    trainer = DualHeadSurogateTrainer(dhs_model,
                                    dhs_model_prediction_opt=torch.optim.Adam, dhs_model_energy_opt=torch.optim.Adam, 
                                    surrogate_lr=0.001, init_m=init_m,
                                    ldk=args.Ld_K)

    # create data loaders
    dataset = TaskDataset(x, y)
    train_dataset_size = int(len(dataset) * (1 - args.val_size))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                            [train_dataset_size, (len(dataset)- train_dataset_size)])
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.train_batch,
                                        shuffle=True)
    validate_dl = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.train_batch,
                                            shuffle=True)
    # train the surrogate model
    trainer.launch(train_dl, validate_dl, args.train_epoch, args.e_train)

    # torch.save(dhs_model.state_dict(), os.path.join(config['log_dir'], 'model_para.pkl'))
    # if config['save_model']:
    #     torch.save(dhs_model.state_dict(), os.path.join('./config/arcoo', config['log_dir'].split('/')[-3], 'model_para.pkl'))

    # select the top k initial designs from the dataset
    indice = torch.topk(y[:, 0], args.online_solutions_batch)[1].unsqueeze(1)
    init_xt = x[indice].squeeze(1)
    init_yt = y[indice].squeeze(1)

    # get energy scalar
    energy_min = dhs_model(init_xt)[1].mean().detach().cpu().numpy()
    energy_max = dhs_model(sample_langevin(init_xt, dhs_model,
                                            stepsize=init_m,
                                            n_steps=args.Ld_K_max,
                                            noise=False
                                            ))[1].mean().detach().cpu().numpy()
    uc = RiskSuppressionFactor(energy_min, energy_max, init_m = init_m)
    
    optimizer = Optimizer(args.opt_config, task, trainer, init_xt, init_yt, dhs_model=dhs_model)

    scores = [torch.max(init_yt).item()]
    optimizer.optimize(uc, scores=scores)
    print('Scores: ', scores)
    print('Final Score', scores[-1])
    save_name = 'Arcoo' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    with open('/root/workshop/Benchmark_new/results/' + save_name + '.pkl', 'wb') as f:
        import pickle
        pickle.dump(scores, f)

if __name__ == '__main__':
    def set_seed(seed):
        import random
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    args = get_parser()
    set_seed(args.seed)
    task = OfflineTask(args.task, args.benchmark, args.seed)
    task.sample_bound(args.num, args.low, args.high)
    x = task.x
    y = task.y
    # print(max(y))
    arcoo(args, task, x, y)
