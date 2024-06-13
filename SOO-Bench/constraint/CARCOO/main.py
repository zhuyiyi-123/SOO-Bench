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
import pandas as pd
from constraint import gtopx_data
import matplotlib.pyplot as plt
from Taskdata import OfflineTask
from par import get_parser

def stand_scores(data):
    mean = np.mean(data)
    std = np.std(data)
    stand_scores = (data - mean) / std
    return stand_scores

def arcoo(args):
    init_m = 0.02
    Ld_K = 64
    Ld_K_max = 64
    opt_config = {'energy_opt':True, 'opt_steps':200}
    task = OfflineTask(args.task, args.benchmark, args.seed)
    args.num = task.var_num * 2000
    task.sample_bound(args.num, args.low, args.high)

    # return
    x = torch.tensor(task.x).cuda()
    y = torch.tensor(task.y).cuda()
    cons = torch.tensor(task.cons).cuda()

    # build the dual-head surrogate model
    dhs_model = DualThreeSurogateModel(np.prod(x.shape[1:]), 2048, int(np.prod(y.shape[1:])), cons).cuda()
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

    # save_model = True
    # if save_model:
    #     torch.save(dhs_model.state_dict(), os.path.join('model_para_surrogate_gtopx1.pkl'))
    # load the surrogate model
    # dhs_model.load_state_dict(torch.load('model_para_surrogate_gtopx8.pkl'))

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
    
    optimizer = Optimizer(opt_config, task, trainer, init_xt, init_yt, dhs_model=dhs_model)

    scores = optimizer.optimize(uc)
    print('Scores: ', scores)
    print('Final Score', scores[-1])
    save_name = 'CArcoo' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    with open('/root/workshop/Benchmark_new/con_results_last/' + save_name + '.pkl', 'wb') as f:
        import pickle
        pickle.dump(scores, f)

if __name__ == '__main__':
    args = get_parser()
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