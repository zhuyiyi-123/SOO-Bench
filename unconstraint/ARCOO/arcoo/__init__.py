import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import transformers
transformers.logging.set_verbosity_error()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ARCOO.data_ import TaskDataset
from ARCOO.utils import RiskSuppressionFactor, sample_langevin
from ARCOO.arcoo.trainers import DualHeadSurogateTrainer
from ARCOO.arcoo.nets import DualHeadSurogateModel
from ARCOO.arcoo.optimizer import Optimizer
import numpy as np
import os
from par import get_parser
import torch
from soo_bench.Taskdata import *
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res


def arcoo(args, task, x, y):
    
    x = torch.Tensor(x).to(DEVICE)
    y = torch.Tensor(y).to(DEVICE)
    
    # build the dual-head surrogate model
    dhs_model = DualHeadSurogateModel(np.prod(x.shape[1:]), args.surrogate_hidden, np.prod(y.shape[1:])).to(DEVICE)
    
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(3):
        current_dir = os.path.dirname(current_dir)
    os.makedirs(os.path.join(current_dir, 'results','unconstraint'), exist_ok=True)
    path = os.path.join(current_dir, 'results','unconstraint', save_name +'.pkl')
    print('saving to: ', path)
    with open (path, 'wb') as f:
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
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, args.seed)
    task.sample_bound(args.num, args.low, args.high)
    x = task.x
    y = task.y
    arcoo(args, task, x, y)
