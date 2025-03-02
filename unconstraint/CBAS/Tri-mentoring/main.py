import numpy as np
import torch
import torch.optim as optim
import os,sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
from my_model import *

from utils import *
import argparse
import pickle
from soo_bench.Taskdata import *

TRIMENTORING_DIR = os.path.dirname(os.path.abspath(__file__))
def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res



def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res

parser = argparse.ArgumentParser(description="pairwise offline")
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--task', type=str,
                    default='gtopx_data')
# gtopx_data
parser.add_argument('--mode', choices=['design', 'train'], type=str, default='design')
# grad descent to train proxy
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--wd', default=0.0, type=float)
# grad ascent to obtain design
parser.add_argument('--Tmax', default=200, type=int)
parser.add_argument('--ft_lr', default=1e-3, type=float)
parser.add_argument('--topk', default=128, type=int)
parser.add_argument('--interval', default=200, type=int)
parser.add_argument('--K', default=10, type=int)
parser.add_argument('--method', choices=['ensemble', 'triteach', 'simple'], type=str, default='triteach')
parser.add_argument('--majority_voting', default=1, type=int)
parser.add_argument('--soft_label', default=1, type=int)
parser.add_argument('--seed1', default=1, type=int)
parser.add_argument('--seed2', default=10, type=int)
parser.add_argument('--seed3', default=100, type=int)
parser.add_argument('--benchmark', default=2, type=int)
parser.add_argument('--num', default=1000, type=int)
parser.add_argument('--low', default=25, type=int)
parser.add_argument('--high', default=75, type=int)
parser.add_argument('--sample_method', default="sample_bound", type=str)
parser.add_argument('--change_optimization_step', default=-1, type=int)
parser.add_argument('--log_step', default=1, type=int)

args = parser.parse_args()


save_name = 'Trimentoring' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
current_dir = os.path.dirname(os.path.abspath(__file__))
for i in range(2):
    current_dir = os.path.dirname(current_dir)
if os.path.exists(os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')):
    print('Already exists:', os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl'))
    exit()
    
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:' + str(args.device) if use_cuda else "cpu")

set_use_cache()
task = OfflineTask(args.task, args.benchmark, args.seed)
num = args.num*task.var_num
if args.sample_method == 'sample_bound':
    task.sample_bound(num, args.low, args.high)
elif args.sample_method == 'sample_limit':
    task.sample_limit(num, args.low, args.high)
task.design = False



def train_proxy(args, seed):
    if not task.design:
        task_y = task.y
        task_x = task.x
    else:
        task_y0 = task.y
        task_x, task_y, shape0 = process_data(task, args.task, task_y0)

    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)
    L = task_x.shape[0]
    indexs = torch.randperm(L)
    task_x = task_x[indexs]
    task_y = task_y[indexs]
    train_L = int(L * 0.90)
    # normalize labels
    train_labels0 = task_y[0: train_L]
    valid_labels = task_y[train_L:]
    # load logits
    train_logits0 = task_x[0: train_L]
    valid_logits = task_x[train_L:]
    T = int(train_L / args.bs) + 1
    # define model
    model = SimpleMLP(task_x.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # begin training
    best_pcc = -1
    from tqdm import tqdm
    for e in tqdm(range(args.epochs), desc='Training'):
        # adjust lr
        adjust_learning_rate(opt, args.lr, e, args.epochs)
        # random shuffle
        indexs = torch.randperm(train_L)
        train_logits = train_logits0[indexs]
        train_labels = train_labels0[indexs]
        tmp_loss = 0
        for t in range(T):
            x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :] 
            y_batch = train_labels[t * args.bs:(t + 1) * args.bs] 
            pred = model(x_batch)
            loss = torch.mean(torch.pow(pred - y_batch, 2))
            tmp_loss = tmp_loss + loss.data
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            valid_preds = model(valid_logits)
        pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
        if pcc > best_pcc:
            best_pcc = pcc
            if not os.path.exists(os.path.join(TRIMENTORING_DIR, 'model')):
                os.mkdir(os.path.join(TRIMENTORING_DIR, 'model'))
            model = model.to(torch.device('cpu'))
            path = os.path.join(TRIMENTORING_DIR, 'model', args.task + '_proxy_' + str(seed) + '.pt')
            torch.save(model.state_dict(), path)
            model = model.to(device)


def design_opt(args):
    
    if not task.design:
        task_y = task.y
        task_x = task.x
        shape0 = task_x.shape
    else:
        task_y0 = task.y
        task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)
    xl = torch.Tensor(task.xl).to(device)
    xu = torch.Tensor(task.xu).to(device)
    indexs = torch.argsort(task_y.squeeze())
    index = indexs[-args.topk:]
    x_init = copy.deepcopy(task_x[index])
    # overall before evaluation unmask1
    final_scores = np.zeros(shape=(args.topk, args.Tmax + 1))
    max_score, median_score, _ = evaluate_sample(task, x_init, args.task, shape0)
    final_scores[:, 0] = task_y[index].reshape(args.topk).detach().cpu().numpy()

    save_xs = [[] for i in range(1, args.Tmax + 1)]
    from tqdm import tqdm
    for x_i in tqdm(range(x_init.shape[0]), desc='Optimzation'):
        if args.method == 'simple':
            proxy = SimpleMLP(task_x.shape[1]).to(device)
            path = os.path.join(TRIMENTORING_DIR, 'model', args.task + '_proxy_' + str(args.seed) + '.pt')
            proxy.load_state_dict(
                torch.load(path, map_location=device))
        else:
            proxy1 = SimpleMLP(task_x.shape[1]).to(device)
            path = os.path.join(TRIMENTORING_DIR, 'model', args.task + '_proxy_' + str(args.seed1) + '.pt')
            proxy1.load_state_dict(
                torch.load(path, map_location=device))
            proxy2 = SimpleMLP(task_x.shape[1]).to(device)
            path = os.path.join(TRIMENTORING_DIR, 'model', args.task + '_proxy_' + str(args.seed2) + '.pt')
            proxy2.load_state_dict(
                torch.load(path, map_location=device))
            proxy3 = SimpleMLP(task_x.shape[1]).to(device)
            path = os.path.join(TRIMENTORING_DIR, 'model', args.task + '_proxy_' + str(args.seed3) + '.pt')
            proxy3.load_state_dict(
                torch.load(path, map_location=device))

        candidate = x_init[x_i:x_i + 1]
        score_before, score_before_median, _ = evaluate_sample(task, candidate, args.task, shape0)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
        for i in range(1, args.Tmax + 1):
            candidate_opt.zero_grad()
            if args.method == 'simple':
                loss = -proxy(candidate)
            elif args.method == 'ensemble':
                loss = -1.0 / 3.0 * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            elif args.method == 'triteach':
                adjust_proxy(proxy1, proxy2, proxy3, candidate.data, x0=task_x, y0=task_y, \
                             K=args.K, majority_voting=args.majority_voting, soft_label=args.soft_label)
                loss = -1.0 / 3.0 * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            loss.backward()
            candidate_opt.step()
            # if i % args.log_step == 0:
            #     score_after, score_aftet_median, _ = evaluate_sample(task, candidate.data, args.task, shape0)
            #     final_scores[x_i, i] = score_after
            with torch.no_grad():
                candidate.data = candidate.data.clamp(xl, xu)
            save_xs[i-1].append(np.array(candidate.data.detach().cpu()).reshape(-1).copy())
        x_init[x_i] = candidate.data
    

    # result = np.max(final_scores, axis=0)
    # print('Scores: ', result)
    # print('Final Score', result[-1])

    save_ys = []
    save_cons = []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)
    print('scores:', np.max(save_ys, axis=1))

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
    

    save_data('Trimentoring', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, is_constraint=False, level=2)

if __name__ == "__main__":
    print("this is my setting", args)
    if args.method == 'simple':
        train_proxy(args, args.seed)
    else:
        train_proxy(args, args.seed1)
        train_proxy(args, args.seed2)
        train_proxy(args, args.seed3)
    args.ft_lr = 1e-1
    args.Tmax = 100
    if args.change_optimization_step > 0:
        args.Tmax = args.change_optimization_step
    design_opt(args)