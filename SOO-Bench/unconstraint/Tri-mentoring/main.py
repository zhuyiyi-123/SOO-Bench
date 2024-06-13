import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')
import transformers

transformers.logging.set_verbosity_error()
from my_model import *
from transformers import logging
from torch.autograd import grad

import os
import re
import requests

import random
from utils import *
import design_bench
import argparse
import time
import sys
import pickle
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
parser.add_argument('--method', choices=['ensemble', 'triteach', 'simple'], type=str, default='simple')
parser.add_argument('--majority_voting', default=1, type=int)
parser.add_argument('--soft_label', default=1, type=int)
parser.add_argument('--seed1', default=1, type=int)
parser.add_argument('--seed2', default=10, type=int)
parser.add_argument('--seed3', default=100, type=int)
parser.add_argument('--benchmark', default=2, type=int)
parser.add_argument('--num', default=50000, type=int)
parser.add_argument('--low', default=20, type=int)
parser.add_argument('--high', default=100, type=int)
parser.add_argument('--log_step', default=1, type=int)
#
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:' + str(args.device) if use_cuda else "cpu")

design_choices = ['TFBind8-Exact-v0', 'Superconductor-RandomForest-v0',
                  'HopperController-Exact-v0', 'AntMorphology-Exact-v0',
                  'DKittyMorphology-Exact-v0', 'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                  'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']
set_seed(args.seed)
if args.task in design_choices:
    task = design_bench.make(args.task)
    task.design = True
    if args.task == 'TFBind10-Exact-v0':
        args.bs = 1024
else:
    task = OfflineTask(args.task, args.benchmark, args.seed)
    task.sample_bound(args.num, args.low, args.high)
    task.design = False


# task.map_normalize_x()
# task.map_normalize_y()

def train_proxy(args):
    if not task.design:
        task_y = task.y
        task_x = task.x
    else:
        task_y0 = task.y
        task_x, task_y, shape0 = process_data(task, args.task, task_y0)

    # task = design_bench.make(args.task)
    # if args.task == 'TFBind10-Exact-v0':
    #     args.bs = 1024
    # task_y0 = task.y
    # task_x, task_y, shape0 = process_data(task, args.task, task_y0)
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
            x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :]  # .to(device)
            y_batch = train_labels[t * args.bs:(t + 1) * args.bs]  # .to(device)
            pred = model(x_batch)
            loss = torch.mean(torch.pow(pred - y_batch, 2))
            tmp_loss = tmp_loss + loss.data
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            valid_preds = model(valid_logits)
        pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
        # print("epoch {} training loss {} pcc {} best pcc {}".format(e, tmp_loss / T, pcc, best_pcc))
        if pcc > best_pcc:
            best_pcc = pcc
            # print("epoch {} has the best loss {}".format(e, best_pcc))
            model = model.to(torch.device('cpu'))
            torch.save(model.state_dict(), "/root/workshop/Benchmark_new/Tri-mentoring/model/" + args.task + "_proxy_" + str(args.seed) + ".pt")
            model = model.to(device)


def design_opt(args):
    save_name = 'Trimentoring' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    if os.path.exists('/root/workshop/Benchmark_new/results/' + save_name + '.pkl'):
        print('already exist.')
        return
    
    # task = design_bench.make(args.task)
    # load_y(args.task)
    # task_y0 = task.y
    # task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    if not task.design:
        task_y = task.y
        task_x = task.x
        shape0 = task_x.shape
    else:
        task_y0 = task.y
        task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)
    indexs = torch.argsort(task_y.squeeze())
    index = indexs[-args.topk:]
    x_init = copy.deepcopy(task_x[index])
    # overall before evaluation unmask1
    final_scores = np.zeros(shape=(args.topk, args.Tmax + 1))
    max_score, median_score, _ = evaluate_sample(task, x_init, args.task, shape0)
    final_scores[:, 0] = task_y[index].reshape(args.topk).detach().cpu().numpy()
    # print("Before  max {} median {}\n".format(max_score, median_score))
    from tqdm import tqdm
    for x_i in tqdm(range(x_init.shape[0]), desc='Optimzation'):
        if args.method == 'simple':
            proxy = SimpleMLP(task_x.shape[1]).to(device)
            proxy.load_state_dict(
                torch.load("/root/workshop/Benchmark_new/Tri-mentoring/model/" + args.task + "_proxy_" + str(args.seed) + ".pt", map_location=device))
        else:
            proxy1 = SimpleMLP(task_x.shape[1]).to(device)
            proxy1.load_state_dict(
                torch.load("/root/workshop/Benchmark_new/Tri-mentoring/model/" + args.task + "_proxy_" + str(args.seed1) + ".pt", map_location=device))
            proxy2 = SimpleMLP(task_x.shape[1]).to(device)
            proxy2.load_state_dict(
                torch.load("/root/workshop/Benchmark_new/Tri-mentoring/model/" + args.task + "_proxy_" + str(args.seed2) + ".pt", map_location=device))
            proxy3 = SimpleMLP(task_x.shape[1]).to(device)
            proxy3.load_state_dict(
                torch.load("/root/workshop/Benchmark_new/Tri-mentoring/model/" + args.task + "_proxy_" + str(args.seed3) + ".pt", map_location=device))
        # define distill data
        candidate = x_init[x_i:x_i + 1]
        # unmask2
        score_before, score_before_median, _ = evaluate_sample(task, candidate, args.task, shape0)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
        for i in range(1, args.Tmax + 1):
            if args.method == 'simple':
                loss = -proxy(candidate)
            elif args.method == 'ensemble':
                loss = -1.0 / 3.0 * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            elif args.method == 'triteach':
                adjust_proxy(proxy1, proxy2, proxy3, candidate.data, x0=task_x, y0=task_y, \
                             K=args.K, majority_voting=args.majority_voting, soft_label=args.soft_label)
                loss = -1.0 / 3.0 * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
            if i % args.log_step == 0:
                score_after, score_aftet_median, _ = evaluate_sample(task, candidate.data, args.task, shape0)
                final_scores[x_i, i] = score_after
                # print("candidate {} score before {} score now {}".format(x_i, score_before.squeeze(),
                #                                                          score_after.squeeze()))
        x_init[x_i] = candidate.data
    max_score, median_score, _ = evaluate_sample(task, x_init, args.task, shape0)
    # print("After  max {} median {}\n".format(max_score, median_score))
    result = np.max(final_scores, axis=0)
    print('Scores: ', result)
    print('Final Score', result[-1])
    save_name = 'Trimentoring' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    with open('/root/workshop/Benchmark_new/results/' + save_name + '.pkl', 'wb') as f:
        import pickle
        pickle.dump(result, f)

if __name__ == "__main__":
    print("this is my setting", args)
    train_proxy(args)
    # if args.task in ['TFBind8-Exact-v0', 'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
    #                  'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
    args.ft_lr = 1e-1
    args.Tmax = 100
    design_opt(args)
    # if args.mode == 'train':
    #     train_proxy(args)
    # elif args.mode == 'design':
    #     if args.task in ['TFBind8-Exact-v0', 'CIFARNAS-Exact-v0', \
    #                      'TFBind10-Exact-v0', 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
    #         args.ft_lr = 1e-1
    #         args.Tmax = 100
    #     design_opt(args)
