import os
import re
import requests
import numpy as np
import functools

import random
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import scipy
import scipy.stats
import itertools
import higher

import math

d = None


def load_d(task):
    global d
    d = np.load("npy/" + task + ".npy", allow_pickle=True)
    print("loading shape", d.shape)


weights = None


def load_weights(task_name, y, gamma):
    global weights
    tmp = np.exp(gamma * y)
    weights = tmp / np.sum(tmp)
    print("weights", np.max(weights), np.min(weights))


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# y_min = None
# y_max = None
# def load_y(task_name):
#     global y_min
#     global y_max
#     dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
#     y_min, y_max = dic2y[task_name]

def process_data(task, task_name, task_y0=None, benchmark=0):
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0',
                     'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                     'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
        task_x = task.to_logits(task.x)
        if task_name == 'TFBind10-Exact-v0':
            interval = np.arange(0, 4161482, 830, dtype=int)[0: 5000]
            index = np.argsort(task_y0.squeeze())
            index = index[interval]
            task_y0 = task_y0[index]
            task_x = task_x[index]
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
        task_x = copy.deepcopy(task.x)
    task_x = task.normalize_x(task_x)
    shape0 = task_x.shape
    task_x = task_x.reshape(task_x.shape[0], -1)
    task_y = task.normalize_y(task_y0)
    return task_x, task_y, shape0


def evaluate_sample(task, x_init, task_name=None, shape0=None):
    # y_min = np.min(task.y)
    # y_max = np.max(task.y)
    # max_v = (np.max(Y1) - y_min) / (y_max - y_min)
    # med_v = (np.median(Y1) - y_min) / (y_max - y_min)
    if not task.design:
        tmp = task.predict(x_init.cpu())[0]
        Y1 = tmp
        return np.max(Y1), np.min(Y1), Y1
    else:
        x_init = x_init.cpu().numpy()
        if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0',
                         'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                         'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
            X1 = x_init.reshape(-1, shape0[1], shape0[2])
        elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                           'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
            X1 = x_init
        # X1 = task.denormalize_x(X1)
        if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0',
                         'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                         'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
            X1 = task.to_integers(X1)
        Y1 = task.predict(X1)
        return np.max(Y1), np.median(Y1), Y1


def adjust_learning_rate(optimizer, lr0, epoch, T):
    lr = lr0 * (1 + np.cos((np.pi * epoch * 1.0) / (T * 1.0))) / 2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_pcc(valid_preds, valid_labels):
    vx = valid_preds - torch.mean(valid_preds)
    vy = valid_labels - torch.mean(valid_labels)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-12) * torch.sqrt(torch.sum(vy ** 2) + 1e-12))
    return pcc


def compute_invpair(r1, r2):
    tau, p_value = scipy.stats.kendalltau(r1, r2)
    N = r1.shape[0]
    return round(N * (N - 1) * (1 - tau) / 4)


def compute_meanrank(candidates):
    m = np.mean(candidates, axis=0)
    rank = np.arange(candidates.shape[1])
    rank[np.argsort(m)] = np.arange(candidates.shape[1])
    return rank


def compute_bestrank(candidates):
    best_ranking = None
    best_loss = 666666
    # generate all possible rankings
    ranking = np.arange(candidates.shape[1])
    rankings = list(itertools.permutations(ranking))
    for ranking in rankings:
        ranking = np.array(ranking)
        ranking_loss = 0
        for i in range(candidates.shape[0]):
            ranking_loss = ranking_loss + compute_invpair(ranking, candidates[i])
        if ranking_loss < best_loss:
            best_loss = ranking_loss
            best_ranking = ranking
    return best_ranking


def adjust_bpr(proxy, pairs, labels, x0, y0, soft_label=True):
    # pairs: 2xN2xD
    # labels: N2, 0 or 1
    # x0: NxD
    # y0: N
    labels.requires_grad = True
    opt = torch.optim.SGD(proxy.parameters(), lr=1e-3)
    if soft_label:
        with higher.innerloop_ctx(proxy, opt) as (fproxy, diffopt):
            # inner level
            prob = torch.sigmoid(fproxy(pairs[0, :]) - fproxy(pairs[1, :])).squeeze()
            ce_loss = -torch.mean(labels * torch.log(prob + 1e-9) + (1 - labels) * torch.log(1 - prob + 1e-9))
            diffopt.step(ce_loss)
            # outer level
            loss_v = torch.mean(torch.pow(fproxy(x0).squeeze() - y0.squeeze(), 2))
            grad = torch.autograd.grad(loss_v, labels)[0].data
            labels.data = labels.data - 0.1 * grad
    # use the soft label to update the model
    labels = labels.data
    labels = torch.clamp(labels, 0.0, 1.0)
    prob = torch.sigmoid(proxy(pairs[0, :]) - proxy(pairs[1, :])).squeeze()
    ce_loss = -torch.mean(labels * torch.log(prob + 1e-9) + (1 - labels) * torch.log(1 - prob + 1e-9))
    opt.zero_grad()
    ce_loss.backward()
    opt.step()


def compute_rank(pred):
    N = pred.shape[0]
    rank = torch.arange(N)
    rank[torch.argsort(-pred)] = torch.arange(N)
    return rank


def pair2vec(pairs, candidates):
    vec1 = candidates[pairs[0, :]].unsqueeze(0)
    vec2 = candidates[pairs[1, :]].unsqueeze(0)
    return torch.cat([vec1, vec2], dim=0)


def compute_tri_inv_pairs(rank1, rank2, rank3):
    N = rank1.shape[0]
    pairs1 = []
    pairs2 = []
    pairs3 = []
    for i in range(N):
        for j in range(i + 1, N):
            rank_diff1 = rank1[i] - rank1[j]
            rank_diff2 = rank2[i] - rank2[j]
            rank_diff3 = rank3[i] - rank3[j]
            # identify violating pairs for model1
            if (rank_diff2 * rank_diff3 > 0) and (rank_diff1 * rank_diff2 < 0):
                if rank_diff1 > 0:
                    pairs1.append([i, j])
                else:
                    pairs1.append([j, i])
            # identify violating pairs for model2
            if (rank_diff1 * rank_diff3 > 0) and (rank_diff2 * rank_diff1 < 0):
                if rank_diff2 > 0:
                    pairs2.append([i, j])
                else:
                    pairs2.append([j, i])
            # identify violating pairs for model3
            if (rank_diff1 * rank_diff2 > 0) and (rank_diff3 * rank_diff1 < 0):
                if rank_diff3 > 0:
                    pairs3.append([i, j])
                else:
                    pairs3.append([j, i])
    pairs1 = torch.Tensor(np.array(pairs1).T).long()
    pairs2 = torch.Tensor(np.array(pairs2).T).long()
    pairs3 = torch.Tensor(np.array(pairs3).T).long()
    return pairs1, pairs2, pairs3


def majority_voting_func(rank1, rank2, rank3):
    N = rank1.shape[0]
    pairs = []
    labels = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append([i, j])
            rank_diff1 = rank1[i] - rank1[j]
            rank_diff2 = rank2[i] - rank2[j]
            rank_diff3 = rank3[i] - rank3[j]
            if rank_diff1 * rank_diff2 > 0:
                if rank_diff1 > 0:
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                if rank_diff3 > 0:
                    labels.append(0)
                else:
                    labels.append(1)
    pairs = torch.Tensor(np.array(pairs).T).long()
    labels = torch.Tensor(np.array(labels))
    return pairs, labels


def rank2pairs(rank):
    N = rank.shape[0]
    pairs = []
    labels = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append([i, j])
            rank_diff = rank[i] - rank[j]
            if rank_diff > 0:
                labels.append(0)
            else:
                labels.append(1)
    pairs = torch.Tensor(np.array(pairs).T).long()
    labels = torch.Tensor(np.array(labels))
    return pairs, labels


def adjust_proxy(proxy1, proxy2, proxy3, candidate, x0, y0, K=10, majority_voting=True, soft_label=True):
    # compute neighbors
    candidate_neighbors = candidate.repeat(K, 1)
    candidate_neighbors = candidate_neighbors + 0.1 * \
                          torch.randn(candidate_neighbors.shape).to(candidate.device)
    # compute predictions & rankings
    pred1 = proxy1(candidate_neighbors).data.squeeze()
    pred2 = proxy2(candidate_neighbors).data.squeeze()
    pred3 = proxy3(candidate_neighbors).data.squeeze()
    # use rank to compute pair label
    rank1 = compute_rank(pred1)
    rank2 = compute_rank(pred2)
    rank3 = compute_rank(pred3)
    if majority_voting:
        pairs, labels = majority_voting_func(rank1, rank2, rank3)
    else:
        rank = compute_rank(pred1 + pred2 + pred3)
        pairs, labels = rank2pairs(rank)
    pairs = pair2vec(pairs, candidate_neighbors).to(candidate.device)
    labels = labels.to(candidate.device)

    # finetune proxy1
    pairs1, labels1 = rank2pairs(rank1)
    labels1 = labels1.to(candidate.device)
    train_index1 = torch.where(labels != labels1)[0]
    if train_index1.shape[0]:
        train_pairs1 = pairs[:, train_index1, :]
        train_labels1 = labels[train_index1]
        adjust_bpr(proxy1, train_pairs1, train_labels1, x0, y0, soft_label=soft_label)
    # finetune proxy2
    pairs2, labels2 = rank2pairs(rank2)
    labels2 = labels2.to(candidate.device)
    train_index2 = torch.where(labels != labels2)[0]
    if train_index2.shape[0]:
        train_pairs2 = pairs[:, train_index2, :]
        train_labels2 = labels[train_index2]
        adjust_bpr(proxy2, train_pairs2, train_labels2, x0, y0, soft_label=soft_label)
    # finetune proxy3
    pairs3, labels3 = rank2pairs(rank3)
    labels3 = labels3.to(candidate.device)
    train_index3 = torch.where(labels != labels3)[0]
    if train_index3.shape[0]:
        train_pairs3 = pairs[:, train_index3, :]
        train_labels3 = labels[train_index3]
        adjust_bpr(proxy3, train_pairs3, train_labels3, x0, y0, soft_label=soft_label)
