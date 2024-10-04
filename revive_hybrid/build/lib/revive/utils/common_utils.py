import ot
import os
import ray
import gym
import json
import h5py
import torch
import random
import urllib
import pickle
import argparse
import warnings
import importlib
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uuid import uuid1
from sklearn import tree
# from cairosvg import svg2pdf
from dtreeviz.trees import *
from tempfile import TemporaryDirectory
from PyPDF2 import PdfFileReader,PdfFileMerger
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from functools import partial
from torch.utils.data.dataloader import DataLoader
from typing import Any, Dict, List

import revive
from revive.computation.graph import DesicionGraph
from revive.computation.inference import *
from revive.computation.utils import *
from revive.computation.modules import *
from revive.data.batch import Batch
from revive.computation.funs_parser import parser

try:
    import cupy as cp
    CUPY_READY = True
except:
    # warnings.warn("Warning: CuPy is not installed, metric computing is going to be slow!")
    CUPY_READY = False


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def setup_seed(seed : int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def load_npz(filename : str):
    data = np.load(filename)
    return {k : v for k, v in data.items()}

def load_h5(filename : str):
    f = h5py.File(filename, 'r', libver="latest", swmr=True)
    data = {k : f[k][:] for k in f.keys()}
    f.close()
    return data

def save_h5(filename : str, data : Dict[str, np.ndarray]):
    with h5py.File(filename, 'w') as f:
        for k, v in data.items():
            f[k] = v

def npz2h5(npz_filename : str, h5_filename : str):
    data = load_npz(npz_filename)
    save_h5(h5_filename, data)

def h52npz(h5_filename : str, npz_filename : str):
    data = load_h5(h5_filename)
    np.savez_compressed(npz_filename, **data)

def load_data(data_file : str):
    if data_file.endswith('.h5'):
        raw_data = load_h5(data_file)
    elif data_file.endswith('.npz'):
        raw_data = load_npz(data_file)
    else:
        raise ValueError(f'Try to load {data_file}, but get unknown data format!')
    return raw_data

def find_policy_index(graph : DesicionGraph, policy_name : str):
    for i, k in enumerate(graph.keys()):
        if k == policy_name:
            break
    return i

def load_policy(filename : str, policy_name : str = None):
    try:
        model = torch.load(filename, map_location='cpu')
    except:
        with open(filename, 'rb') as f:
            model = pickle.load(f)

    if isinstance(model, VirtualEnv):
        model = model._env

    if isinstance(model, VirtualEnvDev):
        node = model.graph.get_node(policy_name)
        model = PolicyModelDev(node)

    if isinstance(model, PolicyModelDev):
        model = PolicyModel(model)

    return model
    
def download_helper(url : str, filename : str):
    'Download file from given url. Modified from `torchvision.dataset.utils`'
    def gen_bar_updater():
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    try:
        print('Downloading ' + url + ' to ' + filename)
        urllib.request.urlretrieve(
            url, filename,
            reporthook=gen_bar_updater()
        )
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + filename)
            urllib.request.urlretrieve(
                url, filename,
                reporthook=gen_bar_updater()
            )
        else:
            raise e

def import_model_from_file(file_path, module_name = "module.name"):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    
    return foo
            
def get_reward_fn(reward_file_path,config_file):
    if reward_file_path:
        logger.info(f'import reward function from {reward_file_path}!')
        # parse function
        reward_file_path_parsed = reward_file_path[:-3]+"_parsed.py"
        if not parser(reward_file_path,reward_file_path_parsed,config_file):
            reward_file_path_parsed = reward_file_path
        source_file = import_model_from_file(reward_file_path_parsed)
        try:
            reward_func = source_file.reward
        except:
            reward_func = source_file.get_reward
    else:
        logger.info(f'No reward function is defined!')
        reward_func = None
        
    return reward_func

def create_env(task : str):
    try:
        if task in ["HalfCheetah-v3", "Hopper-v3", "Walker2d-v3", "ib", "finance", "citylearn"]:
            import neorl
            env = neorl.make(task)
        elif task in ['halfcheetah-meidum-v0', 'hopper-medium-v0', 'walker2d-medium-v0']:
            import d4rl
            env = gym.make(task)
        else:
            env = gym.make(task)
    except:
        warnings.warn(f'Warning: task {task} can not be created!')
        env = None

    return env

def test_one_trail(env : gym.Env, policy : PolicyModel):
    env = deepcopy(env)
    policy = deepcopy(policy)

    obs = env.reset()
    reward = 0
    length = 0
    while True:
        action = policy.infer({'obs' : obs[np.newaxis]})[0]
        obs, r, done, info = env.step(action)
        reward += r
        length += 1

        if done:
            break

    return (reward, length)

def test_on_real_env(env : gym.Env, policy : PolicyModel, number_of_runs : int = 10):
    rewards = []
    episode_lengths = []
    test_func = ray.remote(test_one_trail)

    results = ray.get([test_func.remote(env, policy) for _ in range(number_of_runs)])
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]

    return np.mean(rewards), np.mean(episode_lengths)

def get_input_dim_from_graph(graph : DesicionGraph, 
                             output_name : str, 
                             total_dims : dict):
    '''return the total number of dims used to compute the given node on the graph'''
    input_names = graph[output_name]
    input_dim = 0
    for input_name in input_names:
        input_dim += total_dims[input_name]['input']
    return input_dim

def normalize(data):
    flatten_data = data.reshape((-1, data.shape[-1]))
    mean = flatten_data.mean(axis=0)
    std = flatten_data.std(axis=0)
    std[np.isclose(std, 0)] = 1
    data = (data - mean) / std
    return data   

def plot_traj(traj):
    traj = np.concatenate([*traj.values()], axis=-1)
    max_value = traj.max(axis=0)
    min_value = traj.min(axis=0)
    interval = max_value - min_value
    interval[interval == 0] = 1
    traj = (traj - min_value) / interval
    plt.imshow(traj)
    plt.show()

def check_weight(network : torch.nn.Module):
    """
    Check whether network parameters are nan or inf.
    """
    for k, v in network.state_dict().items():
        if torch.any(torch.isnan(v)):
            print(k + 'has nan')
        if torch.any(torch.isinf(v)):
            print(k + 'has inf')        

def get_models_parameters(*models):
    """return all the parameters of input models in a list"""
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    return parameters

def get_grad_norm(parameters, norm_type : float=2):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type)for p in parameters]), norm_type)
    return total_norm

def get_concat_traj(batch_data : Batch, node_names : List[str]):
    return torch.cat(get_list_traj(batch_data, node_names), dim=-1)

def get_list_traj(batch_data : Batch, node_names : List[str], nodes_fit_index: dict = None) -> list:
    datas = []

    for name in node_names:
        if nodes_fit_index:
            datas.append(batch_data[name][...,nodes_fit_index[name]])
        else:
            datas.append(batch_data[name])

    return datas   

def generate_rewards(traj : Batch, reward_fn):
    """
    Add rewards for batch trajectories.
    :param traj: batch trajectories.
    :param reward_fn: how the rewards generate.
    :return: batch trajectories with rewards.
    """
    head_shape = traj.shape[:-1]
    traj.reward = reward_fn(traj).view(*head_shape, 1)
    return traj

def generate_rollout(expert_data : Batch, 
                     graph : DesicionGraph, 
                     traj_length : int, 
                     sample_fn=lambda dist: dist.sample(), 
                     adapt_stds=None,
                     clip : Union[bool, float] = False,
                     use_target : bool = False):
    """
    Generate trajectories based on current policy.
    :param expert_data: samples from the dataset.
    :param graph: the computation graph
    :param traj_length: trajectory length
    :param sample_fn: sample from a distribution.
    :return: batch trajectories.
    NOTE: this function will mantain the last dimension even if it is 1
    """

    assert traj_length <= expert_data.shape[0], 'cannot generate trajectory beyond expert data'

    expert_data = deepcopy(expert_data)
    if adapt_stds is None:
        adapt_stds = [None] * (len(graph))

    graph.reset()
    generated_data = []
    current_batch = expert_data[0]

    for i in range(traj_length):
        for node_name, adapt_std in zip(list(graph.keys()), adapt_stds):
            if graph.get_node(node_name).node_type == 'network':
                action_dist = graph.compute_node(node_name, current_batch, adapt_std=adapt_std, use_target=use_target)
                # action = sample_fn(action_dist)
                action = action_dist.mode
                if isinstance(clip, bool) and clip:
                    action = torch.clamp(action, -1, 1)
                elif isinstance(clip, float):
                    action = torch.clamp(action, -clip, clip)
                else:
                    pass
                current_batch[node_name] = action
                action_log_prob = action_dist.log_prob(action).unsqueeze(dim=-1).detach() # TODO: do we need this detach?
                current_batch[node_name + "_log_prob"] = action_log_prob
            else:
                action = graph.compute_node(node_name, current_batch)
                current_batch[node_name] = action

        # check the generated current_batch
        # NOTE: this will make the rollout a bit slower.
        #       Remove it if you are sure no explosion will happend.
        for k, v in current_batch.items():
            has_inf = torch.any(torch.isinf(v))
            has_nan = torch.any(torch.isnan(v))
            if has_inf or has_nan:
                logger.warning(f'During rollout detect anomaly data: key {k}, has inf {has_inf}, has nan {has_nan}')
                logger.warning(f'Should generated rollout with length {traj_length}, early stop for only length {i}')
                break

        generated_data.append(current_batch)

        if i == traj_length - 1 : break
        current_batch = expert_data[i+1] # clone to new Batch
        current_batch.update(graph.state_transition(generated_data[-1]))

    generated_data = Batch.stack(generated_data)

    return generated_data

def compute_lambda_return(rewards, values, bootstrap=None, _gamma=0.9, _lambda=0.98):
    next_values = values[1:]
    if bootstrap is None:
        bootstrap = torch.zeros_like(values[-1])

    next_values = torch.cat([next_values, bootstrap.unsqueeze(0)], dim=0)

    g = [rewards[i] + _gamma * (1 - _lambda) * next_values[i] for i in range(rewards.shape[0])]

    lambda_returns = []
    last = next_values[-1]
    for i in reversed(list(range(len(rewards)))):
        last = g[i] + _gamma * _lambda * last
        lambda_returns.append(last)

    return torch.stack(list(reversed(lambda_returns)))

def sinkhorn_gpu(cuda_id):
    cp.cuda.Device(cuda_id).use()
    import ot.gpu
    return ot.gpu.sinkhorn

def wasserstein_distance(X, Y, cost_matrix, method='sinkhorn', niter=50000, cuda_id=0):
    if method == 'sinkhorn_gpu':
        sinkhorn_fn = sinkhorn_gpu(cuda_id)
        transport_plan = sinkhorn_fn(X, Y, cost_matrix, reg=1, enumItermax=niter)    # (GPU) Get the transport plan for regularized OT
    elif method  == 'sinkhorn':
        transport_plan = ot.sinkhorn(X, Y, cost_matrix, reg=1, numItermax=niter)    # (CPU) Get the transport plan for regularized OT
    elif method  == 'emd':
        transport_plan = ot.emd(X, Y, cost_matrix, numItermax=niter)    # (CPU) Get the transport plan for OT with no regularisation
    elif method  == 'emd2':
        distance = ot.emd2(X, Y, cost_matrix)  # (CPU) Get the transport loss
        return distance
    else:
        raise NotImplementedError("The method is not implemented!")

    distance = np.sum(np.diag(np.matmul(transport_plan, cost_matrix.T)))  # Calculate Wasserstein by summing diagonals, i.e., W=Trace[MC^T]

    return distance

def compute_w2_dist_to_expert(policy_trajectorys, 
                              expert_trajectorys, 
                              scaler=None, 
                              data_is_standardscaler=False,
                              max_expert_sampes=20000,
                              dist_metric = "euclidean",
                              emd_method="emd",
                              processes=None,
                              use_cuda=False,
                              cuda_id_list=None):
    """Computes Wasserstein 2 distance to expert demonstrations."""
    policy_trajectorys = policy_trajectorys.copy()
    expert_trajectorys = expert_trajectorys.reshape(-1,expert_trajectorys.shape[-1]).copy()
    policy_trajectorys_shape = policy_trajectorys.shape
    policy_trajectorys = policy_trajectorys.reshape(-1,policy_trajectorys_shape[-1])

    expert_trajectorys_index = np.arange(expert_trajectorys.shape[0])
    if expert_trajectorys.shape[0] < max_expert_sampes:
        max_expert_sampes = expert_trajectorys.shape[0]

    if not data_is_standardscaler:
        if scaler is None:
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            scaler.fit(expert_trajectorys)
        
        policy_trajectorys = scaler.transform(policy_trajectorys)
        expert_trajectorys = scaler.transform(expert_trajectorys)
    policy_trajectorys = policy_trajectorys.reshape(policy_trajectorys_shape)

    expert_trajectory_weights = 1./max_expert_sampes * np.ones(max_expert_sampes)
    policy_trajectory_weights = 1./policy_trajectorys_shape[1] * np.ones(policy_trajectorys_shape[1])

    if not CUPY_READY: emd_method = 'emd' # fallback to cpu mode

    w2_dist_list = []
    if use_cuda and "gpu" in emd_method:
        if cuda_id_list is None:
            cuda_id_list = list(range(cp.cuda.runtime.getDeviceCount()))
        assert len(cuda_id_list) > 0
        
        for i, policy_trajectory in enumerate(policy_trajectorys):
            cuda_id = cuda_id_list[i%len(cuda_id_list)]
            cost_matrix = ot.dist(policy_trajectory, expert_trajectorys[expert_trajectorys_index[:max_expert_sampes]], metric=dist_metric)
            w2_dist_list.append(wasserstein_distance(policy_trajectory_weights, expert_trajectory_weights, cost_matrix, emd_method, cuda_id))
    else:
        pool = multiprocessing.Pool(processes = processes if processes is not None else multiprocessing.cpu_count())
        for policy_trajectory in policy_trajectorys:
            cost_matrix = ot.dist(policy_trajectory, expert_trajectorys[expert_trajectorys_index[:max_expert_sampes]], metric=dist_metric)
            np.random.shuffle(expert_trajectorys_index)
            w2_dist_list.append(pool.apply_async(wasserstein_distance, (policy_trajectory_weights, expert_trajectory_weights, cost_matrix, emd_method)))
        pool.close()
        pool.join()
        w2_dist_list = [res.get() for res in w2_dist_list]

    return np.mean(w2_dist_list)

def dict2parser(config : dict):
    parser = argparse.ArgumentParser()

    def get_type(value):
        if type(value) is bool:
            return lambda x: [False, True][int(x)]
        return type(value)

    for k, v in config.items():
        parser.add_argument(f'--{k}', type=get_type(v), default=v)

    return parser

def list2parser(config : List[Dict]):
    parser = argparse.ArgumentParser()

    def get_type(type_name):
        type_name = eval(type_name) if isinstance(type_name, str) else type_name
        if type_name is bool:
            return lambda x: [False, True][int(x)]
        return type_name

    for d in config:
        names = ['--' + d['name']]
        data_type = get_type(d['type'])
        default_value = d['default']
        addition_args = {}
        if data_type is list:
            data_type = get_type(type(default_value[0])) if type(default_value) is list else get_type(type(default_value))
            addition_args['nargs'] = '+'
        if 'abbreviation' in d.keys(): names.append('-' + d['abbreviation'])
        parser.add_argument(*names, type=data_type, default=default_value, help=d.get('description', ''), **addition_args)

    return parser

def set_parameter_value(config : List[Dict], name : str, value : Any):
    for param in config:
        if param['name'] == name:
            param['default'] = value
            break
    return config

def update_description(default_description, custom_description):
    '''update in-place the default description with a custom description.'''
    names_to_indexes = {description['name'] : i for i, description in enumerate(default_description)}
    for description in custom_description:
        name = description['name']
        index = names_to_indexes.get(name, None)
        if index == None:
            warnings.warn(f'parameter name `{name}` is not in the default description, skip.')
        else:
            default_description[index] = description 

def find_later(path : str, keyword : str) -> List[str]:
    ''' find all the later folder after the given keyword '''
    later = []
    while len(path) > 0:
        path, word = os.path.split(path)
        later.append(word)
        if keyword == word:
            break
    return list(reversed(later))

def get_node_dim_from_dist_configs(dist_configs, node_name):
    node_dim = 0
    for dist_config in dist_configs[node_name]:
        node_dim += dist_config["dim"]

    return node_dim

def save_histogram(histogram_path : str, graph : DesicionGraph, data_loader : DataLoader, device : str, scope : str):
    '''save the histogram'''
    processor = graph.processor
    
    expert_data = []
    generated_data = []
    for expert_batch in iter(data_loader):
        traj_length = expert_batch.shape[0]
        expert_batch.to_torch(device=device)
        generated_batch = generate_rollout(expert_batch, graph, traj_length, lambda dist: dist.mode, clip=True)
        expert_batch.to_numpy()
        generated_batch.to_numpy()
        expert_data.append(expert_batch)
        generated_data.append(generated_batch)
    
    expert_data = {node_name : np.concatenate([batch[node_name] for batch in expert_data], axis=1)  for node_name in graph.keys()}
    generated_data = {node_name : np.concatenate([batch[node_name] for batch in generated_data], axis=1)  for node_name in graph.keys()}

    expert_data = processor.deprocess(expert_data)
    generated_data = processor.deprocess(generated_data)

    fig = plt.figure(figsize=(15, 7), dpi=150)
    for node_name in graph.keys():
        index_name = node_name[5:] if node_name in graph.transition_map.values() else node_name
        for i, dimension in enumerate(graph.descriptions[index_name]):
            dimension_name = list(dimension.keys())[0]
            expert_dimension_data = expert_data[node_name][..., i].reshape((-1))
            generated_dimension_data = generated_data[node_name][..., i].reshape((-1))
            assert expert_dimension_data.shape == generated_dimension_data.shape

            if dimension[dimension_name]['type'] == 'continuous':
                bins = 100
            elif dimension[dimension_name]['type'] == 'discrete':
                bins = min(dimension[dimension_name]['num'], 100)
            else:
                bins = None
                
            title = f'{node_name}.{dimension_name}'
            plt.hist([expert_dimension_data, generated_dimension_data], bins=bins, label=['History_Data', 'Generated_Data'], log=True)
            plt.legend(loc='upper left')
            plt.xlabel(title)
            plt.ylabel("frequency")
            plt.title(title + f"-histogram-{scope}")
            plt.savefig(os.path.join(histogram_path, title + f"-{scope}.png"))
            fig.clf()
    plt.close(fig)

def save_histogram_after_stop(traj_length : int, traj_dir : str, train_dataset, val_dataset):
    '''save the histogram after the training is stopped'''
    histogram_path = os.path.join(traj_dir, 'histogram')
    if not os.path.exists(histogram_path):
        os.makedirs(histogram_path)

    from revive.data.dataset import collect_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_graph = torch.load(os.path.join(traj_dir, 'venv_train.pt'), map_location=device).graph
    train_dataset = train_dataset.trajectory_mode_(traj_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                               collate_fn=partial(collect_data, graph=train_dataset.graph), pin_memory=True)
    save_histogram(histogram_path, train_graph, train_loader, device=device, scope='train')

    val_graph = torch.load(os.path.join(traj_dir, 'venv_val.pt'), map_location=device).graph
    val_dataset = val_dataset.trajectory_mode_(traj_length)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                             collate_fn=partial(collect_data, graph=val_dataset.graph), pin_memory=True)
    save_histogram(histogram_path, val_graph, val_loader, device=device, scope='val')


def tb_data_parse(tensorboard_log_dir, keys: list = []):
    ''' parse data from tensorboard logdir '''
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(tensorboard_log_dir)
    ea.Reload()
    ea_keys = ea.scalars.Keys()

    ea_keys = [k[9:] if k.startswith('ray/tune/') else k for k in ea_keys]
    parse_data = lambda key: [(i.step,i.value) for i in ea.scalars.Items(key)] 
    
    if keys:
        if set(keys) < set(ea_keys):
            logger.info(f"Keys Error: there are some keys not in tensorboard logs!")
        res = { key:parse_data(key) for key in keys }
    else:
        res = { key:parse_data(key) for key in ea_keys }
        
    return res


def double_venv_validation(reward_logs, data_reward={}, img_save_path=""):
    reward_trainPolicy_on_trainEnv = np.array(reward_logs["reward_trainPolicy_on_trainEnv"])
    reward_valPolicy_on_trainEnv = np.array(reward_logs["reward_valPolicy_on_trainEnv"])
    reward_valPolicy_on_valEnv = np.array(reward_logs["reward_valPolicy_on_valEnv"])
    reward_trainPolicy_on_valEnv = np.array(reward_logs["reward_trainPolicy_on_valEnv"])
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Double Venv Validation", fontsize=26)
    
    x = np.arange(reward_trainPolicy_on_trainEnv[:,0].shape[0])

    axs[0].plot(x,reward_trainPolicy_on_trainEnv[:,1], 'r--', label='reward_trainPolicy_on_trainEnv')
    axs[0].plot(x,reward_valPolicy_on_trainEnv[:,1], 'g--', label='reward_valPolicy_on_trainEnv')
    if "reward_train" in data_reward.keys():
        axs[0].plot(x,np.ones_like(reward_trainPolicy_on_trainEnv[:,0]) * data_reward["reward_train"], 'b--', label='reward_train')
        
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    
    axs[1].plot(np.arange(reward_valPolicy_on_valEnv[:,1].shape[0]),reward_valPolicy_on_valEnv[:,1], 'r--', label='reward_valPolicy_on_valEnv')
    axs[1].plot(np.arange(reward_trainPolicy_on_valEnv[:,1].shape[0]),reward_trainPolicy_on_valEnv[:,1], 'g--', label='reward_trainPolicy_on_valEnv')
    if "reward_val" in data_reward.keys():
        axs[1].plot(np.arange(reward_valPolicy_on_valEnv[:,1].shape[0]),np.ones_like(reward_valPolicy_on_valEnv[:,0]) * data_reward["reward_val"], 'b--', label='reward_val')
        
    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Epoch') 
    axs[1].legend()

    fig.savefig(img_save_path)
    plt.close(fig)

def plt_double_venv_validation(tensorboard_log_dir, reward_train, reward_val, img_save_path):
    ''' Drawing double_venv_validation images '''
    reward_logs = tb_data_parse(tensorboard_log_dir, ['reward_trainPolicy_on_valEnv', 'reward_trainPolicy_on_trainEnv', 'reward_valPolicy_on_trainEnv', 'reward_valPolicy_on_valEnv'])
    data_reward = {"reward_train" : reward_train, "reward_val" : reward_val}
    double_venv_validation(reward_logs, data_reward, img_save_path)

def _plt_node_rollout(expert_datas, generated_datas, node_name, data_dims, img_save_dir):
    sub_fig_num = len(data_dims)
    for trj_index, (expert_data, generated_data) in enumerate(zip(expert_datas, generated_datas)):
        img_save_path = os.path.join(img_save_dir,f"{trj_index}_{node_name}")
        if sub_fig_num > 1:
            fig, axs = plt.subplots(sub_fig_num, 1, figsize=(15, 5*sub_fig_num))
            fig.suptitle("Policy Rollout", fontsize=26)
            for index,dim in enumerate(data_dims):
                axs[index].plot(expert_data[:,index], 'r--', label='History Expert Data')
                axs[index].plot(generated_data[:,index], 'g--', label='Policy Rollout Data')

                axs[index].set_ylabel(dim)
                axs[index].set_xlabel('Step')
                axs[index].legend()
            fig.savefig(img_save_path)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(15, 5))
            plt.plot(expert_data, 'r--', label='History Expert Data')
            plt.plot(generated_data, 'g--', label='Policy Rollout Data')

            plt.ylabel(data_dims[0])
            plt.xlabel('Step')
            plt.title("Policy Rollout")
            plt.legend()
            plt.savefig(img_save_path)
            plt.close(fig)

def save_rollout_action(rollout_save_path: str,
                        graph: DesicionGraph, 
                        device: str, 
                        dataset, 
                        nodes,
                        horizion_num = 10):
    '''save the Trj rollout'''
    if not os.path.exists(rollout_save_path):
        os.makedirs(rollout_save_path)

    graph = graph.to(device)

    from revive.data.dataset import collect_data

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=horizion_num, 
                                              shuffle=True,
                                              collate_fn=partial(collect_data, graph=graph), 
                                              pin_memory=True)

    processor = graph.processor
    expert_data = []
    generated_data = []

    for expert_batch in iter(data_loader):
        traj_length = expert_batch.shape[0]
        expert_batch.to_torch(device=device)
        generated_batch = generate_rollout(expert_batch, graph, traj_length, lambda dist: dist.mode, clip=True)
        expert_batch.to_numpy()
        generated_batch.to_numpy()
        expert_data.append(expert_batch)
        generated_data.append(generated_batch)
        break
    
    expert_data = {node_name : np.concatenate([batch[node_name] for batch in expert_data], axis=1)  for node_name in nodes.keys()}
    generated_data = {node_name : np.concatenate([batch[node_name] for batch in generated_data], axis=1)  for node_name in nodes.keys()}

    expert_data = processor.deprocess(expert_data)
    generated_data = processor.deprocess(generated_data)

    #select_indexs = np.random.choice(np.arange(expert_data[list(nodes.keys())[0]].shape[1]), size=10, replace=False)

    for node_name,node_dims in nodes.items():
        horizion_num = min(horizion_num, expert_data.shape[1])
        expert_action_data = expert_data[node_name]
        generated_action_data = generated_data[node_name]
        select_expert_action_data = [expert_action_data[:,index] for index in range(horizion_num)]
        select_generated_action_data = [generated_action_data[:,index] for index in range(horizion_num)]

        node_rollout_save_path = os.path.join(rollout_save_path, node_name)
        if not os.path.exists(node_rollout_save_path):
            os.makedirs(node_rollout_save_path)

        _plt_node_rollout(select_expert_action_data, select_generated_action_data, node_name, node_dims, node_rollout_save_path)


def data_to_dtreeviz(data: pd.DataFrame,
                     target: pd.DataFrame,
                     target_type: (List[str],str),
                     orientation: ('TD', 'LR') = "TD",
                     fancy: bool = False,
                     max_depth: int = 3,
                     output: (str) = None):
    if isinstance(target_type, str) and len(target.columns) > 1:
        target_type = [target_type,] * len(target.columns)
    
    _tmp_pdf_paths = []

    with TemporaryDirectory() as dirname:
        for _target_type, target_name in zip(target_type,target.columns):
            if _target_type == "Classifier" or _target_type == "C":
                _target_type = "Classifier"
                decisiontree = tree.DecisionTreeClassifier(max_depth=max_depth)
            elif _target_type == "Regressor" or _target_type == "R":
                _target_type = "Regressor" 
                decisiontree = tree.DecisionTreeRegressor(max_depth=max_depth,random_state=1)
            else:
                raise NotImplementedError
                
            _target = target[[target_name,]]
            decisiontree.fit(data,_target)

            if _target_type == "Classifier":
                _orientation = "TD"
                _target = _target.values.reshape(-1)
                class_names = list(set(list(_target)))
            else:
                _target = _target.values
                class_names = None
                _orientation = "LR"

            viz = dtreeviz(decisiontree,
                        data.values,
                        _target,
                        target_name=target_name,
                        feature_names=data.columns,
                        class_names=class_names,
                        title = target_name + " " + _target_type + " Tree",
                        orientation = _orientation,
                        fancy=fancy,
                        scale = 2)

            _tmp_pdf_path = os.path.join(dirname, str(uuid1())+".pdf")
            _tmp_pdf_paths.append(_tmp_pdf_path)
            svg2pdf(url=viz.save_svg(), 
                    output_width=800, 
                    output_height=1000, 
                    write_to=_tmp_pdf_path)

        
        if output is None:
            if len(vizs.keys()) == 1:
                return viz
            return vizs

        assert output, f"output should be not None" 

        if len(_tmp_pdf_paths)==len(target_type):
            merger = PdfFileMerger()

            for in_pdf in _tmp_pdf_paths:
                with open(in_pdf,'rb') as pdf:
                    merger.append(PdfFileReader(pdf))
            merger.write(output)

        return

def net_to_tree(tree_save_path: str,
                graph: DesicionGraph, 
                device: str, 
                dataset, 
                nodes):

    if not os.path.exists(tree_save_path):
        os.makedirs(tree_save_path)

    graph = graph.to(device)

    from revive.data.dataset import collect_data

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=256, 
                                              shuffle=True,
                                              collate_fn=partial(collect_data, graph=graph), 
                                              pin_memory=True)

    processor = graph.processor
    expert_data = []
    generated_data = []
    data_num = 0
    for expert_batch in iter(data_loader):
        traj_length = expert_batch.shape[0]
        expert_batch.to_torch(device=device)
        generated_batch = generate_rollout(expert_batch, graph, traj_length, lambda dist: dist.mode, clip=True)
        expert_batch.to_numpy()
        generated_batch.to_numpy()
        expert_data.append(expert_batch)
        generated_data.append(generated_batch)

        data_num += traj_length*256
        if data_num> 20000:
            break
    data_keys = list(graph.keys()) + graph.leaf
    expert_data = {node_name : np.concatenate([batch[node_name] for batch in expert_data], axis=1)  for node_name in data_keys}
    generated_data = {node_name : np.concatenate([batch[node_name] for batch in generated_data], axis=1)  for node_name in data_keys}

    expert_data = {node_name:node_data.reshape(-1,node_data.shape[-1]) for node_name, node_data in expert_data.items()}
    generated_data = {node_name:node_data.reshape(-1,node_data.shape[-1]) for node_name, node_data in generated_data.items()}
    expert_data = processor.deprocess(expert_data)
    generated_data = processor.deprocess(generated_data)

    sample_num = expert_data[list(nodes.keys())[0]].shape[0]
    size = min(sample_num, 20000)
    select_indexs = np.random.choice(np.arange(sample_num), size=size, replace=False)

    for output_node,output_node_dims in nodes.items():
        input_node_dims = []
        input_nodes = graph[output_node]
        for input_node in input_nodes:
            for obs_dim in graph.descriptions[input_node]:
                input_node_dims.append(list(obs_dim.keys())[0])

        input_data = np.concatenate([generated_data[node] for node in input_nodes], axis=-1)
        output_data = generated_data[output_node]

        input_data = input_data[select_indexs]
        output_data = output_data[select_indexs]

        X = pd.DataFrame(input_data, columns=input_node_dims) 
        Y = pd.DataFrame(output_data, columns=output_node_dims) 
        Y_type = "R"
        result = data_to_dtreeviz(X,Y,Y_type, output=os.path.join(tree_save_path,output_node+".pdf"))
