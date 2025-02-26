''''''
"""
    POLIXIR REVIVE, copyright (C) 2021-2022 Polixir Technologies Co., Ltd., is 
    distributed under the GNU Lesser General Public License (GNU LGPL). 
    POLIXIR REVIVE is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
"""

from copy import deepcopy
import torch
from revive.utils.common_utils import set_parameter_value


DEFAULT_CONFIG = [
    {
        'name' : 'global_seed',
        'abbreviation' : 'gs',
        'description' : 'Set the random number seed for the experiment.',
        'type' : int,
        'default' : 42,
        'doc': True,
    },

    # data related config
    {
        'name' : 'val_split_ratio',
        'abbreviation' : 'vsr',
        'description' : 'Ratio to split validate dataset if it is not explicitly given.',
        'type' : float,
        'default' : 0.5,
        'doc': True,
    },
    {
        'name' : 'val_split_mode',
        'abbreviation' : 'vsm',
        'description' : 'Mode of auto splitting training and validation dataset, choose from `outside_traj` and `inside_traj`. ' + 
            '`outside_traj` means the split is happened outside the trajectories, one trajectory can only be in one dataset. ' +
            '`inside_traj` means the split is happened inside the trajectories, former part of one trajectory is in training set, later part is in validation set.',
        'type' : str,
        'default' : 'outside_traj',
        'doc': True,
    },
    {
        'name' : 'ignore_check',
        'abbreviation' : 'igc',
        'description' : 'Flag to ignore data related check, force training.',
        'type' : bool,
        'default' : False,
        'doc': True,
    },
    {
        'name' : 'data_workers',
        'abbreviation' : 'dw',
        'description' : 'Number of workers to load data.',
        'type' : int,
        'default' : 0,
    },   
    {
        'name' : 'continuous_distribution_type',
        'abbreviation' : 'codt',
        'description' : 'type of distribution used to model continuous variables, choose from `normal` and `gmm`.',
        'type' : str,
        'default' : 'normal',
    },   
    {
        'name' : 'discrete_distribution_type',
        'abbreviation' : 'ddt',
        'description' : 'type of distribution used to model discrete variables, choose from `discrete_logistic`, `normal` and `gmm`.',
        'type' : str,
        'default' : 'discrete_logistic',
    },   
    {
        'name' : 'category_distribution_type',
        'abbreviation' : 'cadt',
        'description' : 'type of distribution used to model category variables, currently only support `onehot`',
        'type' : str,
        'default' : 'onehot',
    },   
    {
        'name' : 'conditioned_std',
        'description' : 'Whether the standard deviation is conditioned on the inputs.',
        'type' : bool,
        'default' : True,
    },   
    {
        'name' : 'mixture',
        'description' : 'Number of mixtures if use gmm as distribution type.',
        'type' : int,
        'default' : 5,
    }, 
    {
        "name": "task",
        "description": "Name of the task, if the task is a simulator.",
        "type": str,
        "default": None,
        'tune': False,
    },

    # venv related config
    {
        'name' : 'venv_rollout_horizon',
        'abbreviation' : 'vrh',
        'description' : 'Length of sampled trajectory, validate only if the algorithm works on sequential data.',
        'type' : int,
        'default' : 10,
        'doc': True,
    },
    {
        'name' : 'transition_mode',
        'abbreviation' : 'tsm',
        'description' : 'Mode of transition, choose from `global` and `local`.',
        'type' : str,
        'default' : 'global',
        'doc': True,
    }, 
    {
        'name' : 'venv_gpus_per_worker',
        'abbreviation' : 'vgpw',
        'description' : 'Number of gpus per worker in venv training, small than 1 means launch multiple workers on the same gpu.',
        'type' : float,
        'default' : 1.0,
        'doc': True,
    },  
    {
        'name' : 'venv_metric',
        'description' : 'Metric used to evaluate the trained venv, choose from `nll`, `mae`, `mse`, `wdist`.',
        'type' : str,
        'default' : 'mae',
        'doc': True,
    },  
    {
        'name' : 'venv_algo',
        'description' : 'Algorithm used in venv training. There are currently three algorithms to choose from, `bc` , `revive_p` and `revive_t`.',
        'type' : str,
        'default' : 'revive_p',
        'doc': True,
    },  
    {
        'name' : 'save_start_epoch',
        'abbreviation' : 'sse',
        'description' : 'We only save models after this epoch, default is 0 which means we save models from the beginning.',
        'type' : int,
        'default' : 0,
    },  
    {
        'name' : 'num_venv_store',
        'abbreviation' : 'nvs',
        'description' : 'Max number of the chosen venv among the process of hyper-parameter search.',
        'type' : int,
        'default' : 5,
    },  
    {
        'name' : 'nll_test',
        'description' : 'Whether to perform nll test during training, can be overwrite by `venv_metric`.',
        'type' : bool,
        'default' : True,
    },  
    {
        'name' : 'mae_test',
        'description' : 'Whether to perform mae test during training, can be overwrite by `venv_metric`.',
        'type' : bool,
        'default' : True,
    },  
    {
        'name' : 'mse_test',
        'description' : 'Whether to perform mse test during training, can be overwrite by `venv_metric`.',
        'type' : bool,
        'default' : True,
    },  
    {
        'name' : 'wdist_test',
        'description' : 'Whether to perform Wasserstein Distance test during training, can be overwrite by `venv_metric`.',
        'type' : bool,
        'default' : True,
    }, 
    {
        'name' : 'save_by_node',
        'description' : 'Whether to save the mechanism of the trail model based on the performance of a single node.',
        'type' : bool,
        'default' : False,
    },   
    {
        'name' : 'histogram_log_frequency',
        'abbreviation' : 'hlf',
        'description' : 'How many steps between two histogram summary. 0 means disable.',
        'type' : int,
        'default' : 0,
    },  

    # policy related config
    {
        "name": "policy_gpus_per_worker",
        'abbreviation' : 'pgpw',
        "description": "Number of gpus per worker in venv training, small than 1 means launch multiple workers on the same gpu.",
        "type": float,
        "default": 1.0,
        'doc': True,
    },
    {
        "name": "num_venv_in_use",
        'abbreviation' : 'nviu',
        "description": "Max number of venvs used in policy training, clipped when there is no enough venvs available.",
        "type": float,
        "default": 1,
        'doc': True,
    },
    {
        "name": "behavioral_policy_init",
        'abbreviation' : 'bpi',
        "description": "Whether to use the learned behavioral policy to as the initialization policy training.",
        "type": bool,
        "default": True,
        'doc': True,
    },
    {
        "name": "policy_algo",
        "description": 'Algorithm used in policy training. There are currently two algorithms to choose from, `ppo` and `sac`.',
        "type": str,
        "default": "ppo",
    },
    {
        "name": "test_gamma",
        "description": "Gamma used in venv test.",
        "type": float,
        "default": 1.0,
    },
    {
        "name": "test_horizon",
        'abbreviation' : 'th',
        "description": "Rollout length of the venv test.",
        "type": int,
        "default": 10,
        'doc': True,
    },
    {
        "name": "deterministic_test",
        'abbreviation' : 'dett',
        "description": "Whether to use deterministic rollout in venv test.",
        "type": bool,
        "default": True,
    },
    {
        "name": "policy_double_validation",
        'abbreviation' : 'pdv',
        "description": "Whether to enable double validation in policy training.",
        "type": bool,
        "default": True,
    },
    {
        "name": "real_env_test_frequency",
        'abbreviation' : 'retf',
        "description": "How many steps between two real env test. 0 means disable.",
        "type": int,
        "default": 0,
    },
    {
        "name": "fqe_test_frequency",
        'abbreviation' : 'ftf',
        "description": "How many steps between two fqe test.",
        "type": int,
        "default": 25000000000,
    },

    # parameter tuning related config
    {
        "name": "parameter_tuning_algorithm",
        'abbreviation' : 'pta',
        "description": "Algorithm for tuning parameter, support `random` and `zoopt`.",
        "type": str,
        "default": 'zoopt',
    },
    {
        "name": "parameter_tuning_budget",
        'abbreviation' : 'ptb',
        "description": "Total trails searched by tuning algorithm.",
        "type": int,
        "default": 10000,
    },
    {
        "name": "parameter_tuning_rollout_horizon",
        'abbreviation' : 'ptrh',
        "description": "Rollout horzion when testing parameters, can be overwrote by the data you provide when leaf node on the graph.",
        "type": int,
        "default": 1800,
    },

    # training related config
    {
        'name' : 'use_gpu',
        'description' : 'Whether to use gpu during training.',
        'type' : bool,
        'default' : torch.cuda.is_available(),
    },  
    {
        'name' : 'use_fp16',
        'description' : 'Whether to use mix precision training to speed up the training process, need `apex` being installed.',
        'type' : bool,
        'default' : False,
    },  

    # tune config
    {
        'name' : 'parallel_num',
        'abbreviation' : 'pan',
        'description' : '[zoopt] Number of trails searched simultaneously.',
        'type' : str,
        'default' : 'auto',
    },  
    {
        'name' : 'workers_per_trial',
        'abbreviation' : 'wpt',
        'description' : 'Number of workers per trail, should be set greater than 1 only if gpu per worker is all 1.0.',
        'type' : int,
        'default' : 1,
    },  
    {
        "name": "venv_search_algo",
        'abbreviation' : 'vsa',
        "description": "Hyper-parameter search algorithm used in venv training.",
        "type": str,
        "default": "zoopt",
    },
    {
        "name": "train_venv_trials",
        'abbreviation' : 'tvt',
        "description": "Number of total trails searched by the search algorithm in venv training.",
        "type": int,
        "default": 25,
        'doc': True,
    },
    {
        "name": "policy_search_algo",
        'abbreviation' : 'psa',
        "description": "Hyper-parameter search algorithm used in policy training.",
        "type": str,
        "default": "zoopt",
    },
    {
        "name": "train_policy_trials",
        'abbreviation' : 'tpt',
        "description": "Number of total trails searched by the search algorithm in policy training.",
        "type": int,
        "default": 10,
        'doc': True,
    },
    {
        "name": "global_checkpoint_period",
        "description": "How many seconds between two global checkpoint of tune. DO NOT SET IT TO SMALL VALUES!",
        "type": str,
        "default": "3600",
    },
    {
        "name": "reuse_actors",
        "description": "whether to allow ray to reuse the old actor to skip the initialization of the new actor.",
        "type": bool,
        "default": False,
    },
    {
        "name": "verbose",
        "description": "level of printed log. `0` means no printed log; `1` means minimal printed log; `2` means full log.",
        "type": int,
        "default": 1,
    }
]

DEBUG_CONFIG = deepcopy(DEFAULT_CONFIG)
set_parameter_value(DEBUG_CONFIG, 'mae_test', True)
set_parameter_value(DEBUG_CONFIG, 'mse_test', True)
set_parameter_value(DEBUG_CONFIG, 'histogram_log_frequency', 1)
set_parameter_value(DEBUG_CONFIG, 'fqe_test_frequency', 1)
set_parameter_value(DEBUG_CONFIG, 'train_venv_trials', 5)
set_parameter_value(DEBUG_CONFIG, 'train_policy_trials', 5)
set_parameter_value(DEBUG_CONFIG, 'verbose', 2)
