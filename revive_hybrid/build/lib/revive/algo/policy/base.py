

import os
import ray
import torch
import warnings
import argparse
import importlib
import traceback
import numpy as np
from ray import tune
from loguru import logger
from copy import deepcopy

from ray.tune import Stopper, CLIReporter

from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.constants import (SCHEDULER_STEP_EPOCH, NUM_STEPS,
                                          SCHEDULER_STEP_BATCH, SCHEDULER_STEP)
from ray.util.sgd.utils import (TimerCollection, AverageMeterCollection,
                                NUM_SAMPLES)

from revive.computation.inference import *
from revive.computation.graph import FunctionDecisionNode
from revive.data.batch import Batch
from revive.data.dataset import data_creator
from revive.utils.raysgd_utils import ReviveTorchTrainer, TorchTrainer
from revive.utils.common_utils import *
from revive.utils.tune_utils import TUNE_LOGGERS, CustomSearchGenerator, CustomBasicVariantGenerator
from revive.utils.auth_utils import customer_uploadTrainLog

warnings.filterwarnings('ignore')

def catch_error(func):
    '''push the training error message to data buffer'''
    def wrapped_func(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_message = traceback.format_exc()
            logger.warning('Detect error:{}, Error Message: {}'.format(e,error_message))
            ray.get(self._data_buffer.update_status.remote(self._traj_id, 'error', error_message))
            self._stop_flag = True
            try:
                customer_uploadTrainLog(self.config["trainId"],
                                        os.path.join(os.path.abspath(self._workspace),"revive.log"),
                                        "train.policy",
                                        "fail",
                                        self._max_reward,
                                        self.config["accessToken"])
            except Exception as e:
                logger.info(f"{e}")
            return {
                'stop_flag' : True,
                'reward_trainPolicy_on_valEnv' : - np.inf,
            }
    return wrapped_func

class PolicyOperator(TrainingOperator):

    @property
    def env(self):
        return self.envs_train[0]

    @property
    def policy(self):
        if isinstance(self.train_models, list) or isinstance(self.train_models, tuple):
            return self.train_models[:-1]
        else:
            return self.train_models

    @property
    def val_policy(self):
        if isinstance(self.val_models, list) or isinstance(self.val_models, tuple):
            return self.val_models[:-1]
        else:
            return self.val_models
    
    @property
    def other_models(self):
        if isinstance(self.models, list):
            return self.models[1:]
        else:
            return []

    # NOTE: you need either write the `PARAMETER_DESCRIPTION` or overwrite `get_parameters` and `get_tune_parameters`.
    PARAMETER_DESCRIPTION = [] # a list of dict to describe the parameter of the algorithm

    @classmethod
    def get_parameters(cls, command=None, **kargs):
        parser = argparse.ArgumentParser()

        for description in cls.PARAMETER_DESCRIPTION:
            names = ['--' + description['name']]
            if "abbreviation" in description.keys(): names.append('-' + description["abbreviation"])

            if type(description['type']) is str:
                if 'str' in description['type']:
                    description['type'] = str
                elif 'int' in description['type']:
                    description['type'] = int
                elif 'float' in description['type']:
                    description['type'] = float
                elif 'bool' in description['type']:
                    description['type'] = bool

            parser.add_argument(*names, type=description['type'], default=description['default'])

        return parser.parse_known_args(command)[0].__dict__

    @classmethod
    def get_tune_parameters(cls, config : Dict[str, Any], **kargs):
        r"""
        Use ray.tune to wrap the parameters to be searched.
        """
        reporter = CLIReporter(max_progress_rows=50)
        reporter.add_metric_column("reward_trainPolicy_on_valEnv")

        _search_algo = config['policy_search_algo'].lower()

        tune_params = {
            "name": "policy_tune",
            "progress_reporter": reporter,
            "reuse_actors": config["reuse_actors"],
            "local_dir": config["workspace"],
            "loggers": TUNE_LOGGERS,
            "stop": {
                "stop_flag": True
            },
            "verbose": config["verbose"],
        }

        if _search_algo == 'random':
            random_search_config = {}

            for description in cls.PARAMETER_DESCRIPTION:
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'continuous': 
                        random_search_config[description['name']] = tune.uniform(*description['search_values'])
                    elif description['search_mode'] == 'grid' or description['search_mode'] == 'discrete':
                        random_search_config[description['name']] = tune.choice(description['search_values'])

            config["total_num_of_trials"] = config['train_policy_trials']
            tune_params['config'] = random_search_config
            tune_params['num_samples'] = config["total_num_of_trials"]
            tune_params['search_alg'] = CustomBasicVariantGenerator()

        elif _search_algo == 'zoopt':
            from ray.tune.suggest.zoopt import ZOOptSearch
            from zoopt import ValueType

            if config['parallel_num'] == 'auto':
                if config['use_gpu']:
                    num_of_gpu = int(ray.available_resources()['GPU'])
                    num_of_cpu = int(ray.available_resources()['CPU'])
                    parallel_num = min(int(num_of_gpu / config['policy_gpus_per_worker']), num_of_cpu)
                else:
                    num_of_cpu = int(ray.available_resources()['CPU'])
                    parallel_num = num_of_cpu
            else:
                parallel_num = int(config['parallel_num'])

            assert parallel_num > 0
            config['parallel_num'] = parallel_num

            dim_dict = {}

            for description in cls.PARAMETER_DESCRIPTION:
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'continuous': 
                        dim_dict[description['name']] = (ValueType.CONTINUOUS, description['search_values'], min(description['search_values']))
                    elif description['search_mode'] == 'discrete':
                        dim_dict[description['name']] = (ValueType.DISCRETE, description['search_values'])
                    elif description['search_mode'] == 'grid':
                        dim_dict[description['name']] = (ValueType.GRID, description['search_values'])

            zoopt_search_config = {
                "parallel_num": config['parallel_num']
            }

            config["total_num_of_trials"] = config['train_policy_trials']

            tune_params['search_alg'] = ZOOptSearch(
                algo="Asracos",  # only support Asracos currently
                budget=config["total_num_of_trials"],
                dim_dict=dim_dict,
                metric='reward_trainPolicy_on_valEnv',
                mode="max",
                **zoopt_search_config
            )
            tune_params['search_alg'] = CustomSearchGenerator(tune_params['search_alg'])  # wrap with our generator
            tune_params['num_samples'] = config["total_num_of_trials"]

        elif 'grid' in _search_algo:
            grid_search_config = {}
            config["total_num_of_trials"] = 1

            for description in cls.PARAMETER_DESCRIPTION:
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'grid': 
                        grid_search_config[description['name']] = tune.grid_search(description['search_values'])
                        config["total_num_of_trials"] *= len(description['search_values'])
                    else:
                        warnings.warn(f"Detect parameter {description['name']} is define as searchable in `PARAMETER_DESCRIPTION`. " + \
                            f"However, since grid search does not support search type {description['search_mode']}, the parameter is skipped. " + \
                            f"If this parameter is important to the performance, you should consider other search algorithms. ")

            tune_params['config'] = grid_search_config
            tune_params['search_alg'] = CustomBasicVariantGenerator()

        elif 'bayes' in _search_algo:
            from ray.tune.suggest.bayesopt import BayesOptSearch
            bayes_search_space = {}

            for description in cls.PARAMETER_DESCRIPTION:
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'continuous': 
                        bayes_search_space[description['name']] = description['search_values']
                    else:
                        warnings.warn(f"Detect parameter {description['name']} is define as searchable in `PARAMETER_DESCRIPTION`. " + \
                            f"However, since bayesian search does not support search type {description['search_mode']}, the parameter is skipped. " + \
                            f"If this parameter is important to the performance, you should consider other search algorithms. ")
            
            config["total_num_of_trials"] = config['train_policy_trials']
            
            tune_params['search_alg'] = BayesOptSearch(bayes_search_space, metric="reward_trainPolicy_on_valEnv", mode="max") 
            tune_params['search_alg'] = CustomSearchGenerator(tune_params['search_alg'])  # wrap with our generator
            tune_params['num_samples'] = config["total_num_of_trials"]
        
        else:
            raise ValueError(f'search algorithm {_search_algo} is not supported!')

        return tune_params

    def model_creator(self, config : Dict[str, Any], node : FunctionDecisionNode) -> List[torch.nn.Module]:
        r"""
        Create all the models. The algorithm needs to define models for the nodes to be learned.

        Args:
            :config: configuration parameters
        
        Return: 
            a list of models

        """
        raise NotImplementedError

    def optimizer_creator(self, models : List[torch.nn.Module], config : Dict[str, Any]) -> List[torch.optim.Optimizer]:
        r"""
        Define optimizers for the created models.

        Args:
            :pmodels: list of all the models

            :config: configuration parameters

        Return: 
            a list of optimizers

        """
        raise NotImplementedError

    def data_creator(self, config : Dict[str, Any]):
        r"""
        Create DataLoaders.

        Args:
            :config: configuration parameters

        Return:
             (train_loader, val_loader)
        """
        return data_creator(config, val_horizon=config['test_horizon'], double=self.double_validation)

    def _setup_data(self, config : Dict[str, Any]):
        ''' setup data used in training '''
        self.train_dataset = ray.get(config['dataset'])
        self.val_dataset = ray.get(config['val_dataset'])
        if not self.double_validation:
            train_loader_train, val_loader_val = self.data_creator(config)
            self.register_data(train_loader=train_loader_train, validation_loader=val_loader_val)
            self._train_loader_train = self._train_loader
            self._val_loader_val = self._validation_loader
            self._train_loader_val = None
            self._val_loader_train = None
        else:
            train_loader_train, val_loader_train, train_loader_val, val_loader_val = self.data_creator(config)
        
            self.register_data(train_loader=train_loader_train, validation_loader=val_loader_train)
            self._train_loader_train = self._train_loader
            self._val_loader_train = self._validation_loader

            self.register_data(train_loader=train_loader_val, validation_loader=val_loader_val)
            self._train_loader_val = self._train_loader
            self._val_loader_val = self._validation_loader

    def _setup_models(self, config : Dict[str, Any]):
        r'''setup models, optimizers and dataloaders.'''
        if not self.double_validation:
            self.train_nodes = {policy_name:deepcopy(self._graph.get_node(policy_name)) for policy_name in self.policy_name}
            
            models = self.model_creator(config, self.train_nodes)
            optimizers = self.optimizer_creator(models, config)
            self.models, self.optimizers = self.register(models=(models), optimizers=(optimizers))

            self.train_models = self.models
            self.train_optimizers = self.optimizers
            self.val_models = None
            self.val_optimizers = None
            
            for train_node,policy in zip(self.train_nodes.values(), self.policy):
                train_node.set_network(policy)
        else:
            self.train_nodes = {policy_name:deepcopy(self._graph.get_node(policy_name)) for policy_name in self.policy_name}
            train_models = self.model_creator(config, self.train_nodes)
            train_optimizers = self.optimizer_creator(train_models, config)
            self.val_nodes = {policy_name:deepcopy(self._graph.get_node(policy_name)) for policy_name in self.policy_name}
            val_models = self.model_creator(config, self.val_nodes)
            val_optimizers = self.optimizer_creator(val_models, config)

            self.model_length = len(train_models)
            self.optimizer_length = len(train_optimizers)

            self.models, self.optimizers = self.register(models=(*train_models, *val_models), optimizers=(*train_optimizers, *val_optimizers))

            self.train_models = self.models[:self.model_length]
            self.train_optimizers = self.optimizers[:self.optimizer_length]
            self.val_models = self.models[self.model_length:]
            self.val_optimizers = self.optimizers[self.optimizer_length:]

            for train_node,policy in zip(self.train_nodes.values(), self.policy):
                train_node.set_network(policy)
            for val_node,policy in zip(self.val_nodes.values(), self.val_policy):
                val_node.set_network(policy)

    @catch_error
    def setup(self, config : Dict[str, Any]):
        '''setup everything for training'''
        # parse information from config
        self._data_buffer = config['policy_data_buffer']
        self._workspace = config["workspace"]
        self._user_func = config['user_func']
        self._graph = config['graph']
        self._processor = self._graph.processor
        self.policy_name = self.config['target_policy_name']
        if isinstance(self.policy_name, str):
            self.policy_name = [self.policy_name, ]
        # sort the policy by graph
        self.policy_name = [policy_name for policy_name in self.policy_name if policy_name in self._graph.keys()]

        self.double_validation = self.config['policy_double_validation']
        self._filename = os.path.join(self._workspace, "train_policy.json")
        self._data_buffer.set_total_trials.remote(config.get("total_num_of_trials", 1))
        self._data_buffer.inc_trial.remote()
        self._max_reward = -np.inf

        # get id
        self._ip = ray._private.services.get_node_ip_address()
        
        logger.add(os.path.join(os.path.abspath(self._workspace),"revive.log"))
        
        # set trail seed
        setup_seed(config["global_seed"])

        if 'tag' in self.config: # create by tune
            tag = self.config['tag']
            logger.info("{} is running".format(tag))
            self._traj_id = int(tag.split('_')[0])
            experiment_dir = os.path.join(self._workspace, 'policy_tune')
            for traj_name in filter(lambda x: "ReviveLog" in x, os.listdir(experiment_dir)):
                if len(traj_name.split('_')[1]) == 5: # create by random search or grid search
                    id_index = 3
                else:
                    id_index = 2
                if int(traj_name.split('_')[id_index]) == self._traj_id:
                    self._traj_dir = os.path.join(experiment_dir, traj_name)
                    break
        else:
            self._traj_id = 1
            self._traj_dir = os.path.join(self._workspace, 'policy_train')

        # setup constant
        self._stop_flag = False
        self._batch_cnt = 0
        self._epoch_cnt = 0
        self._max_reward = - np.inf
        self._device = 'cuda' if self._use_gpu else 'cpu' # fix problem introduced in ray 1.1

        # collect venv
        env = ray.get(config['venv_data_buffer'].get_best_venv.remote())
        self.envs = env.env_list
        for env in self.envs:
            env.to(self.device)
            env.requires_grad_(False)
            env.set_target_policy_name(self.policy_name)

        if len(self.envs) == 1:
            warnings.warn('Only one venv found, use it in both training and validation!')
            self.envs_train = self.envs
            self.envs_val = self.envs
        else:
            self.envs_train = self.envs[:len(self.envs)//2]
            self.envs_val = self.envs[len(self.envs)//2:]

        if self.config['num_venv_in_use'] > len(self.envs_train):
            warnings.warn(f"Config requires {self.config['num_venv_in_use']} venvs, but only {len(self.envs_train)} venvs are available.")

        self.envs_train = self.envs_train[:self.config['num_venv_in_use']]
        self.envs_val = self.envs_val[:self.config['num_venv_in_use']]

        self.behaviour_nodes = [self.envs[0].graph.get_node(policy_name) for policy_name in self.policy_name]
        self.behaviour_policys =  [behaviour_node.get_network() for behaviour_node in self.behaviour_nodes]

        # find average reward from expert data
        dataset = ray.get(self.config['dataset'])
        train_data = dataset.data
        train_data.to_torch()
        train_reward = self._user_func(train_data)
        self.train_average_reward = float(train_reward.mean())

        dataset = ray.get(self.config['val_dataset'])
        val_data = dataset.data
        val_data.to_torch()
        val_reward = self._user_func(train_data)
        self.val_average_reward = float(val_reward.mean())     

        # initialize FQE
        self.fqe_evaluator = None

        # prepare for training
        self._setup_data(config)
        self._setup_models(config)
        
        self._save_models(self._traj_dir)
        self._update_metric()

        self.action_dims = {}
        for policy_name in self.policy_name:
            action_dims = []
            for action_dim in self._graph.descriptions[policy_name]:
                action_dims.append(list(action_dim.keys())[0])
            self.action_dims[policy_name] = action_dims

        self.nodes_map = {}
        for node_name in list(self._graph.nodes) + list(self._graph.leaf):
            node_dims = []
            for node_dim in self._graph.descriptions[node_name]:
                node_dims.append(list(node_dim.keys())[0])
            self.nodes_map[node_name] = node_dims


    def _early_stop(self, info : Dict[str, Any]):
        info["stop_flag"] = self._stop_flag
        return info

    def _update_metric(self):
        try:
            policy = torch.load(os.path.join(self._traj_dir, 'policy.pt'))
        except:
            policy = None
        self._data_buffer.update_metric.remote(self._traj_id, {"reward" : self._max_reward, "ip" : self._ip, "policy" : policy, "traj_dir": self._traj_dir})
        self._data_buffer.write.remote(self._filename)
        if self._max_reward == ray.get(self._data_buffer.get_max_reward.remote()):
            self._save_models(self._workspace, with_policy=False)

    def _wrap_policy(self, policys : List[torch.nn.Module,], device: [str, Any] = None) -> PolicyModelDev:
        policy_nodes = deepcopy(self.behaviour_nodes)
        policys = deepcopy(policys)
        for policy_node,policy in zip(policy_nodes,policys):
            policy_node.set_network(policy)
            if device:
                policy_node.to(device)
        policys = PolicyModelDev(policy_nodes)
        return policys

    def _save_models(self, path : str, with_policy : bool = True):
        torch.save(self.policy, os.path.join(path, "tuned_policy.pt"))
        if with_policy:
            policy = self._wrap_policy(self.policy, "cpu")
            torch.save(policy, os.path.join(path, "policy.pt"))

            policy = PolicyModel(policy)
            self.infer_policy = policy
            with open(os.path.join(self._traj_dir, "policy.pkl"), 'wb') as f:
                pickle.dump(policy, f)

    def _get_original_actions(self, batch_data : Batch) -> Batch:
        # return a list with all actions [o, a_1, a_2, ... o']
        # NOTE: we assume key `next_obs` in the data.
        # NOTE: the leading dimensions will be flattened.
        original_data = Batch()
        for policy_name in list(self._graph.leaf) + list(self._graph.keys()):
            action = batch_data[policy_name].view(-1, batch_data[policy_name].shape[-1])
            original_data[policy_name] = self._processor.deprocess_single_torch(action, policy_name)
        return original_data

    def get_ope_dataset(self):
        r''' convert the dataset to OPEDataset used in d3pe '''

        from d3pe.utils.data import OPEDataset

        dataset = ray.get(self.config['dataset'])
        expert_data = dataset.processor.process(dataset.data)

        expert_data = expert_data
        expert_data.to_torch()
        expert_data = generate_rewards(expert_data, reward_fn=lambda data: self._user_func(self._get_original_actions(data)))
        expert_data.to_numpy()

        policy_input_names = self._graph.get_node(self.policy_name).input_names

        if all([node_name in self._graph.transition_map.keys() for node_name in policy_input_names]):
            obs = np.concatenate([expert_data[node_name] for node_name in policy_input_names], axis=-1)
            next_obs = np.concatenate([expert_data['next_' + node_name] for node_name in policy_input_names], axis=-1)
            ope_dataset = OPEDataset(dict(
                obs=obs.reshape((-1, obs.shape[-1])),
                action=expert_data[self.policy_name].reshape((-1, expert_data[self.policy_name].shape[-1])),
                reward=expert_data['reward'].reshape((-1, expert_data['reward'].shape[-1])),
                done=expert_data['done'].reshape((-1, expert_data['done'].shape[-1])),
                next_obs=next_obs.reshape((-1, next_obs.shape[-1])),
            ), start_indexes=dataset._start_indexes)
        else:
            data = dict(
                action=expert_data[self.policy_name][:-1].reshape((-1, expert_data[self.policy_name].shape[-1])),
                reward=expert_data['reward'][:-1].reshape((-1, expert_data['reward'].shape[-1])),
                done=expert_data['done'][:-1].reshape((-1, expert_data['done'].shape[-1])),
            )

            expert_data.to_torch()
            obs = get_input_from_graph(self._graph, self.policy_name, expert_data).numpy()
            expert_data.to_numpy()
            data['obs'] = obs[:-1].reshape((-1, obs.shape[-1]))
            data['next_obs'] = obs[1:].reshape((-1, obs.shape[-1]))

            ope_dataset = OPEDataset(data, dataset._start_indexes - np.arange(dataset._start_indexes.shape[0]))

        return ope_dataset

    def venv_test(self, expert_data : Batch, target_policy, traj_length=None, scope : str = 'trainPolicy_on_valEnv'):
        r""" Use the virtual env model to test the policy model"""
        rewards = []
        envs = self.envs_val if "valEnv" in scope  else self.envs_train
        for env in envs:
            generated_data, info = self._run_rollout(expert_data, target_policy, env, traj_length, deterministic=self.config['deterministic_test'], clip=True)
            reward = generated_data.reward.squeeze(dim=-1)
            t = torch.arange(0, reward.shape[0]).to(reward)
            discount = self.config['test_gamma'] ** t
            discount_reward = torch.sum(discount.unsqueeze(dim=-1) * reward, dim=0)
            rewards.append(discount_reward.mean().item() / self.config['test_horizon'])
        return {f'reward_{scope}' : np.mean(rewards)}
    
    def _fqe_test(self, target_policy, scope='trainPolicy_on_valEnv'):
        if 'offlinerl' in str(type(target_policy)): target_policy.get_action = lambda x: target_policy(x)

        if self.fqe_evaluator is None: # initialize FQE evaluator in the first run
            from d3pe.evaluator.fqe import FQEEvaluator
            self.fqe_evaluator = FQEEvaluator()
            self.fqe_evaluator.initialize(self.get_ope_dataset(), verbose=True)

        with torch.enable_grad():
            info = self.fqe_evaluator(target_policy)
        for k in list(info.keys()): info[f'{k}_{scope}'] = info.pop(k)
        return info

    def _env_test(self, target_policy, scope='trainPolicy_on_valEnv'):
        env = create_env(self.config['task'])
        if env is None:
            return {}

        policy = deepcopy(target_policy)
        policy = policy.to('cpu')
        policy = self._wrap_policy(policy)
        policy = PolicyModel(policy)

        reward, length = test_on_real_env(env, policy)

        return {
            f"real_reward_{scope}" : reward,
            f"real_length_{scope}" : length,
        }

    def _run_rollout(self, 
                     expert_data : Batch, 
                     target_policy, 
                     env : Union[VirtualEnvDev, List[VirtualEnvDev]], 
                     traj_length : int = None, 
                     maintain_grad_flow : bool = False, 
                     deterministic : bool = True,
                     clip : bool = False):
                     
        traj_length = traj_length or expert_data.obs.shape[0]

        if traj_length >= expert_data.shape[0] and len(self._graph.leaf) > 1:
            traj_length = expert_data.shape[0]
            warnings.warn(f'leaf node detected, connot run over the expert trajectory! Reset `traj_length` to {traj_length}!')

        generated_data = self.generate_rollout(expert_data, target_policy, env, traj_length, maintain_grad_flow, deterministic, clip)

        generated_data = generate_rewards(generated_data, reward_fn=lambda \
                data: self._user_func(self._get_original_actions(data)))

        info = {
            "reward": generated_data.reward.mean().item()
        }

        return generated_data, info

    def generate_rollout(self, 
                         expert_data : Batch, 
                         target_policy, 
                         env : Union[VirtualEnvDev, List[VirtualEnvDev]], 
                         traj_length : int, 
                         maintain_grad_flow : bool = False, 
                         deterministic : bool = True,
                         clip : bool = False):
        r"""Generate trajectories based on current policy.

        Args:
            :expert_data: sampled data from the dataset.

            :target_policy: target_policy

            :env: env

            :traj_length: traj_length

            :maintain_grad_flow: maintain_grad_flow

        Return: 
            batch trajectories

        """
        for policy in target_policy:
            policy.reset()
        
        if isinstance(env, list):
            for _env in env: _env.reset()
        else:
            env.reset()

        assert len(self._graph.leaf) == 1 or traj_length <= expert_data.shape[0], \
            'There is leaf node on the graph, cannot generate trajectory beyond expert data'

        generated_data = []
        current_batch = Batch({k : expert_data[0][k] for k in self._graph.leaf})
        batch_size = current_batch.shape[0]

        sample_fn = lambda dist: dist.rsample() if maintain_grad_flow else dist.sample()

        for i in range(traj_length):
            for policy_index, policy_name in enumerate(self.policy_name):
                if isinstance(env, list):
                    result_batch = []
                    for _env in env:
                        _current_batch = deepcopy(current_batch)
                        _current_batch = _env.pre_computation(_current_batch, deterministic, clip, policy_index)
                        result_batch.append(_current_batch)
                    result_batch = Batch.stack(result_batch, axis=0)
                    select_indexes = np.random.randint(0, len(self.envs_train), size=batch_size)
                    current_batch = result_batch[select_indexes, np.arange(batch_size)]
                else:
                    current_batch = env.pre_computation(current_batch, deterministic, clip, policy_index)

                policy_input = get_input_from_graph(self._graph, policy_name, current_batch)
                if 'offlinerl' in str(type(target_policy[policy_index])):
                    action = target_policy[policy_index](policy_input)
                else:
                    action_dist = target_policy[policy_index](policy_input)
                    action = sample_fn(action_dist)
                    action_log_prob = (action_dist.log_prob(action).unsqueeze(dim=-1)).detach()
                    current_batch[policy_name + '_log_prob'] = action_log_prob
                current_batch[policy_name] = action

                if isinstance(env, list):
                    result_batch = []
                    for _env in env:
                        _current_batch = deepcopy(current_batch)
                        _current_batch = _env.post_computation(_current_batch, deterministic, clip, policy_index)
                        result_batch.append(_current_batch)
                    result_batch = Batch.stack(result_batch, axis=0)
                    select_indexes = np.random.randint(0, len(self.envs_train), size=batch_size)
                    current_batch = result_batch[select_indexes, np.arange(batch_size)]
                else:
                    current_batch = env.post_computation(current_batch, deterministic, clip, policy_index)

                generated_data.append(current_batch)

                if i == traj_length - 1 : break

                current_batch = Batch(self._graph.state_transition(current_batch))
                for k in self._graph.leaf: 
                    if not k in self._graph.transition_map.keys(): current_batch[k] = expert_data[i+1][k]

        generated_data = Batch.stack(generated_data)

        return generated_data

    def train_epoch(self, iterator, info, **kwargs):
        self._epoch_cnt += 1

        if hasattr(self, "model"):
            self.model.train()
        if hasattr(self, "models"):
            try:
                for _model in self.models:
                    if isinstance(_model, torch.nn.Module):
                        _model.train()
            except:
                self.models.train()

        metric_meters_train = AverageMeterCollection()

        for batch_idx, batch in enumerate(iter(self._train_loader_train)):
            batch_info = {
                "batch_idx": batch_idx,
                "global_step": self.global_step
            }
            batch_info.update(info)
            metrics = self.train_batch(batch, batch_info=batch_info, scope='train')

            metric_meters_train.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
            self.global_step += 1


        info = metric_meters_train.summary()

        if self.double_validation:
            metric_meters_val = AverageMeterCollection()

            for batch_idx, batch in enumerate(iter(self._val_loader_train)):
                batch_info = {
                    "batch_idx": batch_idx,
                    "global_step": self.global_step
                }
                batch_info.update(info)
                metrics = self.train_batch(batch, batch_info=batch_info, scope='val')

                metric_meters_val.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
                self.global_step += 1

            info.update(metric_meters_val.summary())
            
        if os.path.exists(os.path.join(self._workspace,'.env.json')):
            import json
            with open(os.path.join(self._workspace,'.env.json'), 'r') as f:
                _data = json.load(f)
            if _data["REVIVE_STOP"]:
                self._stop_flag = True
                info["stop_flag"] = self._stop_flag 

        return {k : info[k] for k in filter(lambda k: not k.startswith('last'), info.keys())}

    @catch_error
    def validate(self, val_iterator, info):
        if info is None:
            info = dict()
            pass
        
        # switch to evaluate mode
        if hasattr(self, "model"):
            self.model.eval()
        if hasattr(self, "models"):
            try:
                for _model in self.models:
                    if isinstance(_model, torch.nn.Module):
                        _model.eval()
            except:
                self.models.eval()

        metric_meters_train = AverageMeterCollection()
        with torch.no_grad():
            for batch_idx, batch in enumerate(iter(self._val_loader_val)):
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                metrics = self.validate_batch(batch, batch_info, scope='trainPolicy_on_valEnv')
                metric_meters_train.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))

        info = metric_meters_train.summary()

        if self.double_validation:
            metric_meters_train = AverageMeterCollection()
            with torch.no_grad():
                for batch_idx, batch in enumerate(iter(self._train_loader_val)):
                    batch_info = {"batch_idx": batch_idx}
                    batch_info.update(info)
                    metrics = self.validate_batch(batch, batch_info, scope='trainPolicy_on_trainEnv')
                    metric_meters_train.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))

            info.update(metric_meters_train.summary())

        if (not self.config['real_env_test_frequency'] == 0) and \
            self._epoch_cnt % self.config['real_env_test_frequency'] == 0:
            info.update(self._env_test(self.policy))
        if (not self.config['fqe_test_frequency'] == 0) and \
            self._epoch_cnt % self.config['fqe_test_frequency'] == 0:
            info.update(self._fqe_test(self.policy))
            
            

        if self.double_validation:
            metric_meters_val = AverageMeterCollection()
            with torch.no_grad():
                for batch_idx, batch in enumerate(iter(self._train_loader_val)):
                    batch_info = {"batch_idx": batch_idx}
                    batch_info.update(info)
                    metrics = self.validate_batch(batch, batch_info, scope='valPolicy_on_trainEnv')
                    metric_meters_val.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
                    
            info.update(metric_meters_val.summary())

            metric_meters_val = AverageMeterCollection()
            with torch.no_grad():
                for batch_idx, batch in enumerate(iter(self._val_loader_val)):
                    batch_info = {"batch_idx": batch_idx}
                    batch_info.update(info)
                    metrics = self.validate_batch(batch, batch_info, scope='valPolicy_on_valEnv')
                    metric_meters_val.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))

            info.update(metric_meters_val.summary())

            if (not self.config['real_env_test_frequency'] == 0) and \
                self._epoch_cnt % self.config['real_env_test_frequency'] == 0:
                info.update(self.env_test(self.val_policy))
            if (not self.config['fqe_test_frequency'] == 0) and \
                self._epoch_cnt % self.config['fqe_test_frequency'] == 0:
                info.update(self.fqe_test(self.val_policy))
        
        info = {k : info[k] for k in filter(lambda k: not k.startswith('last'), info.keys())}

        if self.double_validation:
            reward_flag = "reward_trainPolicy_on_valEnv"
        else:
            reward_flag = "reward_trainenv"
        if info[reward_flag] > self._max_reward:
            self._max_reward = info[reward_flag]
            self._save_models(self._traj_dir)
            self._update_metric()

        if self._stop_flag:
            # revive online server
            try:
                customer_uploadTrainLog(self.config["trainId"],
                                        os.path.join(os.path.abspath(self._workspace),"revive.log"),
                                        "train.policy",
                                        "success",
                                        self._max_reward,
                                        self.config["accessToken"])
            except Exception as e:
                logger.info(f"{e}")

            # save double validation plot after training
            if self.double_validation:
                plt_double_venv_validation(self._traj_dir, 
                                        self.train_average_reward, 
                                        self.val_average_reward, 
                                        os.path.join(self._traj_dir, 'double_validation.png'))
            
            graph = self.envs_train[0].graph
            for policy_index, policy_name in enumerate(self.policy_name):
                graph.nodes[policy_name] = deepcopy(self.infer_policy)._policy_model.nodes[policy_index]

            # save rolllout action image
            rollout_save_path = os.path.join(self._traj_dir, 'rollout_images')
            save_rollout_action(rollout_save_path, graph, self.device, self.train_dataset, self.nodes_map)

            # policy to tree and plot the tree
            tree_save_path = os.path.join(self._traj_dir, 'policy_tree')
            net_to_tree(tree_save_path, graph, self.device, self.train_dataset, self.action_dims )

        return info

    def train_batch(self, expert_data : Batch, batch_info : Dict[str, float], scope : str = 'train'):
        raise NotImplementedError

    def validate_batch(self, expert_data : Batch, batch_info : Dict[str, float], scope : str = 'trainPolicy_on_valEnv'):
        expert_data.to_torch(device=self._device)
        if not self.double_validation:
            info = self.venv_test(expert_data, self.policy, traj_length=self.config['test_horizon'], scope="trainenv")
        elif "trainPolicy" in scope:
            info = self.venv_test(expert_data, self.policy, traj_length=self.config['test_horizon'], scope=scope)
        elif "valPolicy" in scope:
            info = self.venv_test(expert_data, self.val_policy, traj_length=self.config['test_horizon'], scope=scope)
        
        return info


class PolicyAlgorithm:
    def __init__(self, algo : str):
        self.algo = algo
        try:
            self.algo_module = importlib.import_module(f'revive.dist.algo.policy.{self.algo.split(".")[0]}')
            logger.info(f"Import encryption policy algorithm module -> {self.algo}!")
        except:
            self.algo_module = importlib.import_module(f'revive.algo.policy.{self.algo.split(".")[0]}')
            logger.info(f"Import policy algorithm module -> {self.algo}!")

        # Assume there is only one operator other than PolicyOperator
        for k in self.algo_module.__dir__():
            if 'Operator' in k and not k == 'PolicyOperator':
                self.operator = getattr(self.algo_module, k)

    def get_trainer(self, config):
        config['algo'] = self.algo
        try:  # if there is a custom function
            _get_trainer = getattr(self.algo_module, 'get_trainer')
            return _get_trainer(config)
        except AttributeError:  # fallback to default
            if config['is_crypto'] == 0:
                return ReviveTorchTrainer(
                    training_operator_cls=self.operator,
                    num_workers=config['workers_per_trial'],
                    config=config,
                    use_gpu=config['use_gpu'],
                    use_fp16=config['use_fp16'],
                    num_gpus_per_worker=config['policy_gpus_per_worker'],
                )
            else:
                return TorchTrainer(
                    training_operator_cls=self.operator,
                    num_workers=config['workers_per_trial'],
                    config=config,
                    use_gpu=config['use_gpu'],
                    use_fp16=config['use_fp16'],
                )
        except Exception as e:
            logger.error('Detect Error: {}'.format(e))
            raise e
    
    def get_trainable(self, config):
        config['algo'] = self.algo
        try:  # if there is a custom function
            _get_trainable = getattr(self.algo_module, 'get_trainable')
            return _get_trainable(config)
        except AttributeError:  # fallback to default
            if config['is_crypto'] == 0:
                return ReviveTorchTrainer.as_trainable(
                    training_operator_cls=self.operator,
                    num_workers=config['workers_per_trial'],
                    config=config,
                    use_gpu=config['use_gpu'],
                    use_fp16=config['use_fp16'],
                    num_gpus_per_worker=config['policy_gpus_per_worker'],
                )
            else:
                return TorchTrainer.as_trainable(
                    training_operator_cls=self.operator,
                    num_workers=config['workers_per_trial'],
                    config=config,
                    use_gpu=config['use_gpu'],
                    use_fp16=config['use_fp16'],
                )          
        except Exception as e:
            logger.error('Detect Error: {}'.format(e))
            raise e

    def get_parameters(self, command=None):
        try:
            return self.operator.get_parameters(command, algo=self.algo)
        except AttributeError:
            raise AttributeError("Custom algorithm need to implement `get_parameters`")
        except Exception as e:
            logger.error('Detect Unknown Error:'.format(e))
            raise e

    def get_tune_parameters(self, config):
        try:
            return self.operator.get_tune_parameters(config, algo=self.algo)
        except AttributeError:
            raise AttributeError("Custom algorithm need to implement `get_tune_parameters`")
        except Exception as e:
            logger.error('Detect Unknown Error:'.format(e))
            raise e
