import os
import ray
import json
import time
import torch
import warnings
from ray import tune
from loguru import logger
from ray.tune import CLIReporter
from collections import OrderedDict

from revive.computation.inference import *
from revive.utils.common_utils import update_description, setup_seed
from revive.algo.venv import VenvAlgorithm
from revive.algo.policy import PolicyAlgorithm
from revive.utils.tune_utils import CustomBasicVariantGenerator, CustomSearchGenerator, TUNE_LOGGERS, SysStopper


class DataBufferEnv:
    def __init__(self, venv_max_num : int = 10):
        '''
            : param venv_max_num: Max number of venv stored in best_venv
        '''
        self.venv_max_num = venv_max_num
        self.num_of_trial = 0
        self.least_metric = float('inf')
        self.max_acc = - float('inf')
        self.total_num_of_trials = -1
        self.best_id = None
        self.metric_dict = OrderedDict()
        self.status_dict = OrderedDict()
        self.best_venv = None
        self.best_model_workspace = None

    def update_status(self, task_id : int, status : str, message : str = ''):
        old_message = '' if not task_id in self.status_dict.keys() else self.status_dict[task_id][1]
        self.status_dict[task_id] = (status, old_message + message)

    def get_status(self) -> Dict[int, Tuple[str, str]]:
        return self.status_dict

    def set_best_venv(self, venv : VirtualEnv):
        self.best_venv = venv

    def get_best_venv(self) -> VirtualEnv:
        return self.best_venv

    def get_best_model_workspace(self) -> str:
        return self.best_model_workspace

    def set_total_trials(self, trials : int):
        self.total_num_of_trials = trials

    def inc_trial(self) -> int:
        self.num_of_trial += 1
        return self.num_of_trial
    
    def get_num_of_trial(self) -> int:
        return self.num_of_trial

    def update_metric(self, task_id : int, metric : Dict[int, Union[float, VirtualEnvDev]]):
        self.metric_dict[task_id] = metric
        self.metric_dict = OrderedDict(sorted(self.metric_dict.items(), key=lambda x: x[1]['metric']))
        self.best_id, info = list(self.metric_dict.items())[0]
        self.least_metric = info['metric']
        self.max_acc = info['acc']
        self.best_model_workspace = info['traj_dir']
        
        venv_list = self.get_venv_list()[:self.venv_max_num]
        venv = VirtualEnv([pair[0] for pair in venv_list] + [pair[1] for pair in venv_list])
        self.set_best_venv(venv)

    def get_max_acc(self) -> float:
        return self.max_acc

    def get_least_metric(self) -> float:
        return self.least_metric

    def get_best_id(self) -> int:
        return self.best_id

    def get_venv_list(self) -> List[VirtualEnvDev]:
        return [(metric['venv_train'], metric['venv_val']) for metric in self.metric_dict.values() if metric['venv_train'] is not None]

    def get_dict(self):
        # clean out venv references
        metric_dict = OrderedDict()
        for id, mdict in self.metric_dict.items():
            new_mdict = {k : v for k, v in mdict.items() if not isinstance(v, VirtualEnvDev)}
            metric_dict[id] = new_mdict

        return {
            "num_of_trial" : self.num_of_trial,
            "total_num_of_trials" : self.total_num_of_trials,
            "least_metric" : self.least_metric,
            "max_acc" : self.max_acc,
            "best_id" : self.best_id,
            "metrics" : metric_dict
        }

    def write(self, filename : str):
        with open(filename, 'w') as f:
            json.dump(self.get_dict(), f, indent=4)


class DataBufferPolicy:
    def __init__(self):
        self.num_of_trial = 0
        self.max_reward = - float('inf')
        self.total_num_of_trials = -1
        self.best_id = None
        self.reward_dict = OrderedDict()
        self.status_dict = OrderedDict()
        self.best_policy = None
        self.best_model_workspace = None

    def update_status(self, task_id : int, status : str, message : str = ''):
        old_message = '' if not task_id in self.status_dict.keys() else self.status_dict[task_id][1]
        self.status_dict[task_id] = (status, old_message + message)

    def get_status(self) -> Dict[int, Tuple[str, str]]:
        return self.status_dict

    def set_best_policy(self, policy : PolicyModel):
        self.best_policy = policy

    def get_best_policy(self) -> PolicyModel:
        return self.best_policy

    def get_best_model_workspace(self) -> str:
        return self.best_model_workspace

    def set_total_trials(self, trials : int):
        self.total_num_of_trials = trials

    def inc_trial(self) -> int:
        self.num_of_trial += 1
        return self.num_of_trial
    
    def get_num_of_trial(self) -> int:
        return self.num_of_trial

    def update_metric(self, task_id : int, metric : Dict[str, Union[float, PolicyModelDev]]):
        self.reward_dict[task_id] = metric
        self.reward_dict = OrderedDict(sorted(self.reward_dict.items(), key=lambda x: x[1]['reward'], reverse=True))
        self.best_id, info = list(self.reward_dict.items())[0]
        self.max_reward = info['reward']
        self.best_model_workspace = info['traj_dir']

        self.set_best_policy(PolicyModel(self.reward_dict[self.best_id]['policy']))

    def get_max_reward(self):
        return self.max_reward

    def get_best_id(self):
        return self.best_id

    def get_dict(self):
        # clean out policy references
        reward_dict = OrderedDict()
        for id, mdict in self.reward_dict.items():
            new_mdict = {k : v for k, v in mdict.items() if not isinstance(v, PolicyModelDev)}
            reward_dict[id] = new_mdict

        return {
            "num_of_trial" : self.num_of_trial,
            "total_num_of_trials" : self.total_num_of_trials,
            "max_reward" : self.max_reward,
            "best_id" : self.best_id,
            "rewards" : reward_dict
        }

    def write(self, filename : str):
        with open(filename, 'w') as f:
            json.dump(self.get_dict(), f, indent=4)


class DataBufferTuner:
    def __init__(self, mode : str, budget : int):
        self.mode = mode
        self.current_trail = 0
        self.budget = budget
        self.best_metric = - float('inf') if self.mode == 'max' else float('inf')
        self.best_parameter = None

    def get_state(self):
        return {
            'best_metric' : self.best_metric, 
            'best_parameter' : self.best_parameter,
            'searched_trail' : self.current_trail,
            'budget' : self.budget
        }

    def update(self, parameter : Dict[str, np.ndarray], metric : float):
        self.current_trail += 1
        if self.mode == 'max':
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_parameter = parameter
        else:
            if metric < self.best_metric:
                self.best_metric = metric
                self.best_parameter = parameter 


class Logger:
    def __init__(self):
        self.log = {}

    def get_log(self):
        return self.log

    def update(self, key, value):
        self.log[key] = value

def trial_str_creator(trial):
    return "{}_{}".format("ReviveLog", trial.trial_id)

class TuneVenvTrain(object):
    def __init__(self, config, logger, command=None):       
        self.config = config
        self.logger = logger
        self.algo = VenvAlgorithm(self.config["venv_algo"])
        if 'venv_algo_config' in config.keys() and self.config['venv_algo'] in config['venv_algo_config'].keys():
            update_description(self.algo.operator.PARAMETER_DESCRIPTION, config['venv_algo_config'][self.config['venv_algo']])
        self.config.update(self.algo.get_parameters(command))
        os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = self.config['global_checkpoint_period']

    def train(self,):
        from ray import tune
        self.logger.update.remote(key="task_state", value="Run")

        torchTrainable = self.algo.get_trainable(self.config) 
        tune_params = self.algo.get_tune_parameters(self.config)
        
        # Seet the seed
        setup_seed(self.config["global_seed"])
        # Set tune stop
        tune_params["stop"] = SysStopper(workspace = self.config['workspace'])
        tune_params["trial_name_creator"] = trial_str_creator
        analysis = tune.run(torchTrainable, **tune_params)
        best_df = analysis.dataframe(metric="least_metric", mode="min")
        best_config = analysis.get_best_config(metric="least_metric", mode="min")

        self.logger.update.remote(key="task_state", value="End")


class TunePolicyTrain(object):
    def __init__(self, config, logger, venv_logger=None, command=None):     
        self.config = config
        self.logger = logger
        self.venv_logger = venv_logger
        self.algo = PolicyAlgorithm(self.config['policy_algo'])
        if 'policy_algo_config' in config.keys() and self.config['policy_algo'] in config['policy_algo_config'].keys():
            update_description(self.algo.operator.PARAMETER_DESCRIPTION, config['policy_algo_config'][self.config['policy_algo']])
        self.config.update(self.algo.get_parameters(command))
        os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = self.config['global_checkpoint_period']

    def train(self,):
        if self.venv_logger is not None:
            while True: # block until venv train finish
                log = ray.get(self.venv_logger.get_log.remote())
                if log.get('task_state') == 'End':
                    break
                time.sleep(10)

        while True: # block until venv available
            if os.path.exists(os.path.join(self.config['workspace'], 'env.pkl')):
                break
            logger.info('Waiting for venv ...' )
            time.sleep(10)

        self.logger.update.remote(key="task_state", value="Run")

        torchTrainable = self.algo.get_trainable(self.config)
        tune_params = self.algo.get_tune_parameters(self.config)
        
        # Seet the seed
        setup_seed(self.config["global_seed"])
        # Set tune stop
        tune_params["stop"] = SysStopper(workspace = self.config['workspace'])
        tune_params["trial_name_creator"] = trial_str_creator
        analysis = tune.run(torchTrainable, **tune_params)
        best_df = analysis.dataframe(metric="reward_trainPolicy_on_valEnv", mode="max")
        best_config = analysis.get_best_config(metric="reward_trainPolicy_on_valEnv", mode="max")

        self.logger.update.remote(key="task_state", value="End")


class VenvTrain(object):
    def __init__(self, config, logger, command=None):
        self.config = config
        self.logger = logger

        self.algo = VenvAlgorithm(self.config["venv_algo"])  # 指定venv的训练算法
        if 'venv_algo_config' in config.keys() and self.config['venv_algo'] in config['venv_algo_config'].keys():
            update_description(self.algo.operator.PARAMETER_DESCRIPTION, config['venv_algo_config'][self.config['venv_algo']])
        self.config.update(self.algo.get_parameters(command))
        self.workspace = os.path.join(self.config["workspace"], 'venv_train')
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

    def train(self):
        self.logger.update.remote(key="task_state", value="Run")
        
        trainer = self.algo.get_trainer(self.config) 
        
        # Seet the seed
        setup_seed(self.config["global_seed"])

        from .tune_utils import VALID_SUMMARY_TYPES
        from torch.utils.tensorboard import SummaryWriter
        from tabulate import tabulate

        writer = SummaryWriter(self.workspace)
        epoch = 0
        while True:
            train_stats = trainer.train()
            _train_stats = list(filter(lambda x: not 'last' in x[0] and type(x[1]) in VALID_SUMMARY_TYPES, train_stats.items()))
            if self.config['verbose'] == 2: print(tabulate(_train_stats, numalign="right"))
            val_stats = trainer.validate()
            _val_stats = list(filter(lambda x: not 'last' in x[0] and type(x[1]) in VALID_SUMMARY_TYPES, val_stats.items()))
            if self.config['verbose'] == 2: print(tabulate(_val_stats, numalign="right"))

            for k, v in [*train_stats.items(), *val_stats.items()]:
                if type(v) in VALID_SUMMARY_TYPES:
                    writer.add_scalar(k, v, global_step=epoch)
                elif isinstance(v, torch.Tensor):
                    v = v.view(-1)
                    writer.add_histogram(k, v, global_step=epoch)
            writer.flush()

            train_stats.update(val_stats)
            if train_stats.get('stop_flag', False):
                logger.info('Early stopped!')
                break

            epoch += 1

        self.logger.update.remote(key="task_state", value="End")
        trainer.shutdown()  # Without this line, GPU memory will leak


class PolicyTrain(object):
    def __init__(self, config, logger, venv_logger=None, command=None):
        self.config = config
        self.logger = logger
        self.venv_logger = venv_logger
        self.algo = PolicyAlgorithm(self.config['policy_algo'])
        if 'policy_algo_config' in config.keys() and self.config['policy_algo'] in config['policy_algo_config'].keys():
            update_description(self.algo.operator.PARAMETER_DESCRIPTION, config['policy_algo_config'][self.config['policy_algo']])
        self.config.update(self.algo.get_parameters(command))
        self.workspace = os.path.join(self.config["workspace"], 'policy_train')
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

    def train(self):
        if self.venv_logger is not None:
            while True: # block until venv train finish
                log = ray.get(self.venv_logger.get_log.remote())
                if log.get('task_state') == 'End':
                    break
                time.sleep(10)

        while True: # block until venv available
            if os.path.exists(os.path.join(self.config['workspace'], 'env.pkl')):
                break
            logger.info('Waiting for venv ...')
            time.sleep(10)

        self.logger.update.remote(key="task_state", value="Run")

        trainer = self.algo.get_trainer(self.config)
        # Seet the seed
        setup_seed(self.config["global_seed"])

        from .tune_utils import VALID_SUMMARY_TYPES
        from torch.utils.tensorboard import SummaryWriter
        from tabulate import tabulate

        writer = SummaryWriter(self.workspace)
        epoch = 0
        while True:
            train_stats = trainer.train()
            _train_stats = list(filter(lambda x: not 'last' in x[0] and type(x[1]) in VALID_SUMMARY_TYPES, train_stats.items()))
            if self.config['verbose'] == 2: print(tabulate(_train_stats, numalign="right"))
            val_stats = trainer.validate()
            _val_stats = list(filter(lambda x: not 'last' in x[0] and type(x[1]) in VALID_SUMMARY_TYPES, val_stats.items()))
            if self.config['verbose'] == 2: print(tabulate(_val_stats, numalign="right"))

            for k, v in [*train_stats.items(), *val_stats.items()]:
                if type(v) in VALID_SUMMARY_TYPES:
                    writer.add_scalar(k, v, global_step=epoch)
                elif isinstance(v, torch.Tensor):
                    v = v.view(-1)
                    writer.add_histogram(k, v, global_step=epoch)
            writer.flush()

            train_stats.update(val_stats)
            if train_stats.get('stop_flag', False):
                logger.info('Early stopped!')
                break

            epoch += 1

        self.logger.update.remote(key="task_state", value="End")
        trainer.shutdown()  # Without this line, GPU memory will leak


def default_evaluate(config):
    static = config.pop('static')
    env = ray.get(static['venv_buffer'].get_best_venv.remote())
    state = static['state']
    objective = static['objective']
    buffer = static['buffer']
    graph = env.graph

    parameter = {}
    for tunable_name in graph.tunable:
        parameter[tunable_name] = np.array(
            [config[parameter_name] for parameter_name in sorted(filter(lambda x: tunable_name in x, config.keys()))]
        )
    state[0].update(parameter)

    states = env.infer_k_steps(state)
    
    value = sum([objective(s) for s in states])

    buffer.update.remote(parameter, value)

    return {'objective' : value}

class ParameterTuner(object):
    def __init__(self, config, mode, initial_state, logger, venv_logger=None):
        self.config = config
        self.mode = mode
        self.initial_state = initial_state
        self.logger = logger
        self.venv_logger = venv_logger

    def run(self):
        if self.venv_logger is not None:
            while True: # block until venv train finish
                log = ray.get(self.venv_logger.get_log.remote())
                if log.get('task_state') == 'End':
                    break
                time.sleep(10)

        while True: # block until venv available
            if os.path.exists(os.path.join(self.config['workspace'], 'env.pkl')):
                break
            logger.info('Waiting for venv ...')
            time.sleep(10)

        self.logger.update.remote(key="task_state", value="Run")

        env = ray.get(self.config['venv_data_buffer'].get_best_venv.remote())
        graph = env.graph
        dataset = ray.get(self.config['dataset'])

        if len(graph.external_factors) - len(graph.tunable) > 0:
            for k, v in self.initial_state.items():
                if len(v.shape) == 2:
                    horizon = v.shape[0]
            state = [{node_name : self.initial_state[node_name] for node_name in graph.transition_map.keys()}] + [{}] * (horizon - 1)
            for i in range(horizon):
                for k, v in self.initial_state.items():
                    if len(v.shape) == 2: state[i][k] = v[i]
            warnings.warn(f'Detect leaf node on graph, reset rollout horizon to {horizon}!')
        else:
            if self.config['parameter_tuning_rollout_horizon'] > dataset.max_length:
                warnings.warn('Detect rollout length higher than max length in the dataset!')
            state = [self.initial_state] + [{}] * (self.config['parameter_tuning_rollout_horizon'] - 1)

        static_config = {'static' : {'venv_buffer' : self.config['venv_data_buffer'], 'state' : state, 'objective' : self.config['user_func'], 'buffer' : self.config['tuner_data_buffer']}}

        reporter = CLIReporter(max_progress_rows=50)
        reporter.add_metric_column("objective")

        tune_params = {
            "name": "parameter_tuning",
            "progress_reporter": reporter,
            'metric' : 'objective',
            'mode' : self.mode,
            "reuse_actors": self.config["reuse_actors"],
            "local_dir": self.config["workspace"],
            "loggers": TUNE_LOGGERS,
            "verbose": self.config["verbose"],
            'num_samples' : self.config['parameter_tuning_budget']
        }

        if self.config['parameter_tuning_algorithm'] == 'random':
            random_search_config = static_config

            for tunable_name in graph.tunable:
                for i, d in enumerate(dataset.raw_columns[tunable_name]):
                    name = list(d.keys())[0]
                    _config = d[name]
                    if _config['type'] == 'continuous':
                        random_search_config[f'{tunable_name}_{"%.09d" % i}'] = tune.uniform(_config['min'], _config['max'])
                    elif _config['type'] == 'discrete':
                        random_search_config[f'{tunable_name}_{"%.09d" % i}'] = tune.grid_search(np.linspace(_config['min'], _config['max'], _config['num']).tolist())
                    elif _config['type'] == 'category':
                        random_search_config[f'{tunable_name}_{"%.09d" % i}'] = tune.grid_search(_config['values'])

            tune_params['config'] = random_search_config
            tune_params['search_alg'] = CustomBasicVariantGenerator()

        elif self.config['parameter_tuning_algorithm'] == 'zoopt':
            from ray.tune.suggest.zoopt import ZOOptSearch
            from zoopt import ValueType

            num_of_cpu = int(ray.available_resources()['CPU'])
            parallel_num = num_of_cpu

            assert parallel_num > 0

            dim_dict = {}

            for tunable_name in graph.tunable:
                for i, d in enumerate(dataset.raw_columns[tunable_name]):
                    name = list(d.keys())[0]
                    _config = d[name]
                    if _config['type'] == 'continuous':
                        dim_dict[f'{tunable_name}_{"%.09d" % i}'] = (ValueType.CONTINUOUS, [_config['min'], _config['max']], 1e-10)
                    elif _config['type'] == 'discrete':
                        dim_dict[f'{tunable_name}_{"%.09d" % i}'] = (ValueType.DISCRETE, np.linspace(_config['min'], _config['max'], _config['num']).tolist())
                    elif _config['type'] == 'category':
                        dim_dict[f'{tunable_name}_{"%.09d" % i}'] = (ValueType.GRID, _config['values'])

            zoopt_search_config = {
                "parallel_num": parallel_num
            }

            tune_params['search_alg'] = ZOOptSearch(
                algo="Asracos",  # only support Asracos currently
                budget=self.config['parameter_tuning_budget'],
                dim_dict=dim_dict,
                metric='objective',
                mode=self.mode,
                **zoopt_search_config
            )
            
            tune_params['config'] = static_config
            tune_params['search_alg'] = CustomSearchGenerator(tune_params['search_alg'])  # wrap with our generator

        analysis = tune.run(
            default_evaluate,
            **tune_params
        )

        self.logger.update.remote(key="task_state", value="End")
