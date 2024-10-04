import json
import os
import pickle
import sys
import uuid
import warnings
from typing import Tuple

import ray
from loguru import logger

from revive.utils.common_utils import get_reward_fn, list2parser, setup_seed
from revive.computation.inference import PolicyModel, VirtualEnv, VirtualEnvDev
from revive.conf.config import DEBUG_CONFIG, DEFAULT_CONFIG
from revive.data.dataset import OfflineDataset
from revive.utils.server_utils import *

warnings.filterwarnings('ignore')


class ReviveServer:
    r"""
    A class that uses `ray` to manage all the training tasks. It can automatic search for optimal hyper-parameters.

    `ReviveServer` will do five steps to initialize: 

    1. Create or connect to a ray cluster. The behavior is controlled by `address` parameter. If the `address` 
    parameter is `None`, it will create its own cluster. If the `address` parameter is specified, it will 
    connect to the existing cluster. 

    2. Load config for training. The config is stored in `revive/config.py`. You can change these parameters 
    by editing the file, passing through command line or through `custom_config`.

    3. Load data and its config, register reward function. The data files are specified by parameters 
    `dataset_file_path`, `dataset_desc_file_path` and `val_file_path`. Note the `val_file_path` is optional. 
    If it is not specified, revive will split the training data. All the data will be put into the ray 
    object store to share among the whole cluster.

    4. Create the folder to store results. The top level folder of these logs are controlled by `log_dir` parameter. 
    If it is not provided, the default value is the `logs` folder under the revive repertory. The second-level folder 
    is controlled by the `run_id` parameter in the training config. If it is not specified, we will generate a random id 
    for the folder. All the training results will be placed in the second-level folder.

    5. Create result server as ray actor, and try to load existing results in the log folder. This class is very useful when 
    you want to train a policy or tune parameters from an already trained simulator.
    
    Initialization a Revive Server.

    Args:
        :dataset_file_path (str): 
            The file path where the training dataset is stored. If the val_file_path
            is "None", Some data will be cut out from the training dataset as the validation dataset. (e.g., "/data/data.npz" )
        :dataset_desc_file_path (str): 
                The file path where the data description file is stored. (e.g., "/data/test.yaml" )
        :val_file_path (str): 
            The file path where the validate dataset is stored. If it's "None", 
            the validation dataset will be cut out from the training dataset.
        :reward_file_path (str): 
            The storage path of the file that defines the reward function.
        :target_policy_name (str): 
            Name of target policy to be optimized. Maximize the defined reward by optimizing the policy.
            If it is None, the first policy in the graph will be chosen.
        :log_dir (str): 
            Training log and saved model storage folder.
        :run_id (str): 
            The ID of the current running experiment is used to distinguish different training. 
            When it is not provided, an ID will be automatically generated
        :address (str): 
            The address of the ray cluster,  If the `address` parameter is `None`, it will create its own cluster.
        :venv_mode ("tune","once","None"): 
            Control the mode of venv training. 
            `tune` means conducting hyper-parameter search; 
            `once` means train with the default hyper-parameters; 
            `None` means skip.'
        :policy_mode ("tune","once","None"): 
            Control the mode of venv training. 
            `tune` means conducting hyper-parameter search; 
            `once` means train with the default hyper-parameters; 
            `None` means skip.'
        :tuning_mode ("max","min","None"):
            Control the mode of parameter tuning. 
            `max` and `min` means enabling tuning and the direction; 
            `None` means skip.'
            This feature is currently unstable
        :tune_initial_state (str): 
            Initial state of parameter tuning, needed when tuning mode is enabled.
        :debug (bool): 
            If it is True, Will enter debug mode for debugging.
        :custom_config: 
            Json file path. The file content can be used to override the default parameters.
        :kwargs: 
            Keyword parameters can be used to override default parameters
    """

    def __init__(self,
                 dataset_file_path : str,
                 dataset_desc_file_path : str,
                 val_file_path : Optional[str] = None,
                 reward_file_path : Optional[str] = None,
                 target_policy_name : str = None,
                 log_dir : str = None,
                 run_id : Optional[str] = None,
                 address : Optional[str] = None,
                 venv_mode : str = 'tune',
                 policy_mode : str = 'tune',
                 tuning_mode : str = 'None',
                 tune_initial_state : Optional[Dict[str, np.ndarray]] = None,
                 debug : bool = False,
                 revive_config_file_path  : Optional[str] = None,
                 **kwargs):
        assert policy_mode == 'None' or tuning_mode == 'None', 'Cannot perform both policy training and parameter tuning!'

        # ray.init(local_mode=True) # debug only

        ''' get config '''
        config = DEBUG_CONFIG if debug else DEFAULT_CONFIG
        parser = list2parser(config)
        self.config = parser.parse_known_args()[0].__dict__
        self.run_id = run_id or uuid.uuid4().hex
        self.workspace = os.path.abspath(os.path.join(log_dir, self.run_id))
        self.config['workspace'] = self.workspace
        os.makedirs(self.workspace, mode=0o777, exist_ok=True)
        assert os.path.exists(self.workspace)
        self.log_path = os.path.join(os.path.abspath(self.workspace),"revive.log")
        logger.add(self.log_path)

        self.revive_config_file_path = revive_config_file_path

        if revive_config_file_path is not None:
            with open(revive_config_file_path, 'r') as f:
                custom_config = json.load(f)
            self.config.update(custom_config)
            for parameter_description in custom_config.get('base_config', {}):
                self.config[parameter_description['name']] = parameter_description['default']
        else:
            self.revive_config_file_path = os.path.join(self.workspace, "config.json")
            with open(self.revive_config_file_path, 'w') as f:
                json.dump(self.config,f)

        ''' preprocess config'''


        # NOTE: in crypto mode, each trail is fixed to use one GPU.
        self.config['is_crypto'] = os.environ.get('REVIVE_CRYPTO', 0)
        setup_seed(self.config['global_seed'])

        self.venv_mode = venv_mode
        self.policy_mode = policy_mode
        self.tuning_mode = tuning_mode
        self.tune_initial_state = tune_initial_state
        
        ''' create dataset '''
        self.data_file = dataset_file_path
        self.config_file = dataset_desc_file_path
        self.val_file = val_file_path
        self.dataset = OfflineDataset(self.data_file, self.config_file, self.config['ignore_check'])
        self._check_license()
        self.runtime_env = {"env_vars": {"PYTHONPATH":os.pathsep.join(sys.path), "PYARMOR_LICENSE": sys.PYARMOR_LICENSE}}
        ray.init(address=address, runtime_env=self.runtime_env)
        if self.val_file:
            self.val_dataset = OfflineDataset(self.val_file, self.config_file, self.config['ignore_check'])
            self.val_dataset.processor = self.dataset.processor # make sure dataprocessing is the same
            self.config['val_dataset'] = ray.put(self.val_dataset)
        else: # split the training set if validation set is not provided
            self.dataset, self.val_dataset = self.dataset.split(self.config['val_split_ratio'], self.config['val_split_mode'])
            self.config['val_dataset'] = ray.put(self.val_dataset)
        self.config['dataset'] = ray.put(self.dataset)
        self.config['graph'] = self.dataset.graph
        self.graph = self.config['graph']
        if not tuning_mode == 'None': assert len(self.dataset.graph.tunable) > 0, 'No tunable parameter detected, please check the config yaml!'
        
        self.config['learning_nodes_num'] = self.dataset.learning_nodes_num

        self.reward_func = get_reward_fn(reward_file_path, self.config_file)
        self.config['user_func'] = self.reward_func
        
        if target_policy_name is None:
            target_policy_name = list(self.config['graph'].keys())[0]
            logger.info(f"target policy name [{target_policy_name}] is chosen as default")
        self.config['target_policy_name'] = target_policy_name

        ''' save a copy of the base graph '''
        with open(os.path.join(self.workspace, 'graph.pkl'), 'wb') as f:
            pickle.dump(self.config['graph'], f)

        ''' setup data buffers '''
        self.driver_ip = ray._private.services.get_node_ip_address()
        self.venv_data_buffer = ray.remote(DataBufferEnv).options(resources={f"node:{self.driver_ip}" : 0.001}).remote(venv_max_num=self.config['num_venv_store'])
        self.policy_data_buffer = ray.remote(DataBufferPolicy).options(resources={f"node:{self.driver_ip}" : 0.001}).remote()
        self.tuner_data_buffer = ray.remote(DataBufferTuner).options(resources={f"node:{self.driver_ip}" : 0.001}).remote(self.tuning_mode, self.config['parameter_tuning_budget'])
        self.config['venv_data_buffer'] = self.venv_data_buffer
        self.config['policy_data_buffer'] = self.policy_data_buffer
        self.config['tuner_data_buffer'] = self.tuner_data_buffer

        ''' try to load existing venv and policy '''
        self._reload_venv(os.path.join(self.workspace, 'env.pkl'))
        self._reload_policy(os.path.join(self.workspace, 'policy.pkl'))

        self.venv_acc = - float('inf')
        self.policy_acc = - float('inf')

        self.venv_logger = None
        self.policy_logger = None
        self.tuner_logger = None
        
        data = {"REVIVE_STOP" : False, "LOG_DIR":os.path.join(os.path.abspath(self.workspace),"revive.log")}
        with open(os.path.join(self.workspace, ".env.json"), 'w') as f:
            json.dump(data, f)

    def _reload_venv(self, path : str):
        r'''Reload a venv from the given path'''
        try:
            with open(path, 'rb') as f:
                self.venv = pickle.load(f)
            self.venv.check_version()

            if not self.graph.is_equal_structure(self.venv.graph):
                warnings.warn('Detect different graph between loaded venv and data config, it is mostly cased by change of config file, trying to rebuild ...')
                
                venv_list = []
                for _venv in self.venv.env_list:
                    graph = deepcopy(self.graph)
                    graph.copy_graph_model(_venv.graph)
                    venv_list.append(VirtualEnvDev(graph))
                self.venv = VirtualEnv(venv_list)

            ray.get(self.venv_data_buffer.set_best_venv.remote(self.venv))
        except Exception as e:
            logger.info(f"Don't load venv -> {e}")
            self.venv = None

    def _reload_policy(self, path : str):
        r'''Reload a policy from the given path'''
        try:
            with open(path, 'rb') as f:
                self.policy = pickle.load(f)
            self.policy.check_version()
            ray.get(self.policy_data_buffer.set_best_policy.remote(self.policy))
        except Exception as e:
            logger.info(f"Don't load policy -> {e}")
            self.policy = None

    def train(self, env_save_path : Optional[str] = None):
        r"""
        Train the virtual environment and policy.
        Steps
           1. Start ray worker train the virtual environment based on the data;
           2. Start ray worker train train policy based on the virtual environment.
        """

        self.train_venv()
        self.train_policy(env_save_path)
        self.tune_parameter(env_save_path)

    def train_venv(self):
        r"""
        Start ray worker train the virtual environment based on the data;
        """
        self.venv_logger = ray.remote(Logger).remote()
        self.venv_logger.update.remote(key="task_state", value="Wait")

        if self.venv_mode == 'None':
            self.venv_logger.update.remote(key="task_state", value="End")
        else:
            if 'wdist' in self.config['venv_metric']:
                self.config['max_distance'] = 2
                self.config['min_distance'] = 0
            elif 'mae' in self.config['venv_metric']:
                self.config['max_distance'] = np.log(2)
                self.config['min_distance'] = np.log(2) - 15
            elif 'mse' in self.config['venv_metric']:
                self.config['max_distance'] = np.log(4)
                self.config['min_distance'] = np.log(4) - 15
            elif 'nll' in self.config['venv_metric']:
                self.config['max_distance'] = 0.5 * np.log(2 * np.pi)
                self.config['min_distance'] = 0.5 * np.log(2 * np.pi) - 10
                
            logger.info(f"Distance is between {self.config['min_distance']} and {self.config['max_distance']}")

            if self.config["venv_algo"] == "revive":
                self.config["venv_algo"] = "revive_p"

            if self.venv_mode == 'once':
                venv_trainer = ray.remote(VenvTrain).remote(self.config, self.venv_logger, command=sys.argv[1:])
                venv_trainer.train.remote()
                # NOTE: after task finish, the actor will be automatically killed by ray, since there is no reference to it
            elif self.venv_mode == 'tune':
                self.venv_trainer = ray.remote(TuneVenvTrain).remote(self.config, self.venv_logger, command=sys.argv[1:])
                self.venv_trainer.train.remote()

    def train_policy(self, env_save_path : Optional[str] = None):
        r"""
        Start ray worker train train policy based on the virtual environment.

        Args:
            :env_save_path: virtual environments path

        .. note:: Before train policy, environment models and reward function should be provided.

        """
        if env_save_path is not None:
            self._reload_venv(env_save_path)

        self.policy_logger = ray.remote(Logger).remote()
        self.policy_logger.update.remote(key="task_state", value="Wait")

        if self.policy_mode == 'None':
            self.policy_logger.update.remote(key="task_state", value="End")
        elif self.policy_mode == 'once':
            assert self.reward_func is not None, 'policy training need reward function'
            policy_trainer = ray.remote(PolicyTrain).remote(self.config, self.policy_logger, self.venv_logger, command=sys.argv[1:])
            policy_trainer.train.remote()
            # NOTE: after task finish, the actor will be automatically killed by ray, since there is no reference to it
        elif self.policy_mode == 'tune':
            assert self.reward_func is not None, 'policy training need reward function'
            self.policy_trainer = ray.remote(TunePolicyTrain).remote(self.config, self.policy_logger, self.venv_logger, command=sys.argv[1:])
            self.policy_trainer.train.remote()

    def tune_parameter(self, env_save_path : Optional[str] = None):
        r"""
        Tune parameters on specified virtual environments.

        Args:
            :env_save_path: virtual environments path

        .. note:: This feature is currently unstable.

        """
        if env_save_path is not None:
            self._reload_venv(env_save_path)

        self.config['user_func'] = self.reward_func
        self.tuner_logger = ray.remote(Logger).remote()
        self.tuner_logger.update.remote(key="task_state", value="Wait")

        if self.tuning_mode == 'None':
            self.tuner_logger.update.remote(key="task_state", value="End")
        else:
            assert self.reward_func is not None, 'tuning parameter needs reward function'
            self.tuner = ray.remote(ParameterTuner).remote(self.config, self.tuning_mode, self.tune_initial_state, self.tuner_logger, self.venv_logger)
            self.tuner.run.remote()

    def stop_train(self) -> None:
        r"""Stop all training tasks.
        """
        _data = {"REVIVE_STOP" : True}
        with open(os.path.join(self.workspace, ".env.json"), 'w') as f:
            json.dump(_data, f)
            
        if self.venv_logger is not None:
            venv_logger = self.venv_logger.get_log.remote()
            venv_logger = ray.get(venv_logger)
            if venv_logger["task_state"] != "End":
                self.venv_logger.update.remote(key="task_state", value="Shutdown")
        if self.policy_logger is not None:
            policy_logger = self.policy_logger.get_log.remote()
            policy_logger = ray.get(policy_logger)
            if policy_logger["task_state"] != "End":
                self.policy_logger.update.remote(key="task_state", value="Shutdown")
        

    def get_virtualenv_env(self) -> Tuple[VirtualEnv, Dict[str, Union[str, float]], Dict[int, Tuple[str, str]]]:
        r"""Get virtual environment models and train log.

        :Returns: virtual environment models and train log

        """
        assert self.dataset is not None

        train_log = {}

        if self.venv_logger is not None:
            try:
                venv_logger = self.venv_logger.get_log.remote()
                venv_logger = ray.get(venv_logger)
                train_log.update({"task_state": venv_logger["task_state"],})
            except AttributeError:
                train_log.update({"task_state": "Shutdown"})

        metric = ray.get(self.venv_data_buffer.get_dict.remote())

        venv_acc = float(metric["max_acc"])
        current_num_of_trials = int(metric["num_of_trial"])
        total_num_of_trials = int(metric["total_num_of_trials"])

        train_log.update({
            "venv_acc" : venv_acc,
            "current_num_of_trials" : current_num_of_trials,
            "total_num_of_trials" : total_num_of_trials,
            
        })

        self.venv_acc = max(self.venv_acc, venv_acc)
        self.venv = ray.get(self.venv_data_buffer.get_best_venv.remote())
        best_model_workspace = ray.get(self.venv_data_buffer.get_best_model_workspace.remote())
    
        if self.venv is not None:
            with open(os.path.join(self.workspace, 'env.pkl'), 'wb') as f:
                pickle.dump(self.venv, f)

            try:
                self.venv.export2onnx(os.path.join(self.workspace, 'env.onnx'), verbose=False)
            except Exception as e:
                pass
                logger.info(f"Can't to export venv to ONNX. -> {e}")
        status_message = ray.get(self.venv_data_buffer.get_status.remote())

        return self.venv, train_log, status_message, best_model_workspace

    def get_policy_model(self) -> Tuple[PolicyModel, Dict[str, Union[str, float]], Dict[int, Tuple[str, str]]]:
        r"""Get policy based on specified virtual environments.

        :Return: policy models and train log

        """
        assert self.dataset is not None

        train_log = {}
        if self.policy_logger is not None:
            try:
                policy_logger = self.policy_logger.get_log.remote()
                policy_logger = ray.get(policy_logger)
                train_log.update({"task_state": policy_logger["task_state"],})
            except AttributeError:
                train_log.update({"task_state": "Shutdown"})

        metric = ray.get(self.policy_data_buffer.get_dict.remote())

        policy_acc = float(metric["max_reward"])
        current_num_of_trials = int(metric["num_of_trial"])
        total_num_of_trials = int(metric["total_num_of_trials"])
    
        train_log.update({
            "policy_acc" : policy_acc,
            "current_num_of_trials" : current_num_of_trials,
            "total_num_of_trials" : total_num_of_trials,
        })

        self.policy_acc = max(self.policy_acc, policy_acc)
        self.policy = ray.get(self.policy_data_buffer.get_best_policy.remote())
        best_model_workspace = ray.get(self.policy_data_buffer.get_best_model_workspace.remote())

        if self.policy is not None:
            with open(os.path.join(self.workspace, 'policy.pkl'), 'wb') as f:
                pickle.dump(self.policy, f)
            try:
                self.policy.export2onnx(os.path.join(self.workspace, 'policy.onnx'), verbose=False)
            except Exception as e:
                logger.info(f"Can't to export venv to ONNX. -> {e}")

        status_message = ray.get(self.policy_data_buffer.get_status.remote())

        return self.policy, train_log, status_message, best_model_workspace

    def get_parameter(self) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
        r"""Get tuned parameters based on specified virtual environments.

        :Return: current best parameters and training log
        """

        train_log = {}

        if self.tuner_logger is not None:
            try:
                tuner_logger = self.tuner_logger.get_log.remote()
                tuner_logger = ray.get(tuner_logger)
                train_log.update({"task_state": tuner_logger["task_state"],})
            except AttributeError:
                train_log.update({"task_state": "Shutdown"})

        metric = ray.get(self.tuner_data_buffer.get_state.remote())

        train_log.update(metric)
        self.best_parameter = train_log.pop('best_parameter')

        return self.best_parameter, train_log

    def _check_license(self):
        from revive.utils.auth_utils import check_license
        check_license(self)
