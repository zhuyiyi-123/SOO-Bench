import importlib
import json
import os
import shutil
import sys
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import revive
import torch
import yaml
from loguru import logger
from ray.util.sgd.utils import BATCH_SIZE
from revive.computation.funs_parser import parser
from revive.computation.graph import *
from revive.data.batch import Batch
from revive.data.processor import DataProcessor
from revive.utils.common_utils import find_later, load_data, plot_traj, import_model_from_file

DATADIR = os.path.abspath(os.path.join(os.path.dirname(revive.__file__), '../data/'))

class OfflineDataset(torch.utils.data.Dataset):
    r"""An offline dataset class.

    Params:
        :data_file: The file path where the training dataset is stored.
        :config_file: The file path where the data description file is stored.
        :horizon: Length of iteration trajectory.
    """

    def __init__(self, data_file : str,
                       config_file : str,
                       ignore_check : bool = False,
                       horizon : int = None):
        self.data_file = data_file
        self.config_file = config_file
        self.ignore_check = ignore_check

        self._raw_data = None
        self._raw_config = None
        self._pre_data(self.config_file, self.data_file)
        self._load_config(self.config_file)
        self._load_data(self.data_file)
        
        # pop up unused keys
        used_keys = list(self.graph.keys()) + self.graph.leaf + ['done']
        for key in list(self.data.keys()):
            if key not in used_keys:
                try:
                    self.data.pop(key)
                    self.raw_columns.pop(key)
                    self.data_configs.pop(key)
                    self.orders.pop(key)
                    warnings.warn(f'Warning: pop up unused key: {key}')
                except Exception as e:
                    logger.info(f"{e}")

        self._check_data()

        # construct the data processor
        self.processing_params = {k : self._get_process_params(self.data[k], self.data_configs[k], self.orders[k]) for k in self.data_configs.keys()}
        for curr_name, next_name in self.graph.transition_map.items():
            self.processing_params[next_name] = self.processing_params[curr_name]
            self.orders[next_name] = self.orders[curr_name]
            self.data_configs[next_name] = self.data_configs[curr_name]
        self.processor = DataProcessor(self.data_configs, self.processing_params, self.orders)
        self.graph.register_processor(self.processor)

        # separate tunable parameter from self.data
        if len(self.graph.tunable) > 0:
            self.tunable_data = Batch()
            for tunable in self.graph.tunable:
                self.tunable_data[tunable] = self.data.pop(tunable)
        else:
            self.tunable_data = None

        self.learning_nodes_num = self.graph.summary_nodes()['network_nodes']

        # by default, dataset is in trajectory mode
        self.trajectory_mode_(horizon)

    def _pre_data(self, config_file, data_file):
        # parse ts_nodes
        with open(config_file, 'r', encoding='UTF-8') as f:
            raw_config = yaml.load(f, Loader=yaml.FullLoader)
        raw_data = load_data(data_file)

        graph_dict = raw_config['metadata'].get('graph', None)
        nodes_config = raw_config['metadata'].get('nodes', None)

        if nodes_config:
            # collect nodes config
            ts_frames_config = {"ts_"+k:v["ts"] for k,v in nodes_config.items() if "ts" in v.keys()}
            ts_frames_config = {k:v for k,v in ts_frames_config.items() if v > 1}
            max_ts_frames = max(ts_frames_config.values())

            nodes = list(graph_dict.keys())
            for output_node in list(graph_dict.keys()):
                nodes += list(graph_dict[output_node])
            nodes = list(set(nodes))
            # parse nno ts_nodes
            for index, node in enumerate(nodes):
                if node.startswith("next_ts_"):
                    nodes[index] = "next_" + node[8:]
                if node.startswith("ts_"):
                    nodes[index] = node[3:]     

            ts_nodes = {"ts_"+node:node for node in nodes if "ts_"+node in ts_frames_config.keys()}
            for ts_node,node in ts_nodes.items():
                if node not in raw_data.keys():
                    logger.error(f"Can't find '{node}' node data.")
                    sys.exit()
            
            # ts_node npz data
            trj_index = [0,] + list(raw_data["index"])
            new_data = {k:[] for k in ts_nodes.keys()}
            new_index = []
            i = 1
            for trj_start_index, trj_end_index in zip(trj_index[:-1], trj_index[1:]):
                new_index.append(trj_end_index+(i*(max_ts_frames-1)))
                i += 1
                for ts_node,node in ts_nodes.items():
                    ts_node_frames = ts_frames_config[ts_node]
                    pad_data = np.concatenate([np.repeat(raw_data[node][trj_start_index:trj_start_index+1],repeats=ts_node_frames-1,axis=0), raw_data[node][trj_start_index:trj_end_index]])
                    new_data[ts_node].append(np.concatenate([pad_data[i:i+(trj_end_index-trj_start_index)] for i in range(ts_node_frames)], axis=1))
            new_data = {k:np.concatenate(v,axis=0) for k,v in new_data.items()}
            raw_data.update(new_data)

            # ts_node columns
            for ts_node, node in ts_nodes.items():
                ts_node_frames = ts_frames_config[ts_node]
                node_columns = [c for c in raw_config['metadata']['columns'] if list(c.values())[0]["dim"] == node]
                ts_node_columns = []
                ts_index = 0
                for _ in range(ts_node_frames):
                    for node_column in node_columns:
                        node_column_value = deepcopy(list(node_column.values())[0])
                        node_column_value["dim"] = ts_node
                        ts_node_columns.append({ts_node+"_"+str(ts_index):node_column_value})
                        ts_index += 1
            
                raw_config['metadata']['columns'] += ts_node_columns
            
            self._raw_config = raw_config
            self._raw_data = raw_data
            np.savez_compressed("test_ts.npz",**raw_data)

    def _check_data(self):
        '''check if the data format is correct'''

        '''1. check if the dimension of data matches the dimension described in yaml'''
        for k, v in self.raw_columns.items():
            assert k in self.data.keys(), f'Cannot find `{k}` in the data file, please check!'
            data = self.data[k]
            assert len(data.shape) == 2, f'Expect data in 2D ndarray, got variable `{k}` in shape {data.shape}, please check!'
            assert data.shape[-1] == len(v), f'Variable `{k}` described in yaml has {len(v)} dims, but got {data.shape[-1]} dims from the data file, please check!'

        for curr_name, next_name in self.graph.transition_map.items():
            assert self.data[curr_name].shape == self.data[next_name].shape, \
                f'Shape mismatch between `{curr_name}` (shape {self.data[curr_name].shape}) and `{next_name}` (shape {self.data[next_name].shape}). ' + \
                f'If it is you who puts `{next_name}` in the data file, please check the way you generate it. ' + \
                f'Otherwise, it is probably you have register a function to compute `{next_name}` but it output a wrong shape, please check the function!'

        '''2. check if the functions are correctly defined'''
        for node_name in self.graph.keys():
            node = self.graph.get_node(node_name)

            if node.node_type == 'function':
                # test 1D case
                input_data = {name : self.data[name][0] for name in node.input_names}
                should_output_data = self.data[node.name][0]
                
                if node.node_function_type == 'torch':
                    input_data = {k : torch.tensor(v) for k, v in input_data.items()}
                    should_output_data = torch.tensor(should_output_data)

                output_data = node.node_function(input_data)

                assert output_data.shape == should_output_data.shape, \
                    f'Testing function for `{node_name}`. Expect function output shape {should_output_data.shape}, got {output_data.shape} instead!'
                assert type(output_data) == type(should_output_data), \
                    f'Testing function for `{node_name}`. Expect function output type {type(should_output_data)}, got {type(output_data)} instead!'
                assert output_data.dtype == should_output_data.dtype, \
                    f'Testing function for `{node_name}`. Expect function output dtype {should_output_data.dtype}, got {output_data.dtype} instead!'

                # test 2D case
                input_data = {name : self.data[name][:2] for name in node.input_names}
                should_output_data = self.data[node.name][:2]
                
                if node.node_function_type == 'torch':
                    input_data = {k : torch.tensor(v) for k, v in input_data.items()}
                    should_output_data = torch.tensor(should_output_data)

                output_data = node.node_function(input_data)

                assert output_data.shape == should_output_data.shape, \
                    f'Testing function for `{node_name}`. Expect function output shape {should_output_data.shape}, got {output_data.shape} instead!'
                assert type(output_data) == type(should_output_data), \
                    f'Testing function for `{node_name}`. Expect function output type {type(should_output_data)}, got {type(output_data)} instead!'
                assert output_data.dtype == should_output_data.dtype, \
                    f'Testing function for `{node_name}`. Expect function output dtype {should_output_data.dtype}, got {output_data.dtype} instead!'

                # test 3D case
                input_data = {name : self.data[name][:2][np.newaxis] for name in node.input_names}
                should_output_data = self.data[node.name][:2][np.newaxis]
                
                if node.node_function_type == 'torch':
                    input_data = {k : torch.tensor(v) for k, v in input_data.items()}
                    should_output_data = torch.tensor(should_output_data)

                output_data = node.node_function(input_data)

                assert output_data.shape == should_output_data.shape, \
                    f'Testing function for `{node_name}`. Expect function output shape {should_output_data.shape}, got {output_data.shape} instead!'
                assert type(output_data) == type(should_output_data), \
                    f'Testing function for `{node_name}`. Expect function output type {type(should_output_data)}, got {type(output_data)} instead!'
                assert output_data.dtype == should_output_data.dtype, \
                    f'Testing function for `{node_name}`. Expect function output dtype {should_output_data.dtype}, got {output_data.dtype} instead!'

                # test value
                input_data = {name : self.data[name] for name in node.input_names}
                should_output_data = self.data[node.name]
                
                if node.node_function_type == 'torch':
                    input_data = {k : torch.tensor(v) for k, v in input_data.items()}

                output_data = node.node_function(input_data)

                if node.node_function_type == 'torch':
                    output_data = output_data.numpy()
                
                error = np.abs(output_data - should_output_data)

                if np.max(error) > 1e-8:
                    message = f'Test values for function "{node.name}", find max mismatch {np.max(error, axis=0)}. Please check the function.'
                    if self.ignore_check or np.max(error) < 1e-4:
                        logger.warning(message)
                    else:
                        message += '\nIf you are sure that the function is right and the value error is acceptable, configure "ignore_check=True" in the config.json to skip.'
                        message += '\nIf you are using the "train.py" script. You can add the "--ignore_check 1" to skip. E.g. python train.py --ignore_check 1'
                        logger.error(message)
                        raise ValueError(message)                    

        '''3. check if the transition variables match'''
        for curr_name, next_name in self.graph.transition_map.items():
            curr_data = []
            next_data = []
            for start, end in zip(self._start_indexes, self._end_indexes):
                curr_data.append(self.data[curr_name][start+1:end])
                next_data.append(self.data[next_name][start:end-1])
            if not np.allclose(np.concatenate(curr_data), np.concatenate(next_data), 1e-4):
                error = np.abs(np.concatenate(curr_data) - np.concatenate(next_data))
                message = f'Test transition values for {curr_name} and {next_name}, find max mismatch {np.max(error, axis=0)}. ' + \
                    f'If {next_name} is provided by you, please check the data file. If you provide {next_name} as a function, please check the function.'
                if self.ignore_check:
                    logger.warning(message)
                else:
                    logger.error(message)
                    raise ValueError(message)

    def _load_data(self, data_file : str):
        '''
            load data from the data file and conduct following processes:
            1. parse trajectory length and start and end indexes from data. 
            if `index` is not provided, consider trajectory length is equal to 1.
            2. if `done` is not in the data, create an all-zero data for it.
            3. try to compute values of unprovided node with expert function.
            4. if any transition variable is not available, truncate the trajectories by 1.
        '''
        if self._raw_data:
            raw_data = self._raw_data
        else:
            raw_data = load_data(data_file)


        # make sure data is in float32
        for k, v in raw_data.items():
            if v.dtype != np.float32:
                raw_data[k] = v.astype(np.float32)

        # mark the start and end of each trajectory
        try:
            self._end_indexes = raw_data.pop('index').astype(int)
        except:
            # if no index, consider the data is with a length of 1
            warnings.warn('key `index` is not provided, assuming data with length 1!')
            self._end_indexes = np.arange(0, raw_data[list(self.graph.keys())[0]].shape[0]) + 1
        
        # check if the index is correct
        assert np.all((self._end_indexes[1:] - self._end_indexes[:-1]) > 0), f'index must be incremental order, but got {self._end_indexes}.'
        for node_name in raw_data.keys():
            if node_name not in self.graph.tunable:
                datasize = raw_data[node_name].shape[0]
                assert datasize == self._end_indexes[-1], \
                    f'detect index exceed the provided data, the max index is {self._end_indexes[-1]}, but got {node_name} in shape {raw_data[node_name].shape}.' 
        
        self._start_indexes = np.concatenate([np.array([0]), self._end_indexes[:-1]])
        self._traj_lengths = self._end_indexes - self._start_indexes
        self._min_length = np.min(self._traj_lengths)
        self._max_length = np.max(self._traj_lengths)

        # check if `done` is defined in the data
        if not 'done' in raw_data.keys(): 
            # when done is not available, set to all zero
            warnings.warn('key `done` is not provided, set it to all zero!')
            raw_data['done'] = np.zeros((datasize, 1), dtype=np.float32)

        self.data = Batch(raw_data)

        # compute node if they are defined on the graph but not present in the data
        for node_name in self.graph.keys():
            if node_name not in self.data.keys():
                warnings.warn(f'Detect node {node_name} is not avaliable in the provided data, trying to compute it ...')
                node = self.graph.get_node(node_name)
                if node_name in self.graph.transition_map.values() and not node.node_type == 'function': continue
                assert node.node_type == 'function', \
                    f'You need to provide the function to compute node {node_name} since it is not given in the data!'
                inputs = {name : self.data[name] for name in node.input_names}
                convert_func = torch.tensor if node.node_function_type == 'torch' else np.array
                inputs = {k : convert_func(v) for k, v in inputs.items()}
                output = node.node_function(inputs)
                self.data[node_name] = np.array(output).astype(np.float32) 

        # check if all the transition variables are in the data
        need_truncate = False
        for node_name in self.graph.transition_map.values():
            if not node_name in self.data.keys():
                warnings.warn(f'transition variable {node_name} is not provided and cannot be computed!')
                need_truncate = True
        if need_truncate:
            for node_name in self.graph.transition_map.values(): # clean out existing variables 
                if node_name in self.data.keys(): 
                    self.data.pop(node_name) 
            warnings.warn('truncating the trajectory by 1 step to generate transition variables!')
            assert self._min_length > 1, 'cannot truncate trajectory with length 1'
            new_data = {k : [] for k in self.data.keys()}
            for node_name in self.graph.transition_map.values(): 
                new_data[node_name] = []
            for start, end in zip(self._start_indexes, self._end_indexes):
                for k in self.data.keys():
                    new_data[k].append(self.data[k][start:end-1])
                for curr_name, next_name in self.graph.transition_map.items():
                    new_data[next_name].append(self.data[curr_name][start+1:end])
            for k in new_data.keys(): new_data[k] = np.concatenate(new_data[k], axis=0)
            self._start_indexes -= np.arange(self._start_indexes.shape[0])
            self._end_indexes -= np.arange(self._end_indexes.shape[0]) + 1
            self._traj_lengths -= 1
            self._min_length -= 1
            self._max_length -= 1
            self.data = Batch(new_data)   

        return self.data
    
    # NOTE: mode should be set before create dataloader
    def transition_mode_(self):
        ''' Set the dataset in transition mode. `__getitem__` will return a transition. '''
        self.end_indexes = np.arange(0, self.data[list(self.graph.keys())[0]].shape[0]) + 1
        self.start_indexes = np.concatenate([np.array([0]), self.end_indexes[:-1]])
        self.traj_lengths = self.end_indexes - self.start_indexes
        self.min_length = np.min(self.traj_lengths)
        self.max_length = np.max(self.traj_lengths)  
        self.set_horizon(1)
        self.index_to_traj = [0] + list(np.cumsum(self._traj_lengths))
        self.mode = 'transition'
        self.fix_sample = True
        return self

    def trajectory_mode_(self, horizon : Optional[int] = None, fix_sample : bool = True):
        r''' Set the dataset in trajectory mode. `__getitem__` will return a clip of trajectory. '''
        self.end_indexes = self._end_indexes
        self.start_indexes = self._start_indexes
        self.traj_lengths = self._traj_lengths
        self.min_length = self._min_length
        self.max_length = self._max_length
        horizon = horizon or self.min_length
        self.set_horizon(horizon)
        self.index_to_traj = [0] + list(np.cumsum(self.traj_lengths // self.horizon))
        self.mode = 'trajectory'
        self.fix_sample = fix_sample
        return self

    def _find_trajectory(self, index : int):
        ''' perform binary search for the index of true trajectories from the index of the sample trajectory '''
        left, right = 0, len(self.index_to_traj) - 1
        mid = (left + right) // 2
        while not (index >= self.index_to_traj[mid] and index < self.index_to_traj[mid+1]):
            if index < self.index_to_traj[mid]:
                right = mid - 1
            else:
                left = mid + 1
            mid = (left + right) // 2
        return mid

    def set_horizon(self, horizon : int):
        r''' Set the horzion for loading data '''
        if horizon > self.min_length:
            logger.warning(f'Warning: the min length of dataset is {self.min_length}, which is less than the horzion {horizon} you require. ' + \
                          f'Fallback to use horzion = {self.min_length}.')
        self.horizon = min(horizon, self.min_length)

    def get_dist_configs(self, model_config):
        r'''
        Get the config of distributions for each node based on the given model config.

        Args:
            :model_config: The given model config.


        Return:
            :dist_configs: config of distributions for each node.
            :total_dims: dimensions for each node when it is considered as input and output. 
                        (Output dimensions can be different from input dimensions due to the parameterized distribution)
        '''
        dist_configs = {k : self._get_dist_config(self.data_configs[k], model_config) for k in self.data_configs.keys()}
        total_dims = {k : self._get_dim(dist_configs[k]) for k in dist_configs.keys()}
        return dist_configs, total_dims

    def _load_config(self, config_file : str):
        """
            load data description from `.yaml` file. Few notes:
            1. the name of each dimension will be discarded since they doesn't help the computation.
            2. dimensions of each node will be reordered (category, discrete, continuous) to speed up computation.
            3. register expert functions and tunable parameters if defined.
        """
        if self._raw_config:
            raw_config = self._raw_config
        else:
            with open(config_file, 'r', encoding='UTF-8') as f:
                raw_config = yaml.load(f, Loader=yaml.FullLoader)

        # collect description for the same node
        data_config = raw_config['metadata']['columns']
        self.columns = data_config
        keys = set([list(d.values())[0]['dim'] for d in data_config])

        raw_columns = {}
        config = {}
        order = {}
        fit = {}
        

        for config_key in keys:
            raw_columns[config_key], config[config_key], order[config_key], fit[config_key] = self._load_config_for_single_node(data_config, config_key)

        # parse graph
        graph_dict = raw_config['metadata'].get('graph', None)
        metric_nodes = raw_config['metadata'].get('metric_nodes', None)

        graph = DesicionGraph(graph_dict, raw_columns, fit, metric_nodes)
        # copy the raw columns for transition variables to allow them as input to other nodes
        for curr_name, next_name in graph.transition_map.items():
            raw_columns[next_name] = raw_columns[curr_name]

        # mark tunable parameters
        for node_name in raw_config['metadata'].get('tunable', []): graph.mark_tunable(node_name)
        
        expert_functions = raw_config['metadata'].get('expert_functions', None)
        custom_nodes = raw_config['metadata'].get('custom_nodes', None)

        def get_function_type(file_path : str, function_name : str) -> str:
            '''get the function type from type hint'''
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    if line.startswith('def ') and function_name in line:
                        if 'Tensor' in line:
                            return 'torch'
                        elif 'ndarray' in line:
                            return 'numpy'
                        else:
                            warnings.warn('Type hint is not provided, assume it is an numpy function!')
                            return 'numpy'
            raise ValueError(f'Cannot find function {function_name} in {file_name}.py, please check your yaml!')
        
        later = find_later(self.config_file, 'data')
        head = '.'.join(later[:-1])
        # register expert functions to the graph
        if expert_functions is not None:
            for node_name, function_description in expert_functions.items():
                # NOTE: currently we assume the expert functions are also placed in the same folder as the yaml file.
                if 'node_function' in function_description.keys(): # `node function` should be like [file].[function_name]`
                    graph.register_node(node_name, FunctionDecisionNode)
                    file_name, function_name = function_description['node_function'].split('.')
                    file_path = os.path.join(os.path.dirname(self.config_file), file_name + '.py')
                    function_type = get_function_type(file_path, function_name)
                    parse_file_path = file_path[:-3]+"_parsed.py"
                    if not parser(file_path,parse_file_path,self.config_file):
                        parse_file_path = file_path
                    function_type = get_function_type(parse_file_path, function_name)
                    file_name = os.path.split(os.path.splitext(parse_file_path)[0])[-1]
                    sys.path.insert(0, os.path.dirname(parse_file_path))
                    source_file = importlib.import_module(f'{file_name}')
                    func = eval(f'source_file.{function_name}')
                    graph.get_node(node_name).register_node_function(func, function_type)
                    logger.info(f'register node function ({function_type} version) for {node_name}')

        # register custom nodes to the graph
        if custom_nodes is not None:
            for node_name, custom_node in custom_nodes.items():
                # NOTE: currently we assume the custom nodes are also placed in the same folder as the yaml file.
                # `custom node should be given in the form of [file].[node_class_name]`
                file_name, class_name = custom_node.split('.')
                source_file = importlib.import_module(f'{head}.{file_name}')
                node_class = eval(f'source_file.{class_name}')
                graph.register_node(node_name, node_class)
                logger.info(f'register custom node `{node_class}` for {node_name}')

        # register other nodes with the default `NetworkDecisionNode`
        for node_name, node in graph.nodes.items():
            if node is None:
                graph.register_node(node_name, NetworkDecisionNode)
                logger.info(f'register the default `NetworkDecisionNode` for {node_name}')

        self.raw_columns, self.data_configs, self.orders, self.graph, self.fit = raw_columns, config, order, graph, fit

    def _load_config_for_single_node(self, raw_config : list, node_name : str):
        '''
            load config for a single node. 
            :return 
                raw_config: columns belong to this node
                config: collected columns with type in order: category, discrete, continuous 
                order: order of the index to convert to the collected columns
        '''
        raw_config = list(filter(lambda d: list(d.values())[0]['dim'] == node_name, raw_config))      

        config = []

        discrete_count = 0
        discrete_min = []
        discrete_max = []
        discrete_num = []
        fit = []

        continuous_count = 0
        continuous_min = []
        continuous_max = []

        order = []

        for index, d in enumerate(raw_config):
            name = list(d.keys())[0]
            _config = d[name]

            if _config['type'] == 'category':
                assert 'values' in _config.keys(), f'Parsing columns for node `{node_name}`, you must provide `values` for a `category` dimension.'
                _config['dim'] = len(_config['values'])
                order.append((index, 1))
                config.append(_config)
            elif _config['type'] == 'continuous':
                order.append((index, 3))
                continuous_count += 1
                continuous_max.append(_config.get('max', None))
                continuous_min.append(_config.get('min', None))
            elif _config['type'] == 'discrete':
                assert 'num' in _config.keys() and _config['num'] > 1, f'Parsing columns for node `{node_name}`, you must provide `num` > 1 for a `discrete` dimension.'
                discrete_count += 1 
                order.append((index, 2))
                discrete_max.append(_config.get('max', None))
                discrete_min.append(_config.get('min', None))
                discrete_num.append(_config['num'])
            else:
                logger.error(f"Data type {_config['type']} is not support. Please check the yaml file.")
                raise NotImplementedError

            if "fit" in _config.keys() and not _config["fit"]:
                if _config['type'] == 'category':
                    fit += [False,]*len(_config['values'])
                else:
                    fit.append(False)
            else:
                if _config['type'] == 'category':
                    fit += [True,]*len(_config['values'])
                else:
                    fit.append(True)

        order = sorted(order, key=lambda x: x[1])
        forward_order = [o[0] for o in order]
        order = [(ordered_index, origin_index) for ordered_index, origin_index in enumerate(forward_order)]
        order = sorted(order, key=lambda x: x[1])
        backward_order = [o[0] for o in order]
        order = {
            'forward' : forward_order,
            'backward' : backward_order,
        }
        
        if discrete_count > 0:
            config.append({'type' : 'discrete', "dim" : discrete_count, 'max' : discrete_max, 'min' : discrete_min, 'num' : discrete_num})
        
        if continuous_count > 0:
            config.append({'type' : 'continuous', 'dim' : continuous_count, 'max' : continuous_max, 'min' : continuous_min})

        return raw_config, config, order, fit

    def _get_dist_config(self, data_config : List[Dict[str, Any]], model_config : Dict[str, Any]):
        dist_config = []
        for config in data_config:
            config = config.copy()
            if config['type'] == 'category':
                assert model_config['category_distribution_type'] in ['onehot'], \
                    f"distribution type {model_config['category_distribution_type']} is not support for category variables!"
                config['type'] = model_config['category_distribution_type']
                config['output_dim'] = config['dim']
            elif config['type'] == 'discrete':
                assert model_config['discrete_distribution_type'] in ['gmm', 'normal', 'discrete_logistic'], \
                    f"distribution type {model_config['discrete_distribution_type']} is not support for discrete variables!"
                config['type'] = model_config['discrete_distribution_type']
                if config['type'] == 'discrete_logistic':
                    config['output_dim'] = config['dim'] * 2
                    config['num'] = config['num']
                elif config['type'] == 'normal':
                    config['conditioned_std'] = model_config['conditioned_std']
                    config['output_dim'] = config['dim'] * (1 + config['conditioned_std'])
                elif config['type'] == 'gmm':
                    config['mixture'] = model_config['mixture']
                    config['conditioned_std'] = model_config['conditioned_std']
                    config['output_dim'] = config['mixture'] * ((1 + config['conditioned_std']) * config['dim'] + 1)
            else:
                assert model_config['continuous_distribution_type'] in ['gmm', 'normal'], \
                    f"distribution type {model_config['continuous_distribution_type']} is not support for discrete variables!"
                config['type'] = model_config['continuous_distribution_type']
                if config['type'] == 'normal':
                    config['conditioned_std'] = model_config['conditioned_std']
                    config['output_dim'] = config['dim'] * (1 + config['conditioned_std'])
                elif config['type'] == 'gmm':
                    config['mixture'] = model_config['mixture']
                    config['conditioned_std'] = model_config['conditioned_std']
                    config['output_dim'] = config['mixture'] * ((1 + config['conditioned_std']) * config['dim'] + 1)
            dist_config.append(config)
        return dist_config

    def _get_process_params(self, data : np.ndarray, data_config : Dict[str, Union[int, str, List[float]]], order : List[int]):
        ''' get necessary parameters for data processor '''
        data = data.copy()
        data = data.take(order['forward'], axis=-1)

        additional_parameters = []
        forward_slices = []
        backward_slices = []
        forward_start_index = 0
        backward_start_index = 0

        for config in data_config:
            if config['type'] == 'category':
                forward_end_index = forward_start_index + 1
                additional_parameters.append(np.array(config['values']).astype(np.float32))
            elif config['type'] == 'continuous':
                forward_end_index = forward_start_index + config['dim']
                _data = data[:, forward_start_index : forward_end_index]
                _data = _data.reshape((-1, _data.shape[-1]))
                data_max = _data.max(axis=0)
                data_min = _data.min(axis=0)
                for i in range(config['dim']):
                    if config['max'][i] is None: config['max'][i] = data_max[i]
                    if config['min'][i] is None: config['min'][i] = data_min[i]
                max_num = np.array(config['max']).astype(np.float32)
                min_num = np.array(config['min']).astype(np.float32)
                interval = max_num - min_num
                interval[interval==0] = 2 # prevent dividing zero
                additional_parameters.append(((max_num + min_num) / 2, 0.5 * interval))
            elif config['type'] == 'discrete':
                forward_end_index = forward_start_index + config['dim']
                _data = data[:, forward_start_index : forward_end_index]
                _data = _data.reshape((-1, _data.shape[-1]))
                data_max = _data.max(axis=0)
                data_min = _data.min(axis=0)
                for i in range(config['dim']):
                    if config['max'][i] is None: config['max'][i] = data_max[i]
                    if config['min'][i] is None: config['min'][i] = data_min[i]
                max_num = np.array(config['max']).astype(np.float32)
                min_num = np.array(config['min']).astype(np.float32)
                interval = max_num - min_num
                interval[interval==0] = 2 # prevent dividing zero
                additional_parameters.append(((max_num + min_num) / 2, 0.5 * interval, np.array(config['num'])))

            backward_end_index = backward_start_index + config['dim']
            forward_slices.append(slice(forward_start_index, forward_end_index))
            backward_slices.append(slice(backward_start_index, backward_end_index))
            forward_start_index = forward_end_index
            backward_start_index = backward_end_index

        return {
            'forward_slices' : forward_slices,
            'backward_slices' : backward_slices,
            'additional_parameters' : additional_parameters,
        }

    def _get_dim(self, dist_configs):
        return {
            'input' : sum([d['dim'] for d in dist_configs]),
            'output' : sum([d['output_dim'] for d in dist_configs])
        }

    def __len__(self) -> int:
        return np.sum(self.traj_lengths // self.horizon)

    def __getitem__(self, index : int, raw : bool = False) -> Batch:
        if self.mode == 'trajectory':
            traj_index = self._find_trajectory(index)
            if self.fix_sample: # fix the starting point of each slice in the trajectory
                start_index = self.start_indexes[traj_index] + self.horizon * (index - self.index_to_traj[traj_index])
            else: # randomly sample valid start point from the trajectory
                length = self.end_indexes[traj_index] - self.start_indexes[traj_index]
                start_index = self.start_indexes[traj_index] + np.random.randint(0, length - self.horizon + 1)
            raw_data = self.data[start_index : (start_index + self.horizon)]
            if self.tunable_data is not None:
                raw_data.update(self.tunable_data[traj_index])
                # tunable_data = Batch()
                # for tunable, data in self.tunable_data.items():
                #     tunable_data[tunable] = data[traj_index][np.newaxis].repeat(self.horizon, axis=0)
                # raw_data.update(tunable_data)
        elif self.mode == 'transition':
            raw_data = self.data[index]
            if self.tunable_data is not None:
                traj_index = self._find_trajectory(index)
                raw_data.update(self.tunable_data[traj_index])
        if raw:
            return raw_data
        return self.processor.process(raw_data)

    def split(self, ratio : float = 0.5, mode : str = 'outside_traj', recall : bool = False) -> Tuple['OfflineDataset', 'OfflineDataset']:
        r''' split the dataset into train and validation with the given ratio and mode
        
        Args:
            :ratio: Ratio to split validate dataset if it is not explicitly given.
            :mode: Mode of auto splitting training and validation dataset, choose from `outside_traj` and `inside_traj`. 
                  `outside_traj` means the split is happened outside the trajectories, one trajectory can only be in one dataset. ' +
                  `inside_traj` means the split is happened inside the trajectories, former part of one trajectory is in training set, later part is in validation set.

        Return: 
            (TrainDataset, ValidateDataset)
        
        '''
        val_dataset = deepcopy(self)

        if mode == 'outside_traj':
            total_traj_num = len(self._start_indexes)
            val_traj_num = int(total_traj_num * ratio)
            train_traj_num = total_traj_num - val_traj_num

            if not (val_traj_num > 0 and train_traj_num > 0):
                message = f'Cannot split a dataset with {total_traj_num} trajectories to {train_traj_num} (training) and {val_traj_num} (validation).'
                if recall:
                    raise RuntimeError(message)
                else:
                    warnings.warn(message)
                    warnings.warn('Fallback to `inside_traj` mode!')
                    return self.split(ratio=ratio, mode='inside_traj', recall=True)

            total_traj_index = list(range(total_traj_num))
            np.random.shuffle(total_traj_index)
            val_traj_index = sorted(total_traj_index[:val_traj_num])
            train_traj_index = sorted(total_traj_index[val_traj_num:])

            self._rebuild_from_traj_index(train_traj_index)
            val_dataset._rebuild_from_traj_index(val_traj_index)

        elif mode == 'inside_traj':
            training_slices = []
            validation_slices = []

            for total_point in self._traj_lengths:
                slice_point = int(total_point * (1 - ratio))
                train_point = slice_point
                validation_point = total_point - slice_point

                if not (train_point > 0 and validation_point > 0):
                    message = f'Cannot split a trajectory with {total_point} steps to {train_point} (training) and {validation_point} (validation).'
                    if recall:
                        raise RuntimeError(message)
                    else:
                        warnings.warn(message)
                        warnings.warn('Fallback to `outside_traj` mode!')
                        return self.split(ratio=ratio, mode='outside_traj', recall=True)

                training_slices.append(slice(0, slice_point))
                validation_slices.append(slice(slice_point, total_point))

            self._rebuild_from_slices(training_slices)
            val_dataset._rebuild_from_slices(validation_slices)    
        else:
            raise ValueError(f'Split mode {mode} is not understood, please check your config!')
        
        return self, val_dataset

    def _rebuild_from_traj_index(self, traj_indexes : List[int]) -> 'OfflineDataset':
        ''' rebuild the dataset by subsampling the trajectories '''

        # rebuild data
        new_data = []
        for traj_index in traj_indexes:
            start = self._start_indexes[traj_index]
            end = self._end_indexes[traj_index]
            new_data.append(self.data[start:end])
        self.data = Batch.cat(new_data)

        # rebuild index
        self._traj_lengths = self._end_indexes[traj_indexes] - self._start_indexes[traj_indexes]
        self._end_indexes = np.cumsum(self._traj_lengths)
        self._start_indexes = np.concatenate([np.array([0]), self._end_indexes[:-1]])
        self._min_length = np.min(self._traj_lengths)
        self._max_length = np.max(self._traj_lengths)

        self.trajectory_mode_() if self.mode == 'trajectory' else self.transition_mode_()

        return self

    def _rebuild_from_slices(self, slices : List[slice]) -> 'OfflineDataset':
        ''' rebuild the dataset by slicing inside the trajectory '''

        assert len(self._traj_lengths) == len(slices)

        # rebuild data
        new_data = []
        lengths = []
        for s, start, end in zip(slices, self._start_indexes, self._end_indexes):
            data = self.data[start:end][s]
            lengths.append(data.shape[0])
            new_data.append(data)
        self.data = Batch.cat(new_data)

        # rebuild index
        self._traj_lengths = np.array(lengths)
        self._end_indexes = np.cumsum(self._traj_lengths)
        self._start_indexes = np.concatenate([np.array([0]), self._end_indexes[:-1]])
        self._min_length = np.min(self._traj_lengths)
        self._max_length = np.max(self._traj_lengths)

        self.trajectory_mode_() if self.mode == 'trajectory' else self.transition_mode_()

        return self

class UniformSampler(torch.utils.data.Sampler):
    r""" A uniform data sampler

    Args:
        data_source (OfflineDataset): dataset to sample from
        num_samples (int): number of samples to draw.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
    """
    def __init__(self, data_source, number_samples, replacement=True):
        self.data_source = data_source
        self.number_samples = number_samples
        self.replacement = replacement

        if not self.replacement:
            assert self.number_samples <= len(self.data_source), \
                "Cannot draw more samples than dataset itself, consider to decrease batch size or set `replacement=True`"

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            rand_tensor = torch.randint(high=n, size=(self.number_samples,), dtype=torch.int64)
            sample_index = rand_tensor.tolist()
        else:
            sample_index = torch.randperm(n).tolist()[:self.number_samples]

        return iter([sample_index])

    def __len__(self):
        return 1

class InfiniteUniformSampler(torch.utils.data.Sampler):
    r""" A infinite data sampler, sampler that provides infinite length of data index

    Args:
        data_source (OfflineDataset): dataset to sample from
        num_samples (int): number of samples to draw.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
    """
    def __init__(self, data_source, number_samples, replacement=True):
        self.data_source = data_source
        self.number_samples = number_samples
        self.replacement = replacement

        if not self.replacement:
            assert self.number_samples <= len(self.data_source), \
                "Cannot draw more samples than dataset itself, consider to decrease batch size or set `replacement=True`"

    def __iter__(self):
        n = len(self.data_source)
        while True:
            if self.replacement:
                rand_tensor = torch.randint(high=n, size=(self.number_samples,), dtype=torch.int64)
                sample_index = rand_tensor.tolist()
            else:
                sample_index = torch.randperm(n).tolist()[:self.number_samples]

            yield sample_index

class InfiniteDataLoader:
    r""" Wrapper that enables infinite pre-fetching, must use together with InfiniteUniformSampler"""
    def __init__(self, dataloader : torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.iter = iter(self.dataloader)

    def __iter__(self):
        return iter([next(self.iter)])

def collect_data(expert_data : List[Batch], graph : DesicionGraph) -> Batch:
    r''' Collection function for PyTorch DataLoader '''
    expert_data = Batch.stack(expert_data, axis=-2)
    expert_data.to_torch()
    if graph.transition_map:
        selected_name = list(graph.transition_map.keys())[0]
        if len(expert_data[selected_name].shape) == 3:
            for tunable_name in graph.tunable:
                expert_data[tunable_name] = expert_data[tunable_name].expand(expert_data[selected_name].shape[0], *[-1] * len(expert_data[tunable_name].shape))
    return expert_data

def get_loader(dataset : OfflineDataset, config : dict, is_sample : bool = True):
    """ Get the PoTorch DataLoader for training """
    batch_size = config[BATCH_SIZE]
    if is_sample:
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=InfiniteUniformSampler(dataset, batch_size),
                                             collate_fn=partial(collect_data, graph=dataset.graph), pin_memory=True, num_workers=config['data_workers'])
        loader = InfiniteDataLoader(loader)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=partial(collect_data, graph=dataset.graph), pin_memory=True, num_workers=config['data_workers'])
    return loader

def data_creator(config : dict, 
                 training_mode : str = 'trajectory', 
                 training_horizon : int = None,
                 training_is_sample : bool = True,
                 val_mode : str = 'trajectory',
                 val_horizon : int = None,
                 val_is_sample : bool = False,
                 double : bool = False):
    """
        Get train data loader and validation data loader.

        :return: train data loader and validation data loader
    """
    train_dataset = ray.get(config['dataset'])
    val_dataset = ray.get(config['val_dataset'])

    config['dist_configs'], config['total_dims'] = train_dataset.get_dist_configs(config)
    config['learning_nodes_num'] = train_dataset.learning_nodes_num

    if training_horizon is None and val_horizon is not None:
        training_horizon = val_horizon
    if training_horizon is not None and val_horizon is None:
        val_horizon = training_horizon
    
    if not double:
        train_dataset = train_dataset.trajectory_mode_(training_horizon) if training_mode == 'trajectory' else train_dataset.transition_mode_()
        val_dataset = val_dataset.trajectory_mode_(val_horizon) if val_mode == 'trajectory' else val_dataset.transition_mode_()

        train_loader = get_loader(train_dataset, config, training_is_sample)
        val_loader = get_loader(val_dataset, config, val_is_sample)

        return train_loader, val_loader
    else: # perform double venv training
        ''' NOTE: train_dataset_val means training set used in validation '''
        train_dataset_train = deepcopy(train_dataset)
        val_dataset_train = deepcopy(val_dataset)

        train_dataset_val = deepcopy(train_dataset)
        val_dataset_val = deepcopy(val_dataset)

        train_dataset_train = train_dataset_train.trajectory_mode_(training_horizon) if training_mode == 'trajectory' else train_dataset_train.transition_mode_()
        val_dataset_train = val_dataset_train.trajectory_mode_(training_horizon) if training_mode == 'trajectory' else val_dataset_train.transition_mode_()
        train_dataset_val = train_dataset_val.trajectory_mode_(val_horizon) if val_mode == 'trajectory' else train_dataset_val.transition_mode_()
        val_dataset_val = val_dataset_val.trajectory_mode_(val_horizon) if val_mode == 'trajectory' else val_dataset_val.transition_mode_()

        train_loader_train = get_loader(train_dataset_train, config, training_is_sample)
        val_loader_train = get_loader(val_dataset_train, config, training_is_sample)
        train_loader_val = get_loader(train_dataset_val, config, val_is_sample)
        val_loader_val = get_loader(val_dataset_val, config, val_is_sample)

        return train_loader_train, val_loader_train, train_loader_val, val_loader_val


if __name__ == '__main__':
    dataset = OfflineDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, collate_fn=partial(Batch.stack, axis=1), shuffle=True)

    data = next(iter(loader))
    print(data)
    print(data[:, 0].obs)
    
    single_data = dataset.__getitem__(np.random.randint(len(dataset)), raw=True)
    processed_data = dataset.processor.process(single_data)
    deprocessed_data = dataset.processor.deprocess(processed_data)

    processor = dataset.processor
    processed_obs = processor.process_single(single_data.obs, 'obs')
    deprocessed_obs = processor.deprocess_single(processed_obs, 'obs')

    for k in single_data.keys():
        assert np.all(np.isclose(deprocessed_data[k], single_data[k], atol=1e-6)), [k, deprocessed_data[k] - single_data[k]]
    assert np.all(np.isclose(deprocessed_data.obs, deprocessed_obs, atol=1e-6)), [processed_data.obs - processed_obs]

    plot_traj(single_data)

    # test sampler
    data = torch.rand(1000, 4)
    dataset = torch.utils.data.TensorDataset(data)
    sampler = UniformSampler(dataset, 3)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    for _ in range(10):
        for b in loader:
            print(b)
