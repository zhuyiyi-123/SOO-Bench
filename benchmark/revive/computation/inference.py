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

import torch
import warnings
import numpy as np
from copy import deepcopy
from typing import Callable, Dict, List, Union, Optional

from revive import __version__
from revive.computation.graph import DesicionGraph, DesicionNode
from revive.computation.utils import *


class VirtualEnvDev(torch.nn.Module):
    def __init__(self, graph : DesicionGraph) -> None:
        super(VirtualEnvDev, self).__init__()
        self.models = torch.nn.ModuleList()
        self.graph = graph
        for node in self.graph.nodes.values():
            if node.node_type == 'network':
                self.models.append(node.get_network())
        self.set_target_policy_name(list(self.graph.keys())) # default
        self.revive_version = __version__
        self.device = "cpu"

    def to(self, device):
        if device != self.device:
            self.device = device
            for node in self.graph.nodes.values():
                if node.node_type == 'network':
                    node.to(self.device)

    def check_version(self):
        if not self.revive_version == __version__:
            warnings.warn(f'detect the venv is create by version {self.revive_version}, but current version is {__version__}, maybe not compactable.')

    def reset(self) -> None:
        self.graph.reset()

    def set_target_policy_name(self, target_policy_name : list) -> None:
        self.target_policy_name = target_policy_name

        # find target index
        self.index = []
        for i, (output_name, input_names) in enumerate(self.graph.items()):
            if output_name in self.target_policy_name:
                self.index.append(i)
        self.index.sort()
        

    def _data_preprocess(self, data : np.ndarray, data_key : str = "obs") -> torch.Tensor:
        data = self.graph.processor.process_single(data, data_key)
        data = to_torch(data, device=self.device)

        return data

    def _data_postprocess(self, data : torch.Tensor, data_key : str) -> np.ndarray:
        data = to_numpy(data)
        data = self.graph.processor.deprocess_single(data, data_key)
        
        return data

    def _infer_one_step(self, 
                        state : Dict[str, np.ndarray], 
                        deterministic : bool = True,
                        clip : bool = True) -> Dict[str, np.ndarray]:

        self.check_version()
        state = deepcopy(state)
        
        sample_fn = get_sample_function(deterministic)
        
        for k in list(state.keys()):
            state[k] = self._data_preprocess(state[k], k)

        for node_name in self.graph.keys():
            if not node_name in state.keys(): # skip provided values
                output = self.graph.compute_node(node_name, state)
                if isinstance(output, torch.Tensor):
                    state[node_name] = output
                else:
                    state[node_name] = sample_fn(output)
                if clip: state[node_name] = torch.clamp(state[node_name], -1, 1)

        for k in list(state.keys()):
            state[k] = self._data_postprocess(state[k], k)

        return state

    def infer_k_steps(self, 
                      states : List[Dict[str, np.ndarray]], 
                      deterministic : bool = True,
                      clip : bool = True) -> List[Dict[str, np.ndarray]]:
        
        outputs = []
        backup = {}
        tunable = {name : states[0][name] for name in self.graph.tunable}

        for state in states:
            state.update(backup)
            output = self._infer_one_step(state, deterministic=deterministic, clip=clip)
            outputs.append(output)
            backup = self.graph.state_transition(output)
            backup.update(tunable)

        return outputs

    def infer_one_step(self, 
                       state : Dict[str, np.ndarray], 
                       deterministic : bool = True,
                       clip : bool = True) -> Dict[str, np.ndarray]:
        return self._infer_one_step(state, deterministic=deterministic, clip=clip)

    def forward(self, 
                data : Dict[str, torch.Tensor], 
                deterministic : bool = True,
                clip : bool = False) -> Dict[str, torch.Tensor]:
        ''' run the target node '''
        self.check_version()

        sample_fn = get_sample_function(deterministic)

        node_name = self.target_policy_name
        output = self.graph.compute_node(node_name, data)
        
        if isinstance(output, torch.Tensor):
            data[node_name] = output
        else:
            data[node_name] = sample_fn(output)
        if clip: data[node_name] = torch.clamp(data[node_name], -1, 1)

        return data

    def pre_computation(self, 
                        data : Dict[str, torch.Tensor], 
                        deterministic : bool = True,
                        clip : bool = False,
                        policy_index : int = 0) -> Dict[str, torch.Tensor]:
        '''run all the node before target node. skip if the node value is already available.'''
        self.check_version()

        sample_fn = get_sample_function(deterministic)

        for node_name in list(self.graph.keys())[:self.index[policy_index]]:
            if not node_name in data.keys():
                output = self.graph.compute_node(node_name, data)
                if isinstance(output, torch.Tensor):
                    data[node_name] = output
                else:
                    data[node_name] = sample_fn(output)
                if clip: data[node_name] = torch.clamp(data[node_name], -1, 1)
            else:
                print(f'Skip {node_name}, since it is provided in the inputs.')

        return data

    def post_computation(self, 
                         data : Dict[str, torch.Tensor], 
                         deterministic : bool = True,
                         clip : bool = False,
                         policy_index : int = 0) -> Dict[str, torch.Tensor]:
        '''run all the node after target node'''
        self.check_version()

        sample_fn = get_sample_function(deterministic)

        for node_name in list(self.graph.keys())[self.index[policy_index]+1:]:
            output = self.graph.compute_node(node_name, data)
            if isinstance(output, torch.Tensor):
                data[node_name] = output
            else:
                data[node_name] = sample_fn(output)
            if clip: data[node_name] = torch.clamp(data[node_name], -1, 1)

        return data

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        self.graph.export2onnx(onnx_file, verbose)

class VirtualEnv:
    def __init__(self, env_list : List[VirtualEnvDev]):
        self._env = env_list[0]
        self.env_list = env_list
        self.graph = self._env.graph
        self.revive_version = __version__
        self.device = "cpu"

    def to(self, device):
        r"""
        Move model to the device specified by the parameter.

        Examples::

            >>> venv_model.to("cpu")
            >>> venv_model.to("cuda")
            >>> venv_model.to("cuda:1")

        """
        if device != self.device:
            self.device = device
            for env in self.env_list:
                env.to(device)

    def check_version(self):
        r"""Check if the revive version of the saved model and the current revive version match."""
        if not self.revive_version == __version__:
            warnings.warn(f'detect the venv is create by version {self.revive_version}, but current version is {__version__}, maybe not compactable.')

    def reset(self) -> None:
        r"""
            When using RNN for model training, this method needs to be called before model reuse 
            to reset the hidden layer information.
        """
        for env in self.env_list:
            env.reset()

    @property
    def target_policy_name(self) -> str:
        r''' Get the target policy name. '''
        return self._env.target_policy_name

    def set_target_policy_name(self, target_policy_name) -> None:
        r''' Set the target policy name. '''
        for env in self.env_list:
            env.set_target_policy_name(target_policy_name)

    def replace_policy(self, policy : 'PolicyModel') -> None:
        ''' Replace the target policy with the given policy. '''
        assert self.target_policy_name == policy.target_policy_name, \
            f'policy name does not match, require {self.target_policy_name} but get {policy.target_policy_name}!'
        for env in self.env_list:
            env.graph.nodes[self.target_policy_name] = policy._policy_model.node

    @torch.no_grad()
    def infer_k_steps(self, 
                      states : Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]], 
                      k : Optional[int] = None, 
                      deterministic : bool = True,
                      clip : bool = True) -> List[Dict[str, np.ndarray]]:
        r"""
        Generate k steps interactive data.

        Args:
            :states: a dict of initial input nodes

            :k: how many steps to generate

            :deterministic: 
                if True, the most likely actions are generated; 
                if False, actions are generated by sample.
                Default: True

        Return: 
            k steps interactive data dict

        Examples::

            >>> state = {"obs": obs_array, "static_obs": static_obs_array}
            >>> ten_step_output = venv_model.infer_k_steps(state, k=10)

        """
        self.check_version()

        if isinstance(states, dict):
            states = [states]
            if k is not None:
                for i in range(k - 1):
                    states.append({})

        return self._env.infer_k_steps(deepcopy(states), deterministic, clip=clip)

    @torch.no_grad()
    def infer_one_step(self, 
                       state : Dict[str, np.ndarray], 
                       deterministic : bool = True,
                       clip : bool = True) -> Dict[str, np.ndarray]:
        r"""
        Generate one step interactive data given action.

        Args:
            :state: a dict of input nodes

            :deterministic: 
                if True, the most likely actions are generated; 
                if False, actions are generated by sample.
                Default: True

        Return: 
            one step outputs

        Examples::

            >>> state = {"obs": obs_array, "static_obs": static_obs_array}
            >>> one_step_output = venv_model.infer_one_step(state)

        """        
        self.check_version()
        return self._env.infer_one_step(deepcopy(state), deterministic, clip=clip)

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        r"""
        Exporting the model to onnx mode.

        Reference: https://pytorch.org/docs/stable/onnx.html

        Args:
            :onnx_file: the onnx model file save path.

            :verbose: if True, prints a description of the model being exported to stdout. 
                      In addition, the final ONNX graph will include the field ``doc_string``` 
                      from the exported model which mentions the source code locations for ``model``.

        """
        self._env.export2onnx(onnx_file, verbose)

class PolicyModelDev(torch.nn.Module):
    def __init__(self, nodes : List[DesicionNode,]):
        super().__init__()
        self.nodes = nodes
        self.node = self.nodes[0]
        self.models = [node.get_network() for node in self.nodes]
        self.target_policy_name = [node.name for node in self.nodes]
        self.target_policy_name = self.target_policy_name[0]
        self.revive_version = __version__
        self.device = "cpu"

    def to(self, device):
        if device != self.device:
            self.device = device
            self.node.to(self.device)

    def check_version(self):
        if not self.revive_version == __version__:
            warnings.warn(f'detect the policy is create by version {self.revive_version}, but current version is {__version__}, maybe not compactable.')

    def reset(self):
        for node in self.nodes:
            node.reset()

    def _data_preprocess(self, data : np.ndarray, data_key : str = "obs") -> torch.Tensor:
        data = self.node.processor.process_single(data, data_key)
        data = to_torch(data, device=self.device)

        return data

    def _data_postprocess(self, data : torch.tensor, data_key : str = "action1") -> np.ndarray:
        data = to_numpy(data)
        data = self.node.processor.deprocess_single(data, data_key)

        return data

    def infer(self, state : Dict[str, np.ndarray], deterministic : bool = True, clip : bool = False) -> np.ndarray:
        self.check_version()
        state = deepcopy(state)

        sample_fn = get_sample_function(deterministic)

        for k, v in state.items():
            state[k] = self._data_preprocess(v, data_key=k)
        
        output = self.node(state)
        if isinstance(output, torch.Tensor):
            action = output
        else:
            action = sample_fn(output)

        if clip: action = torch.clamp(action, -1, 1)

        action = self._data_postprocess(action, self.target_policy_name)

        return action

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        self.node.export2onnx(onnx_file, verbose)

class PolicyModel:
    def __init__(self, policy_model_dev : PolicyModelDev, post_process : Optional[Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray]] = None):
        self._policy_model = policy_model_dev
        self.post_process = post_process
        self.revive_version = __version__
        self.device = "cpu"

    def to(self, device: str):
        r"""
        Move model to the device specified by the parameter.

        Examples::

            >>> policy_model.to("cpu")
            >>> policy_model.to("cuda")
            >>> policy_model.to("cuda:1")

        """
        if device != self.device:
            self.device = device
            self._policy_model.to(self.device)

    def check_version(self):
        r"""Check if the revive version of the saved model and the current revive version match."""
        if not self.revive_version == __version__:
            warnings.warn(f'detect the policy is create by version {self.revive_version}, but current version is {__version__}, maybe not compactable.')

    def reset(self):
        r"""
            When using RNN for model training, this method needs to be called before model reuse 
            to reset the hidden layer information.
        """
        self._policy_model.reset()

    @property
    def target_policy_name(self) -> None:
        r''' Get the target policy name. '''
        return self._policy_model.target_policy_name

    @torch.no_grad()
    def infer(self, 
              state : Dict[str, np.ndarray], 
              deterministic : bool = True, 
              clip : bool = True,
              additional_info : Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        r"""
        Generate action according policy.

        Args:
            :state: a dict contain *ALL* the input nodes of the policy node

            :deterministic: 
                if True, the most likely actions are generated; 
                if False, actions are generated by sample.
                Default: True

            :clip:
                if True, The output will be cropped to the range set in the yaml file; 
                if False, actions are generated by sample.
                Default: True 

            :additional_info: a dict of additional info for post process

        Return: 
            action

        Examples::

            >>> state = {"obs": obs_array, "static_obs": static_obs_array}
            >>> action = policy_model.infer(state)

        """
        self.check_version()

        action = self._policy_model.infer(deepcopy(state), deterministic, clip=clip)

        if self.post_process is not None:
            state.update(additional_info)
            action = self.post_process(state, action)

        return action

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        r"""
        Exporting the model to onnx mode.

        Reference: https://pytorch.org/docs/stable/onnx.html

        Args:
            :onnx_file: the onnx model file save path.

            :verbose: if True, prints a description of the model being exported to stdout. 
                      In addition, the final ONNX graph will include the field ``doc_string``` 
                      from the exported model which mentions the source code locations for ``model``.

        """
        if self.post_process is not None:
            warnings.warn('Currently, post process will not be exported.')
        self._policy_model.export2onnx(onnx_file, verbose)