import loguru
import torch
import warnings
import numpy as np
from loguru import logger
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Union

from revive.data.processor import DataProcessor
from revive.computation.dists import ReviveDistribution
from revive.computation.modules import *

class DesicionNode:
    ''' An abstract node for making decisions '''

    node_type : str = None # mark the type of the node

    def __init__(self, name : str, input_names : List[str], input_descriptions : List[Dict[str, Dict[str, Any]]]):
        assert len(input_descriptions) == len(input_names)
        self.name = name
        self.input_names = input_names
        self.input_descriptions = input_descriptions
        self.processor = None

    def __call__(self, data : Dict[str, torch.Tensor], *args, **kwargs) -> Union[torch.Tensor, ReviveDistribution]:
        ''' 
            Run a forward computation of this node. 
            NOTE: The input data was transferred by self.processor. You can use `self.processor.deprocess_torch(data)` to get the original data.
        '''
        raise NotImplementedError

    def register_processor(self, processor : DataProcessor):
        ''' register the global data processor to this node '''
        self.processor = processor

    def remove_processor(self):
        ''' remove the registered data processor '''
        self.processor = None

    def get_inputs(self, data : Dict[str, torch.Tensor]) -> OrderedDict:
        ''' get only input variables '''
        inputs = OrderedDict()
        for input_name in self.input_names:
            inputs[input_name] = data[input_name]
        return inputs
            
    def to(self, device : str) -> 'DesicionNode':
        ''' change the device of this node '''
        return self

    def requires_grad_(self, mode : bool = False) -> 'DesicionNode':
        ''' change the requirement of gradient for this node '''
        return self

    def train(self) -> 'DesicionNode':
        ''' set the state of this node to training '''
        return self

    def eval(self) -> 'DesicionNode':
        ''' set the state of this node to evaluation '''
        return self

    def reset(self) -> None:
        ''' reset the state of this node, useful when node is an RNN '''
        pass

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        ''' export the node to onnx file, with input from original data space '''
        assert self.processor is not None, 'please register processor before export!'

        node = deepcopy(self)
        node = node.to('cpu')
        node.requires_grad_(False)
        node.eval()

        class ExportHelper(torch.nn.Module):
            def forward(self, state : Dict[str, torch.Tensor]) -> torch.Tensor:
                state = node.processor.process_torch(state)
                
                output = node(state)
                if isinstance(output, torch.Tensor):
                    action = output
                else:
                    action = output.mode

                action = torch.clamp(action, -1, 1)

                action = node.processor.deprocess_single_torch(action, node.name)

                return action

        demo_inputs = {}
        for name, description in zip(self.input_names, self.input_descriptions):
            demo_inputs[name] = torch.randn(len(description), dtype=torch.float32)
        
        torch.onnx.export(ExportHelper(), demo_inputs, onnx_file, verbose=verbose, input_names=self.input_names, output_names=[self.name], opset_version=11)

    def __str__(self) -> str:
        info = []
        info.append(f'node class : {type(self)}')
        info.append(f'node name : {self.name}')
        info.append(f'node inputs : {self.input_names}')
        info.append(f'processor : {self.processor}')
        info.append(f'node type : {self.node_type}')
        return '\n'.join(info)

class NetworkDecisionNode(DesicionNode):
    node_type = 'network'

    def __init__(self, name: str, input_names: List[str], input_descriptions: List[Dict[str, Dict[str, Any]]]):
        super().__init__(name, input_names, input_descriptions)
        self.network = None

    def set_network(self, network : torch.nn.Module):
        ''' set the network from a different source '''
        self.network = network

    def get_network(self) -> torch.nn.Module:
        ''' return all the network in this node '''
        return self.network

    def initialize_network(self, 
                           input_dim : int,
                           output_dim : int,
                           hidden_features : int,
                           hidden_layers : int,
                           backbone_type : str,
                           dist_config : list,
                           is_transition : bool = False,
                           hidden_activation : str = 'leakyrelu',
                           norm : str = None, 
                           transition_mode : Optional[str] = None,
                           obs_dim : Optional[int] = None,
                           *args, **kwargs):

        ''' initialize the network of this node '''

        assert self.network is None, 'Cannot initialize one node twice!'

        if is_transition:
            if backbone_type in ['mlp', 'res', 'transformer']:
                network = FeedForwardTransition(input_dim, output_dim,
                                                hidden_features, hidden_layers,
                                                norm=norm, 
                                                hidden_activation=hidden_activation,
                                                dist_config=dist_config, 
                                                backbone_type=backbone_type,
                                                mode=transition_mode, 
                                                obs_dim=obs_dim)
            elif backbone_type in ['gru', 'lstm']:
                network = RecurrentTransition(input_dim, output_dim,
                                              hidden_features, hidden_layers,
                                              dist_config, backbone_type=backbone_type,
                                              mode=transition_mode, obs_dim=obs_dim)
            else:
                raise ValueError(f'Initializing node `{self.name}`, backbone type {backbone_type} is not supported!')
        else:
            if backbone_type in ['mlp', 'res', 'transformer']:
                network = FeedForwardPolicy(input_dim, output_dim,
                                            hidden_features, hidden_layers,
                                            dist_config=dist_config,
                                            norm=norm, 
                                            hidden_activation=hidden_activation,
                                            backbone_type=backbone_type)         
            elif backbone_type in ['gru', 'lstm']:
                network = RecurrentPolicy(input_dim, output_dim,
                                          hidden_features, hidden_layers,
                                          dist_config, backbone_type)
            else:
                raise ValueError(f'Initializing node `{self.name}`, backbone type {backbone_type} is not supported!')

        self.network = network

    def __call__(self, data : Dict[str, torch.Tensor], *args, **kwargs) -> ReviveDistribution:
        ''' 
            Run a forward computation of this node. 
            NOTE: The input data was transferred by self.processor. You can use `self.processor.deprocess_torch(data)` to get the original data.
        '''
        data = self.get_inputs(data)
        inputs = torch.cat([data[k] for k in self.input_names], dim=-1)
        output_dist = self.network(inputs.to('cpu'), *args, **kwargs) 
        return output_dist
        
    def to(self, device : str) -> 'DesicionNode':
        ''' change the device of this node '''
        self.network = self.network.to(device)
        # self.network = self.network
        return self

    def requires_grad_(self, mode : bool = False) -> 'DesicionNode':
        ''' change the requirement of gradient for this node '''
        self.network.requires_grad_(mode)
        return self

    def train(self) -> 'DesicionNode':
        ''' set the state of this node to training '''
        self.network.train()
        return self

    def eval(self) -> 'DesicionNode':
        ''' set the state of this node to evaluation '''
        self.network.eval()
        return self

    def reset(self) -> None:
        ''' reset the state of this node, useful when node is an RNN '''
        try:
            self.network.reset()
        except:
            pass
        return self

    def __str__(self) -> str:
        info = [super(NetworkDecisionNode, self).__str__()]
        info.append(f'network : {self.network}')
        return '\n'.join(info)

class FunctionDecisionNode(DesicionNode):
    node_type = 'function'

    def __init__(self, name: str, input_names: List[str], input_descriptions: List[Dict[str, Dict[str, Any]]]):
        super().__init__(name, input_names, input_descriptions)
        self.node_function = None
        self.node_function_type = None

    def register_node_function(self, 
                               node_function : Union[Callable[[Dict[str, np.ndarray]], np.ndarray], 
                                                     Callable[[Dict[str, torch.Tensor]], torch.Tensor]],
                               node_function_type : str):
        self.node_function = node_function
        self.node_function_type = node_function_type

    def remove_node_function(self):
        self.node_function = None
        self.node_function_type = None

    def __call__(self, data : Dict[str, torch.Tensor], *args, **kwargs) -> Union[torch.Tensor, ReviveDistribution]:
        ''' NOTE: if there is any provided function defined in numpy, this process cannot maintain gradients '''
        data = self.get_inputs(data)

        torch_data = list(data.values())[0]
        data_type = 'torch'
        deprocessed_data = self.processor.deprocess_torch(data)

        if self.node_function_type == 'numpy':
            for k in deprocessed_data.keys(): deprocessed_data[k] = deprocessed_data[k].detach().cpu().numpy() # torch -> numpy
            data_type = 'numpy'
        
        output = self.node_function(deprocessed_data)
        if data_type == 'numpy':
            if np.isinf(np.mean(output)).item():
                logger.error(f"Find inf in {self.name} node function output {output}")
                raise ValueError(f"Find inf in {self.name} node function output {output}")
                
            if np.isnan(np.mean(output)).item():
                logger.error(f"Find nan in {self.name} node function output {output}")
                raise ValueError(f"Find nan in {self.name} node function output {output}")

            output = self.processor.process_single(output, self.name)
            output = torch.as_tensor(output).to(torch_data) # numpy -> torch
        else:
            if torch.isinf(torch.mean(output)).item():
                logger.error(f"Find inf in {self.name} node function output {output}")
                raise ValueError(f"Find inf in {self.name} node function output {output}")
                
            if torch.isnan(torch.mean(output)).item():
                logger.error(f"Find nan in {self.name} node function output {output}")
                raise ValueError(f"Find nan in {self.name} node function output {output}")

            output = self.processor.process_single_torch(output, self.name)

        return output

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        ''' export the node to onnx file, with input from original data space '''
        if self.node_function_type == 'numpy':
            warnings.warn(f'Detect function in node `{self.name}` with type numpy, export may be incorrect.')
        super(FunctionDecisionNode, self).export2onnx(onnx_file, verbose)

    def __str__(self) -> str:
        info = [super(FunctionDecisionNode, self).__str__()]
        info.append(f'node function : {self.node_function}')
        info.append(f'node function type: {self.node_function_type}')
        return '\n'.join(info)

class DesicionGraph:
    r''' A collection of DecisionNodes '''

    def __init__(self, 
                 graph_dict : Dict[str, List[str]], 
                 descriptions : Dict[str, List[Dict[str, Dict[str, Any]]]],
                 fit,
                 metric_nodes,) -> None:
        self.descriptions = descriptions
        self.graph_dict = self.sort_graph(graph_dict)
        self.fit = fit
        self.leaf = self.get_leaf(self.graph_dict)
        self.transition_map = self._get_transition_map(self.graph_dict)
        self.external_factors = list(filter(lambda x: not x in self.transition_map.keys(), self.leaf))
        self.tunable = []
        self.nodes = OrderedDict()
        for node_name in self.graph_dict.keys():
            self.nodes[node_name] = None

        if metric_nodes is None:
            self.metric_nodes = list(self.nodes.keys())
        else:
            self.metric_nodes = []
            for node in self.nodes.keys():
                if node in metric_nodes:
                    if node in self.leaf:
                        logger.info(f"Node '{node}' is a leaf node, it should't be a metric node.")
                        continue
                    assert node in self.nodes.keys(), f"Metric node '{node}' is not in Graph, Please check yaml."
                    self.metric_nodes.append(node)
        assert len(self.metric_nodes) >= 1, f"At least one non-leaf node is required for metric."

        self.is_target_network = False

    def register_node(self, node_name : str, node_class):
        r''' Register a node with given node class '''
        assert self.nodes[node_name] is None, f'Cannot register node `{node_name}`, the node is already registered as `{type(self.nodes[node_name])}`'
        input_names = self.graph_dict[node_name]
        self.nodes[node_name] = node_class(node_name, input_names, [self.descriptions[input_name] for input_name in input_names])

        # TODO: UPDATE
        if node_class.node_type == 'function':
            if node_name in self.metric_nodes:
                self.metric_nodes.remove(node_name)
                assert len(self.metric_nodes) >= 1, f"At least one non-leaf node is required for metric."

    @property
    def learnable_node_names(self) -> List[str]:
        r'''A list of names for learnable nodes the graph'''
        node_names = []
        for node_name, node in self.nodes.items():
            if not node.node_type == 'function':
                node_names.append(node_name)
        return node_names

    def register_target_nodes(self):
        self.target_nodes = deepcopy(self.nodes)

    def del_target_nodes(self):
        assert not self.is_target_network
        del self.target_nodes
        
    def use_target_network(self,):
        if self.is_target_network is False:
            self.target_nodes, self.nodes = self.nodes, self.target_nodes
            self.is_target_network = True

    def not_use_target_network(self,):
        if self.is_target_network is True:
            self.target_nodes, self.nodes = self.nodes, self.target_nodes
            self.is_target_network = False

    def update_target_network(self, polyak=0.99):
        with torch.no_grad():
            for node_name, node in self.nodes.items():
                if not node.node_type == 'function':
                    target_node = self.target_nodes[node_name]
                    for p, p_targ in zip(node.network.parameters(), target_node.network.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)


    def mark_tunable(self, node_name : str) -> None:
        r'''Mark a leaf variable as tunable'''
        assert node_name in self.external_factors, 'Only external factors can be tunable!'
        if node_name in self.tunable:
            warnings.warn(f'{node_name} is already marked as a tunable node, skip.')
        else:
            self.tunable.append(node_name)

    def register_processor(self, processor : DataProcessor):
        r'''Register data processor to the graph and nodes'''
        self.processor = processor
        for node in self.nodes.values():
            node.register_processor(self.processor)

    def get_node(self, node_name : str, use_target: bool = False) -> DesicionNode:
        '''get the node by name'''
        if self.nodes[node_name].node_type == 'network':
            if use_target:
                assert hasattr(self, "target_nodes"), "Not have target nodes. You should register target nodes firstly."
                return self.target_nodes[node_name]

        return self.nodes[node_name]

    def compute_node(self, node_name : str, inputs : Dict[str, torch.Tensor], use_target: bool = False, *args, **kwargs):
        '''compute the node by name'''
        return self.get_node(node_name, use_target)(inputs, *args, **kwargs)

    def get_relation_node_names(self) -> List[str]:
        ''' 
            get all the nodes that related to the learning (network) nodes.
            NOTE: this is the default list if you have matcher and value functions. 
        '''
        node_names = []
        for node in self.nodes.values():
            node_name = node.name
            input_names = node.input_names

            if not (node.node_type == 'function'): # skip function nodes
                for name in input_names + [node_name]:
                    if not (name in node_names):
                        node_names.append(name)
        return node_names

    def summary_nodes(self) -> Dict[str, int]:
        network_nodes = 0
        function_nodes = 0
        unregistered_nodes = 0
        unknown_nodes = 0
        
        for node in self.nodes.values():
            if node is None:
                unregistered_nodes += 1
            else:
                if node.node_type == 'network':
                    network_nodes += 1
                elif node.node_type == 'function':
                    function_nodes += 1
                else:
                    unknown_nodes += 1
                

        return {
            'network_nodes' : network_nodes,
            'function_nodes' : function_nodes,
            'unregistered_nodes' : unregistered_nodes,
            'unknown_nodes' : unknown_nodes,
        }

    def collect_models(self) -> List[torch.nn.Module]:
        '''return all the network that registered in this graph'''
        return [node.get_network() for node in self.nodes.values() if node.node_type == 'network']

    def is_equal_structure(self, another_graph : 'DesicionGraph') -> bool:
        ''' check if new graph shares the same structure '''
        return self.graph_dict == another_graph.graph_dict

    def copy_graph_model(self, source_graph : 'DesicionGraph') -> bool:
        ''' copy all the node networks from source graph '''
        for node_name in self.nodes.keys():
            target_node = self.get_node(node_name)
            source_node = source_graph.get_node(node_name)
            if source_node.node_type == 'network' and target_node.node_type == 'network':
                target_node.set_network(source_node.get_network())

    def get_leaf(self, graph : Dict[str, List[str]] = None) -> List[str]:
        ''' return the leaf of the graph in *alphabet order* '''
        if graph is None:
            graph = self
        outputs = [name for name in graph.keys()]
        inputs = []
        for names in graph.values(): inputs += names
        inputs = set(inputs)
        leaf = [name for name in inputs if name not in outputs]
        leaf = sorted(leaf) # make sure the order of leaf is fixed given a graph
        return leaf

    def _get_transition_map(self, graph : Dict[str, List[str]]) -> Dict[str, str]:
        outputs = [name for name in graph.keys()]
        inputs = []
        for names in graph.values(): inputs += names
        inputs = set(inputs)
        transition = {name[5:] : name for name in outputs if name.startswith('next_') and name[5:] in inputs}
        return transition

    def sort_graph(self, graph_dict : dict) -> OrderedDict:
        '''Sort arbitrary computation graph to the topological order'''
        ordered_graph = OrderedDict()
        computed = self.get_leaf(graph_dict)
        
        # sort output
        while len(graph_dict) > 0:
            find_new_node = False
            for output_name in sorted(graph_dict.keys()):
                input_names = graph_dict[output_name]
                if all([name in computed for name in input_names]):
                    ordered_graph[output_name] = graph_dict.pop(output_name)
                    computed.append(output_name)
                    find_new_node = True
                    break
            
            if not find_new_node:
                raise ValueError('Cannot find any computable node, check if there are loops or isolations on the graph!\n' + \
                    f'current computed nodes: {computed}, node waiting to be computed: {graph_dict}')

        # sort input
        for output_name, input_names in ordered_graph.items():
            sorted_input_names = []
            if output_name.startswith('next_') and output_name[5:] in input_names:
                sorted_input_names.append(output_name[5:])
                input_names.pop(input_names.index(output_name[5:]))
            for name in computed:
                if name in input_names:
                    sorted_input_names.append(name)
            ordered_graph[output_name] = sorted_input_names

        assert self._is_sort_graph(ordered_graph), f"{ordered_graph}, graph is not correctly sorted!"

        return ordered_graph

    def _is_sort_graph(self, graph : OrderedDict) -> bool:
        ''' check if a graph is sorted '''
        computed = self.get_leaf(graph)

        for output_name, input_names in graph.items():
            for name in input_names:
                if not name in computed:
                    return False
            computed.append(output_name)

        return True

    def to(self, device : str) -> 'DesicionGraph':
        for node in self.nodes.values(): node.to(device)
        # for node in self.nodes.values(): node
        return self

    def requires_grad_(self, mode : bool = False) -> 'DesicionGraph':
        for node in self.nodes.values(): node.requires_grad_(mode)
        return self

    def eval(self) -> 'DesicionGraph':
        for node in self.nodes.values(): node.eval()
        return self

    def reset(self) -> 'DesicionGraph':
        ''' reset graph, useful for stateful graph '''
        for node in self.nodes.values(): node.reset()
        return self

    def __getitem__(self, name : str) -> List[str]:
        return self.graph_dict[name]

    def keys(self) -> Iterable[str]:
        return self.graph_dict.keys()

    def values(self) -> Iterable[List[str]]:
        return self.graph_dict.values()

    def items(self) -> Iterable[Dict[str, List[str]]]:
        return self.graph_dict.items()

    def __len__(self) -> int:
        return len(self.graph_dict)

    def __str__(self) -> str:
        node_info = [node.__str__() for node in self.nodes.values()]
        return '\n\n'.join(node_info)

    def __call__(self, state : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ''' compute the whole graph, from leaf node to all output nodes '''
        assert all([node_name in state.keys() for node_name in self.leaf])

        for node_name, node in self.nodes.items():
            output = node(state)
            if isinstance(output, torch.Tensor):
                action = output
            else:
                action = output.mode

            action = torch.clamp(action, -1, 1)

            state[node_name] = action

        actions = {node_name : state[node_name] for node_name in self.nodes.keys()}

        return actions

    def state_transition(self, state : Dict[str, torch.Tensor], copy : bool = False) -> Dict[str, torch.Tensor]:
        new_state = {}
        for new_name, old_name in self.transition_map.items():
            new_state[new_name] = state[old_name]
        if copy: new_state = deepcopy(new_state)
        return new_state

    def export2onnx(self, onnx_file : str, verbose : bool = True):
        ''' export the graph to onnx file, with input from original data space '''
        assert self.processor is not None, 'please register processor before export!'
        for node_name, node in self.nodes.items():
            if node.node_type == 'function':
                if node.node_function_type == 'numpy':
                    warnings.warn(f'Detect function in node {node_name} with type numpy, may be incorrect.')

        graph = deepcopy(self)
        graph = graph.to('cpu')
        graph.requires_grad_(False)
        graph.eval()

        class ExportHelper(torch.nn.Module):
            def forward(self, state : Dict[str, torch.Tensor]) -> torch.Tensor:
                state = graph.processor.process_torch(state)

                actions = graph(state)

                actions = graph.processor.deprocess_torch(actions)

                return tuple([actions[node_name] for node_name in graph.nodes.keys()])

        demo_inputs = {}
        for name in self.leaf:
            description = self.descriptions[name]
            demo_inputs[name] = torch.randn(len(description), dtype=torch.float32)
        
        torch.onnx.export(ExportHelper(), demo_inputs, onnx_file, verbose=verbose, input_names=self.leaf, output_names=list(self.nodes.keys()), opset_version=11)
