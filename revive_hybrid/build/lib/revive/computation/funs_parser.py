import re
import yaml
from shutil import copyfile
from loguru import logger

import revive.computation.operators as opt

OPERATORS_NAME = sorted(opt.__all__, reverse=False)

def get_nodes(yaml_file_path,):
    with open(yaml_file_path, 'r', encoding='UTF-8') as f:
        raw_config = yaml.load(f, Loader=yaml.FullLoader)

    data_config = raw_config['metadata']['columns']
    nodes_name = set([list(d.values())[0]['dim'] for d in data_config])
    nodes = { node_name : [list(d.keys())[0]  for d in data_config if list(d.values())[0]['dim'] == node_name] for node_name in nodes_name}

    for node in raw_config['metadata']['graph'].keys():
        if "next_" in node and node[5:] in nodes.keys():
            nodes[node] = nodes[node[5:]]

    return nodes


def matching_bracket(string, idx, brackets=[]):
    a = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">"
    }
    if brackets:
        a = { k:v for k in a.items() for k in brackets}
    
    close_bracket = a[string[idx]]

    if idx != string.rindex(string[idx]):     
        b = string.rindex(string[idx])         
        c = string[b:].index(close_bracket)            
        str_list = list(string)
        str_list[b], str_list[b+c] = ".", "."              
        return matching_bracket(''.join(str_list), idx)  
    else:
        d = string[idx:].index(close_bracket)
        return idx + d

def strip(s):
    return re.sub("\s|\t|\n", "", s)

def find_all(s, sub_s):
    return [m.start() for m in re.finditer(sub_s, s)]

def find_node(line):
    nodes = {}
    for node_name in sorted(NODES.keys(), reverse=True):
        res = find_all(line, node_name)
        for index in res:
            bracket_start_index = index+len(node_name)
            if bracket_start_index==len(line):
                continue
            if line[bracket_start_index] != "[" or line[bracket_start_index+1:bracket_start_index+5] == "...,":
                continue
            bracket_stop_index = matching_bracket(line, bracket_start_index) + 1
            nodes[bracket_start_index] = bracket_stop_index

    nodes = [[index, nodes[index]] for index in sorted(nodes.keys())]
    return nodes

def find_column(line):
    columns = {}
    for node_name in NODES.keys():
        for column_name in NODES[node_name]:
            res = find_all(line, column_name)
            for index in res:
                if line[index-1:index+len(column_name)+1] == f'"{column_name}"' or \
                    line[index-1:index+len(column_name)+1] == f"'{column_name}'":
                    columns[index] = [column_name,node_name]
    
    columns = [[index, columns[index]] for index in sorted(columns.keys())]
    return columns
    
def find_operator(line):
    operators = {}
    for operator_name in OPERATORS_NAME:
        res = find_all(line, operator_name)
        for index in res:  
            if line[index-1] in [" ", "=", "(", ","] and not bool(re.match('[a-zA-Z0-9_]+', line[index+len(operator_name)])):
                operators[index] = operator_name
    
    operators = [[index, operators[index]] for index in sorted(operators.keys())]
    
    return operators

def convert_operator(oral_operator_name):
    return "opt." + oral_operator_name
    
def convert_column(oral_column,flag):
    column_name, node_name = oral_column
    return f"{NODES[node_name].index(column_name)}"

def convert_node(bracket_start_index):
    return f"{NODES[node_name].index(column_name)}"

def convert_operators(line):
    """ Convert operator 
    
    Example: add -> opt.add
    """
    operators = find_operator(line)
    if operators:
        first_operator_index, first_operator_name = operators[0]
        line = line[:first_operator_index] \
               + convert_operator(first_operator_name) \
               + line[first_operator_index+len(first_operator_name):]
        return convert_operators(line)
    else:
        return line
        
def convert_columns(line):
    """ Convert column name to column index 
    
    Example: "obs_1" -> 1
    """
    columns = find_column(line)
    if columns:
        first_column_index, first_column_name = columns[0]
        if "[" == line[first_column_index-2]:
            flag = "L"
        elif "]" == line[first_column_index+len(first_column_name[0])+1]:
            flag = "R"
        else:
            flag = "C"

        line = line[:first_column_index-1] \
               + convert_column(first_column_name,flag) \
               + line[first_column_index+len(first_column_name[0])+1:]
        return convert_columns(line)
    else:
        return line
    
def convert_nodes(line):
    """ Convert node to tensor 
    
    Example: obs[1,2] -> obs[...,[1,2]]
    """
    nodes = find_node(line)
    if nodes:
        bracket_start_index, bracket_stop_index = nodes[0]
        if ":" in line[bracket_start_index:bracket_stop_index]:
            line = line[:bracket_start_index] \
                   + "[...," \
                   + line[bracket_start_index+1:bracket_stop_index-1] \
                   + "]" \
                   + line[bracket_stop_index:]
        else:
            line = line[:bracket_start_index] \
                   + "[...," \
                   + line[bracket_start_index:bracket_stop_index] \
                   + "]" \
                   + line[bracket_stop_index:]
        return convert_nodes(line)
    else:
        return line
    
def convert_line(line):
    line = strip(line)
    line = convert_operators(line)
    line = convert_columns(line)
    line = convert_nodes(line)
    return line

def checkt_convert(line):
    comvert_line_copy = line
    for fn in [convert_operators,convert_columns,convert_nodes]:
        comvert_line_copy = fn(comvert_line_copy)
    if comvert_line_copy == line:
        return False
    return True 

def convert_fn_def(origin_code : list):
    ''' find the intent of original code '''
    codes = []
    index = 0
    for i, code in enumerate(origin_code):
        if code.startswith('def '):
            bracket_start_index = code.index("(")
            codes.append(code[:bracket_start_index]+"(data: Dict[str, torch.Tensor]) -> torch.Tensor:\n")
            index = i
        if ":" in code:
            index = i
            break
            
    args = strip("".join(origin_code[:index+1]))
    bracket_start_index = args.index("(")
    bracket_stop_index = matching_bracket(args,bracket_start_index)
    args = args[bracket_start_index+1:bracket_stop_index].split(",")
    for arg in args:
        codes.append(f'    {arg}=data["{arg}"]\n')
    
    other_codes = origin_code[index+1:]
            
    return codes, other_codes

def get_fn_list(origin_code_list):
    fn_start_index = []
    fn_stop_index = []
    for i,code in enumerate(origin_code_list):
        if code.startswith('def '):
            if fn_start_index:
                fn_stop_index.append(i)
            fn_start_index.append(i)
    fn_stop_index.append(i+1)   
    
    return [origin_code_list[i:j] for i,j in zip(fn_start_index,fn_stop_index)]
    

def parser(input_file : str,
           output_file : str,
           yaml_file : str):
    
    global NODES
    NODES = get_nodes(yaml_file)
    
    with open(input_file, 'r') as f:
        origin_code_list = f.readlines()
        
    for code in origin_code_list:
        if code.startswith("import torch"):
            logger.info(f'Not parser function in {input_file}')
            copyfile(input_file, output_file)
            return False
        
    logger.info(f'Parser function in {input_file}')
                              
    output_codes = []
    output_codes.append("import torch\n")
    output_codes.append("from typing import Dict\n")
    output_codes.append("\n")
    output_codes.append("import revive.computation.operators as opt\n")
    output_codes.append("\n")
    
    fn_list = get_fn_list(origin_code_list)
    
    for fn in fn_list:
        start_codes, other_codes = convert_fn_def(fn)
        output_codes += start_codes

        for code in other_codes:
            if not checkt_convert(code):
                output_codes.append(code)  
                continue
            if " return " in code:
                if "(" not in code and "[" not in code:
                    output_codes.append(code)
                else:
                    sub_code = convert_line("return="+code[code.index("return")+len("return"):])
                    sub_code = sub_code.replace("return=", "return ")
                    output_codes.append(" "*(len(code) - len(code.lstrip()))+sub_code+"\n")
            else:
                output_codes.append(" "*(len(code) - len(code.lstrip()))+convert_line(code)+"\n")
            
    with open(output_file, 'w') as f:
        f.writelines(output_codes)

    return True
