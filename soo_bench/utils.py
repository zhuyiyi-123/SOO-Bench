import numpy as np
import os
import pandas as pd


def dict2array(in_dict):
    # @TODO: finish the batch_size condition
    if len(in_dict['mode_inputs'].shape) == 1:
        arr_mode_inputs = in_dict['mode_inputs']
        arr_parallel_inputs = in_dict['parallel_inputs']
        arr_series_speed = in_dict['series_speed']
        arr_series_torque = in_dict['series_torque']
        in_arr = np.hstack([arr_mode_inputs, arr_parallel_inputs, arr_series_speed, arr_series_torque])
    else: # @TODO
        in_arr = np.array([])
        for i in range(len(in_dict['mode_inputs'])):
            arr = np.array([])
            arr_mode_inputs = in_dict['mode_inputs'][i]
            arr_parallel_inputs = in_dict['parallel_inputs'][i]
            arr_series_speed = in_dict['series_speed'][i]
            arr_series_torque = in_dict['series_torque'][i]
            in_arr.append(np.hstack([arr_mode_inputs, arr_parallel_inputs, arr_series_speed, arr_series_torque]))
    
    return in_arr

def array2dict(in_arr_list):
    # @TODO: consider the input as list or array
    if len(in_arr_list) == 1:
        in_dict = dict()
        in_dict['mode_inputs'] = in_arr_list[0][0:7]
        in_dict['parallel_inputs'] = in_arr_list[0][7:9]
        in_dict['series_speed'] = in_arr_list[0][9:62]
        in_dict['series_torque'] = in_arr_list[0][62:115]
    else:
        in_dict = dict()
        in_dict['mode_inputs'] = []
        in_dict['parallel_inputs'] = []
        in_dict['series_speed'] = []
        in_dict['series_torque'] = []
        for i in range(len(in_arr_list)):
            in_dict['mode_inputs'].append(in_arr_list[i][0:7])
            in_dict['parallel_inputs'].append(in_arr_list[i][7:9])
            in_dict['series_speed'].append(in_arr_list[i][9:62])
            in_dict['series_torque'].append(in_arr_list[i][62:115])
        in_dict['mode_inputs'] = np.array(in_dict['mode_inputs'])
        in_dict['parallel_inputs'] = np.array(in_dict['parallel_inputs'])
        in_dict['series_speed'] = np.array(in_dict['series_speed'])
        in_dict['series_torque'] = np.array(in_dict['series_torque'])
    
    return in_dict

def array2list(out_arr):
    return np.squeeze(out_arr, axis=1)

def calculate_SI(sequence):
    data = sequence
    opt_step = len(data)
    curve = data[:opt_step]

    curve = [item[0] if isinstance(item, list) else item for item in curve]
    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    # print(S, S_d, S_O)
    SI = (S-S_d) / (S_O - S_d)
    SI = np.around(SI, 3)
    return SI

class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def record_solutions_epoch(self, e, solutions, results, results_info):
        """
        Args:
            e (_type_): _description_
            solutions (_type_): _description_
            results (_type_): _description_
            results_info (_type_): _description_

        Returns:
            _type_: _description_
        """
        best = np.min(results)
        e_path = os.path.join(self.log_dir, ("epoch_" + str(e) + '_best_'+ str(best)))
        if not os.path.exists(e_path):
            os.makedirs(e_path)
        for i in range(len(results)):
            solution = dict()
            result_info = dict()
            solution['mode_inputs'] = solutions['mode_inputs'][i]
            solution['parallel_inputs'] = solutions['parallel_inputs'][i]
            solution['series_speed'] = solutions['series_speed'][i]
            solution['series_torque'] = solutions['series_torque'][i]
            result_info['fuel'] = results_info['fuel'][i]
            result_info['punish_mode'] = results_info['punish_mode'][i]
            result_info['punish_soc'] = results_info['punish_soc'][i]
            
            self.record_solution(solution, results[i], result_info, e_path)
        return 1
    
    def record_solution(self, solution, result, result_info, e_path):
        """
        Args:
            solution (_type_): _description_
            result (_type_): _description_
            result_info (_type_): _description_

        Returns:
            _type_: _description_
        """
        solution_dict = dict()
        solution_df = pd.DataFrame() 
        solution_dict['results'] = [result, result_info['fuel'], result_info['punish_mode'], result_info['punish_soc']]
        solution_dict['mode_inputs'] = solution['mode_inputs']
        solution_dict['parallel_inputs'] = solution['parallel_inputs']
        solution_dict['series_speed'] = solution['series_speed']
        solution_dict['series_torque'] = solution['series_torque']
        for k in solution_dict.keys():	
            solution_df = pd.concat([solution_df, pd.DataFrame(solution_dict[k], columns=[k])],axis=1)
        solution_df.to_csv(os.path.join(e_path, str(result[0]) +".csv"))
        
        return 1

    def record_results(self, results):
        """
        Args:
            results (_type_): _description_

        Returns:
            _type_: _description_
        """
        results.to_csv(os.path.join(self.log_dir, 'results.cvs'))
        return 1
