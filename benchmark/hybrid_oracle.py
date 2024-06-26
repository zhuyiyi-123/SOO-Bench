import sys
import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import cma
import pandas as pd

from revive.utils.common_utils import generate_rollout, load_h5
from revive.data.batch import Batch
from utils import *



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SearchParams:
    def __init__(self):
        self.env_dir = r"./data/jl_envs.pkl"
        self.data_dir = r"./data/jl_data.h5"
        self.env = self._load_env()
        self.data = self._load_data()
        self.tra_num = None
        self.tra_len = 1801  # 默认轨迹长度1801
        self.demo_inputs = self.get_demo_inputs()

        self.mode_inputs_dims = self.demo_inputs["mode_inputs"].shape[0]
        self.parallel_inputs_dims = self.demo_inputs["parallel_inputs"].shape[0]
        self.series_speed_dims = self.demo_inputs["series_speed"].shape[0]
        self.series_torque_dims = self.demo_inputs["series_torque"].shape[0]

    def _load_env(self):
        # sys.path.append("../data")
        with open(self.env_dir, "rb") as f:
            e = pkl.load(f)
        return e

    def _load_data(self):
        d = load_h5(self.data_dir)
        return d

    def get_demo_inputs(self):
        demo = {}
        mode_inputs = self.data["mode_index"][0]
        parallel_inputs = self.data["engine_index"][0, :2]
        series_inputs = self.data["engine_index"][0, 2:]
        series_speed = series_inputs[:int(len(series_inputs) / 2)]
        series_torque = series_inputs[-int(len(series_inputs) / 2):]
        demo["mode_inputs"] = mode_inputs
        demo["parallel_inputs"] = parallel_inputs
        demo["series_speed"] = series_speed
        demo["series_torque"] = series_torque
        return demo

    def search(self, mode_inputs: np.ndarray, parallel_inputs: np.ndarray, series_speed: np.ndarray,
               series_torque: np.ndarray):
        """
        获取参数结果。支持批量运算。
        :param mode_inputs: 模式边界值输入: n * mode_inputs_dims
        :param parallel_inputs: 并联边界值输入: n * parallel_inputs_dims
        :param series_speed: 串联转速输入: n * series_speed_dims
        :param series_torque: 串联扭矩输入: n * series_torque_dims
        :return: results: 输入参数的返回结果: n * 1
        """
        search_data = self.prepare_search_inputs(mode_inputs, parallel_inputs, series_speed, series_torque)
        inputs = self.process_data(search_data)
        with torch.no_grad():
            out = generate_rollout(inputs, self.env.graph, self.tra_len)
        out = self.deprocess_data(out)
        results, result_info = self.get_results(out)
        tra_info = out
        
        return results, tra_info, result_info

    def get_results(self, tra, fuel_w=1., mode_w=10., soc_w=10.):
        """
        compute the return of trajectory.
        return contains sevral parts:
        1. fuel reward: 油耗, 主要目标
        2. soc legalization punishment: 目标空间的约束, 电量变化在合理范围内
        3. starting engine punishment: 这个好像没有体现? 反倒是一个解空间的约束
        """
        fuel_ratio = tra["fuel"].sum(0)
        fuel_ratio_2_fuel = lambda x: x / 626.765002440701
        fuel = fuel_ratio_2_fuel(fuel_ratio)

        mode_index = tra["mode_index"][0].squeeze()
        punish_mode = (mode_index[..., [0]] > mode_index[..., [3]]) * 1. + (
                mode_index[..., [4]] < mode_index[..., [6]]) * 1.

        # 假设delta_soc在[-1, 2]之间为合理范围, 若delta soc小于-1或delta soc大于2则惩罚为delta_soc绝对值
        start_soc = tra["soc"][0]
        last_soc = tra["next_soc"][-1]
        delta_soc = (last_soc - start_soc)
        legal_soc = (delta_soc > -1.0) * (delta_soc < 2.0) * 1.
        punish_soc = ((delta_soc < -1.0) * 1. + (delta_soc > 2.0) * 1.) * np.abs(delta_soc)
        total_return =  legal_soc * fuel * fuel_w + punish_mode * mode_w + punish_soc * soc_w
        # total_return = - legal_soc * fuel * fuel_w - punish_soc * soc_w

        # result_info = {
        #     "fuel": fuel,
        #     "punish_mode": punish_mode,
        #     "punish_soc": punish_soc,
        # }
        result_info = None
        return total_return, result_info
    
    def csearch(self, mode_inputs: np.ndarray, parallel_inputs: np.ndarray, series_speed: np.ndarray,
               series_torque: np.ndarray):
        search_data = self.prepare_search_inputs(mode_inputs, parallel_inputs, series_speed, series_torque)
        inputs = self.process_data(search_data)
        with torch.no_grad():
            out = generate_rollout(inputs, self.env.graph, self.tra_len)
        out = self.deprocess_data(out)
        results, constraints, result_info = self.cget_results(out)
        tra_info = out
        return results, constraints, tra_info, result_info
    
    def cget_results(self, tra, fuel_w=1., mode_w=10., soc_w=10.):
        fuel_ratio = tra["fuel"].sum(0)
        fuel_ratio_2_fuel = lambda x: x / 626.765002440701
        fuel = fuel_ratio_2_fuel(fuel_ratio)

        mode_index = tra["mode_index"][0].squeeze()
        punish_mode = (mode_index[..., [0]] > mode_index[..., [3]]) * 1. + (
                mode_index[..., [4]] < mode_index[..., [6]]) * 1.

        # 假设delta_soc在[-1, 2]之间为合理范围, 若delta soc小于-1或delta soc大于2则惩罚为delta_soc绝对值
        constraints = []
        start_soc = tra["soc"][0]
        last_soc = tra["next_soc"][-1]
        delta_soc = (last_soc - start_soc)
        # legal_soc = (delta_soc > -1.0) * (delta_soc < 2.0) * 1.
        punish_soc = ((delta_soc < -1.0) * 1. + (delta_soc > 2.0) * 1.) * np.abs(delta_soc)
        # total_return = - legal_soc * fuel * fuel_w - punish_mode * mode_w - punish_soc * soc_w
        # constraints.append(punish_mode)
        # constraints.append(punish_soc)
        for i in range(len(punish_mode)):
            constraints.append([punish_mode[i,0], punish_soc[i,0]])
        # print('constraints = ', constraints)
        # result_info = {
        #     "fuel": fuel,
        #     "punish_mode": punish_mode,
        #     "punish_soc": punish_soc,
        # }
        result_info = None
        return fuel, constraints, result_info

    def deprocess_data(self, data):
        """
        反归一化
        """
        out = self.env.graph.processor.deprocess_torch(data)
        depro_data = {}
        for k, v in out.items():
            depro_data[k] = v.detach().cpu().numpy()
        return depro_data

    def process_data(self, search_data):
        """
        归一化等数据处理
        """
        data = deepcopy(self.data)
        data.update(search_data)
        for k, v in data.items():
            v = torch.tensor(v, dtype=torch.float32)
            if v.ndim < 3:
                v = v.unsqueeze(1).repeat((1, self.tra_num, 1))
            data[k] = self.env.graph.processor.process_single_torch(v, k)
        inputs = Batch(data)
        return inputs

    def prepare_search_inputs(self, mode_inputs: np.ndarray, parallel_inputs: np.ndarray, series_speed: np.ndarray,
                              series_torque: np.ndarray):
        """
        预处理参数输入
        """
        if mode_inputs.ndim > 1:
            assert mode_inputs.shape[0] == parallel_inputs.shape[0] == series_speed.shape[0] == series_torque.shape[0]
        else:
            mode_inputs = np.expand_dims(mode_inputs, 0)
            parallel_inputs = np.expand_dims(parallel_inputs, 0)
            series_speed = np.expand_dims(series_speed, 0)
            series_torque = np.expand_dims(series_torque, 0)

        self.tra_num = mode_inputs.shape[0]
        mode_index = np.expand_dims(mode_inputs, 0)
        engine_index = np.concatenate([parallel_inputs, series_speed, series_torque], axis=-1)
        engine_index = np.expand_dims(engine_index, 0)

        mode_index = mode_index.repeat(self.tra_len, axis=0)
        engine_index = engine_index.repeat(self.tra_len, axis=0)
        data = {"mode_index": mode_index, "engine_index": engine_index}
        return data


def array2dict(x): # 24/5/10
    '''
        把二维数组型方案数据转化为SearchParam.search()可接受的参数形式
        parameter:
            x: 二维数组，其中元素是单个方案数据
        return:
            一个字典,其键包括'mode_inputs', 'parallel_inputs', 'series_speed','series_torque'
    '''
    xx = np.array(x, dtype=np.float32)
    
    mode_inputs = xx[:, 0:7]
    parallel_inputs = xx[:, 7:9]
    series_speed = xx[:, 9:62]
    series_torque = xx[:, 62:115]
    return {
            'mode_inputs': mode_inputs, 
            'parallel_inputs':parallel_inputs,
            'series_speed':series_speed,
            'series_torque':series_torque
            }

def dict2array(x): # 24/5/10
    '''
        array2dict逆操作,把字典转化为原二维数组的形式
        x: 一个字典,其键包括'mode_inputs', 'parallel_inputs', 'series_speed','series_torque'
    '''
    assert(len(x['mode_inputs']) == len(x['parallel_inputs']) == len(x['series_speed']) == len(x['series_torque']) )

    ret = np.concatenate((x['mode_inputs'], x['parallel_inputs'],x['series_speed'], x['series_torque']), axis=1)
    return ret



class ISearchParams: # 24/5/10
    '''SearchParams 与OfflineDataSet类间接口'''
    def __init__(self, batch = 128):
        self.search_params = SearchParams()
        self.batch = batch

    def search(self, x):
        if len(x)>self.batch:
            res = []
            for i in range(0,len(x), self.batch):
                ret = self.search(x[i:i+self.batch])
                res += ret[0]
            return res,0,0

        res = array2dict(x)

        results, tra_info, result_info = self.search_params.search(**res)
        
        results = np.reshape(results, (np.size(results),)).tolist()
        return results, tra_info, result_info
    
    def csearch(self, x):
        if len(x)>self.batch:
            res = []
            cons = []
            for i in range(0,len(x), self.batch):
                ret = self.csearch(x[i:i+self.batch])
                res += ret[0]
                cons += ret[1]
            return res, cons, 0,0

        res = array2dict(x)
        results, constraints, tra_info, result_info = self.search_params.csearch(**res)
        results = np.reshape(results, (np.size(results),)).tolist()
        return results, constraints, tra_info, result_info



if __name__ == "__main__":
    pass

    # searcher = SearchParams() 
    # logger = Logger("results/best_off") # TODO?
    # results_df = pd.DataFrame()
    # off_data = pd.read_csv('data/data_clear.csv')
    # best_off = off_data.loc[5].values
    
    
    # # data_list = []
    # # for i in range(11):
    # #     data_list.append(off_data.loc[i].values)
    # # solutions_dict = array2dict(data_list)
    # # results, tra_info, results_info = searcher.search(**solutions_dict)
    # # print(results)
    
    # es = cma.CMAEvolutionStrategy(best_off, 0.3)
    # step = 0
    # while not es.stop() and step < 100:
    #     solutions = es.ask()
    #     solutions_dict = array2dict(solutions)
    #     results, tra_info, results_info = searcher.search(**solutions_dict)
    #     outs = array2list(-results)
    #     es.tell(solutions, outs)
    #     results_df[step] = outs
    #     logger.record_solutions_epoch(step, solutions_dict, -results, results_info)
    #     step += 1
    # logger.record_results(results_df)
    # print(es.result.xbest)
    # es.result_pretty()    # cma.plot()