import numpy as np
import pickle
import os, sys
import csv
import pandas as pd
import tqdm
from soo_bench.Taskdata import*

task = OfflineTask('gtopx_data', 5, seed=1)


def get_data(save_data, T, mode, adjust=True):
    '''给定一个算法的中间结果，获取优化曲线上的点
    Args:
        save_data: 算法中间结果
        T: 优化曲线上的点数
        mode: 优化曲线上的点如何取值，'max'表示取最大值，'min'表示取最小值，'median'表示取中位数
        adjust: 是否调整优化曲线的x轴，以避免指标出现除零问题
    '''
    algorithm_name = save_data['algorithm_name']
    is_constraint = save_data['constraint']
    args = save_data['args']
    offline_x = save_data['offline_x']
    offline_y = save_data['offline_y']
    offline_cons = save_data['offline_cons']
    save_xs = save_data['xs']
    save_ys = save_data['ys']
    save_cons = save_data['cons']
    other = save_data['other']
    
    data = [np.max(offline_y)]
    for i in range(len(save_ys)):
        if mode == 'max':
            data.append(np.max(save_ys[i]))
        elif mode == 'min':
            data.append(np.min(save_ys[i]))
        elif mode == 'median':
            data.append(np.median(save_ys[i]))
        # data.append(np.max(save_ys[i]))
    data = np.array(data)
    
    # zero_line = offline_worse
    offline_best = np.max(offline_y)
    algorithm_best = max(np.max(save_ys), offline_best)
    delta = max(algorithm_best - offline_best, 1e-2)
    
    zero_line = offline_best - delta
    
    if adjust:
        data -= zero_line
    
    data = data[:T]
    return data

def get_maxv(save_data, T, T_max, mode):
    '''给定一个算法的中间结果，获取优化曲线最末尾的结果，即最终结果
    Args:
        save_data: 算法中间结果
        T: 优化曲线上的点数
        mode: 优化曲线上的点如何取值，'max'表示取最大值，'min'表示取最小值，'median'表示取中位数
    '''
    data = get_data(save_data, T, mode, adjust=False)
    if len(data)<T:
        return None
    return data[-1]

def get_SI(save_data, T, T_max, mode):
    '''给定一个算法的中间结果，获取优化曲线的SI指标
    Args:
        save_data: 算法中间结果
        T: 优化曲线上的点数
        mode: 优化曲线上的点如何取值，'max'表示取最大值，'min'表示取最小值，'median'表示取中位数
    '''
    data = get_data(save_data, T, mode)
    if len(data)<T:
        return None
    opt_step = T
    curve = data[:T]
    curve = [item[0] if isinstance(item, list) else item for item in curve]

    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    # SI = (S - S_d) / (S_O - S_d)
    # SI = np.around(SI, 3)
    SI = S / S_O
    return SI

def get_OI(save_data, T, T_max, mode):
    '''给定一个算法的中间结果，获取优化曲线的OI指标
    Args:
        save_data: 算法中间结果
        T: 优化曲线上的点数
        mode: 优化曲线上的点如何取值，'max'表示取最大值，'min'表示取最小值，'median'表示取中位数
    '''
    data = get_data(save_data, T, mode)
    if len(data)<T:
        return None
    opt_step = T
    curve = data[:T]
    curve = [item[0] if isinstance(item, list) else item for item in curve]

    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    # SI = (S - S_d) / (S_O - S_d)
    # SI = np.around(SI, 3)
    OI = S / S_d
    return OI

def get_SO(save_data, T, T_max, mode):
    '''给定一个算法的中间结果，获取优化曲线的SO指标
    Args:
        save_data: 算法中间结果
        T: 优化曲线上的点数
        mode: 优化曲线上的点如何取值，'max'表示取最大值，'min'表示取最小值，'median'表示取中位数
    '''
    data = get_data(save_data, T, mode)
    if len(data)<T:
        return None
    opt_step = T
    curve = data[:T]
    curve = [item[0] if isinstance(item, list) else item for item in curve]

    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    # SI = (S - S_d) / (S_O - S_d)
    # SI = np.around(SI, 3)
    OI = S / S_d
    SI = S / S_O
    SO = SI*OI/(0.5*SI+0.5*OI)
    return SO


def get_SOW(save_data, T, T_max, mode):
    '''给定一个算法的中间结果，获取优化曲线的SO指标
    Args:
        save_data: 算法中间结果
        T: 优化曲线上的点数
        mode: 优化曲线上的点如何取值，'max'表示取最大值，'min'表示取最小值，'median'表示取中位数
    '''
    data = get_data(save_data, T, mode)
    if len(data)<T:
        return None
    w = max(0, 1 - T /(T_max + 1))
    
    opt_step = T
    curve = data[:T]
    curve = [item[0] if isinstance(item, list) else item for item in curve]
    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    # SI = (S - S_d) / (S_O - S_d)
    # SI = np.around(SI, 3)
    OI = S / S_d
    SI = S / S_O
    SO = SI*OI/(w*SI+(1-w)*OI)
    return SO



#####            以下是用于生成表格的代码            #####


def get_csv_txt(result, lowhigh, algorithms, tasknamebenchmark, num,Ts, T_max, mode):
    '表格生成代码，以“块”为生成单位，块为二维子表，最后把块以特定顺序堆叠，再统一生成一张完整表格'
    try:
        len(Ts)
    except:
        Ts = [Ts]
    
    dct = dict()
    for save_data in result:
        algorithm_name = save_data['algorithm_name']
        is_constraint = save_data['constraint']
        args = save_data['args']
        offline_x = save_data['offline_x']
        offline_y = save_data['offline_y']
        offline_cons = save_data['offline_cons']
        save_xs = save_data['xs']
        save_ys = save_data['ys']
        save_cons = save_data['cons']
        other = save_data['other']
        label = f'{algorithm_name}/{args.task}/{args.benchmark}/{args.num}/{args.seed}/{args.low}/{args.high}'
        dct[label] = save_data
        
    def get_head_block():
        name = tasknamebenchmark.replace(' ','_')+'_'+ mode + '_'+str(lowhigh[0])+'-'+str(lowhigh[1])+'-'+str(num)
        block = [[name],
                 ['']]
        return block
    
    def row_title_block(T):
        block = [[f'T={T}','','','',''],
                 ['MAX', 'SI', 'OI', 'SO', 'SO-w']]
        return block
    
    def col_head_block(algorithm):
        block = [[algorithm]]
        return block
    
    def data_block(lowhigh,algorithm, dataset, benchmark,num, T, T_max, mode):
        "数据块的生成部分"
        offline_best = '-'
        
        def get_cell(li):
            ret = '-'
            if len(li) >= 6:
                mean = np.mean(li)
                std = np.std(li)
                ret = f'{mean:.3f}±{std:.3f}'
            return ret
        maxv_li = []
        si_li = []
        oi_li = []
        so_li = []
        sow_li = []
        offline_best_li = []
        for seed in range(8):
            # 算法最优
            try:
                save_data = dct[f'{algorithm}/{dataset}/{benchmark}/{num}/{seed}/{lowhigh[0]}/{lowhigh[1]}']
            except:
                continue
            
            # if offline_best == '-':
            #     offline_best = f'{np.max(save_data["offline_y"]):.3f}'
            
            offline_best_li.append(np.max(save_data["offline_y"]))
            maxv = get_maxv(save_data,T, T_max,mode)
            si = get_SI(save_data, T, T_max, mode)
            oi = get_OI(save_data, T, T_max, mode)
            so = get_SO(save_data, T, T_max, mode)
            sow = get_SOW(save_data, T, T_max, mode)
            if maxv is not None: maxv_li.append(maxv)
            if si is not None: si_li.append(si)
            if oi is not None: oi_li.append(oi)
            if so is not None: so_li.append(so)
            if sow is not None: sow_li.append(sow)
        
        block = [[get_cell(maxv_li), get_cell(si_li), get_cell(oi_li), get_cell(so_li), get_cell(sow_li)]]
        
        offline_best = get_cell(offline_best_li)
        return block,offline_best
    
    def generate(lowhigh,algorithms, dataset,benchmark, num,Ts,T_max, mode):
        blocks = []
        block_row = [get_head_block()]
        for T in Ts:
            block_row.append(row_title_block(T))
        blocks.append(block_row)
        
        offline_best = '-'
        for algorithm in algorithms:
            block_row = [col_head_block(algorithm)]
            for T in Ts:
                block, offline_best_ = data_block(lowhigh, algorithm,dataset,benchmark,num,T, T_max,mode)
                if offline_best_ != '-':
                    offline_best = offline_best_
                block_row.append(block)
            blocks.append(block_row)
            
        offline_best_block = [['OFFLINE_BEST', offline_best]]
        block_row = [offline_best_block]
        blocks.append(block_row)
        
        return blocks
    
    def convert_to_csv(blocks):
        # print(blocks)
        after = []
        for block_row in blocks:
            for row in range(len(block_row[0])):
                line = []
                for block in block_row:
                    for cell in block[row]:
                        line.append(cell)
                after.append(line)
        txt = ''
        for line in after:
            txt += ','.join(line) + '\n'
        return txt

    dataset,benchmark = tasknamebenchmark.split(' ')
    blocks = generate(lowhigh, algorithms, dataset,benchmark,num,Ts,T_max,mode)
    txt = convert_to_csv(blocks)
    
    return txt


def load(rootpath):
    '''从目标目录中读取所有结果文件，对于约束数据，将约束不满足的值设为最差值'''
    results = []
    for file in tqdm.tqdm(os.listdir(rootpath), 'loading'):
        with open(os.path.join(rootpath, file), 'rb') as f:
            save_data = pickle.load(f)
        
        save_data['offline_x']  = np.array(save_data['offline_x'])
        save_data['offline_y']  = np.array(save_data['offline_y'])
        save_data['xs'] = np.array(save_data['xs'])
        save_data['ys'] = np.array(save_data['ys'])
        save_data['file_name'] = file
        
        if save_data['constraint']:
            offline_worse = np.min(save_data['offline_y'])
            
            save_data['offline_cons'] = np.array(save_data['offline_cons'])
            idxs = np.where(np.array(save_data['offline_cons']) < 0)[0]
            save_data['offline_y'] [idxs] = offline_worse
            
            
            save_data['cons'] = np.array(save_data['cons'])
            idxs = np.where(np.array(save_data['cons']) < 0)[:2]
            save_data['ys'][idxs] = offline_worse
        results.append(save_data)
    return results

CACHE = dict()
def main(rootpath, save_to, lowhigh,  tasknamebenchmark, algorithms=None, num=1000, T=150,T_max=150, mode='max'):
    '以指定参数生成表格并保存'
    
    if os.path.exists(rootpath) == False:
        print(f'skip {rootpath} cause it does\'t exists')
        return
    
    # 读取保存的结果
    if rootpath in CACHE:
        results = CACHE[rootpath]
    else:
        results = load(rootpath)
        CACHE[rootpath] = results
    
    # # 初始化默认值，不指定算法则统计所有算法
    algolist = [x['algorithm_name'] for x in results]
    algolist = sorted(list(set(algolist)))
    if algorithms is None:
        algorithms = algolist
    
    # 生成csv表格
    csv_txt = get_csv_txt(results, lowhigh, algorithms, tasknamebenchmark, num, T, T_max, mode)
    
    # 保存csv表格
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, 'w') as f:
        f.write(csv_txt)
    
    print('finish to gather results:', save_to)


        
# 生成无约束表格

tasknamebenchmarks = ['gtopx_data 2',
                     'gtopx_data 3',
                     'gtopx_data 4',
                     'gtopx_data 6',
                     'mujoco_data 1',
                     'mujoco_data 2']
lowhighs = [(0,50),
            (10,60),
            (20,70),
            (25,75),
            (30,80),
            (40,90),
            (0,60),
            (0,70),
            (0,80),
            (0,90)]
modes = ['max','min','median']


# tasknamebenchmark = 'gtopx_data 2'
# lowhigh = [25,75]
# mode = 'max'


constraint = False
algorithm = None
T = [50,100,150]
num = 1000
T_max = 150

pars = []
for mode in modes:    
    for tasknamebenchmark in tasknamebenchmarks:
        for lowhigh in lowhighs:
            pars.append((mode,tasknamebenchmark, lowhigh))

for mode, tasknamebenchmark, lowhigh in tqdm.tqdm(pars, desc='result_gather'):
    name = mode + '_'+ tasknamebenchmark.replace(' ','_')+'_'+ str(lowhigh[0])+'-'+str(lowhigh[1])+'-'+str(num)
    if constraint:
        rootpath = os.path.abspath(os.path.join(__file__, '..','..','results','constraint'))
        save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary','details',mode, 'cons',f'results_c_{name}.csv'))
    else:
        rootpath = os.path.abspath(os.path.join(__file__, '..','..','results','unconstraint'))
        save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary','details',mode, 'uncons',f'results_u_{name}.csv'))
    main(rootpath, save_to,lowhigh,tasknamebenchmark, algorithm, num, T, T_max, mode)

# 生成有约束表格

tasknamebenchmarks = [
                     'gtopx_data 5',
                     'gtopx_data 1',
                     'gtopx_data 7',
                     'cec_data 1',
                     'cec_data 2',
                     'cec_data 3',
                     'cec_data 4',
                     'cec_data 5'
                     ]

lowhighs = [(0,50),
            (10,60),
            (20,70),
            (25,75),
            (30,80),
            (40,90),
            (0,60),
            (0,70),
            (0,80),
            (0,90)]
modes = ['max','min','median']


constraint = True
algorithm = None
T = [50,100,150]
num = 1000
T_max = 150

pars = []
for mode in modes:    
    for tasknamebenchmark in tasknamebenchmarks:
        for lowhigh in lowhighs:
            pars.append((mode,tasknamebenchmark, lowhigh))

for mode, tasknamebenchmark, lowhigh in tqdm.tqdm(pars, desc='result_gather'):
    name = mode + '_'+ tasknamebenchmark.replace(' ','_')+'_'+ str(lowhigh[0])+'-'+str(lowhigh[1])+'-'+str(num)
    if constraint:
        rootpath = os.path.abspath(os.path.join(__file__, '..','..','results','constraint'))
        save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary','details',mode, 'cons',f'results_c_{name}.csv'))
    else:
        rootpath = os.path.abspath(os.path.join(__file__, '..','..','results','unconstraint'))
        save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary','details',mode, 'uncons',f'results_u_{name}.csv'))
    main(rootpath, save_to,lowhigh,tasknamebenchmark, algorithm, num, T, T_max, mode)


## 压缩
details_path = os.path.abspath(os.path.join(__file__, '..','..','results','summary','details'))
os.chdir(os.path.abspath(os.path.join(__file__, '..','..','results','summary')))
os.system(f'zip -r ./details.zip ./details')