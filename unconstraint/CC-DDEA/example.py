import os,sys
import torch
import numpy as np
import cc_ddea
from exper import get_parser
from soo_bench.Taskdata import *
from soo_bench.utils import calculate_SI

gpu_device = torch.device('cuda:0')

iter_max = 50

max_run = 1


def main(args):
    set_use_cache()
    ys = []
    task = OfflineTask(args.task, args.benchmark, seed = args.seed)
    if args.num == 0:
        args.num = 100
    num = args.num*task.var_num
    if args.sample_method == 'sample_bound':
        task.sample_bound(num, args.low, args.high)
    elif args.sample_method == 'sample_limit':
        task.sample_limit(num, args.low, args.high)
    x = task.x
    y = task.y
    xl = torch.Tensor(task.xl).to(gpu_device)
    xu = torch.Tensor(task.xu).to(gpu_device)

    for i in range(max_run): # max_run == 1
        x_train = x
        y_train = -y

        lb, ub = task.xl, task.xu
        d = task.var_num
        indice = np.argsort(y[:, 0])[-128:][::-1].reshape(-1, 1)
        pop = x[indice].squeeze(1)
        iter_sub_max = 10
        ys.append(max(y))
        x, save_xs = cc_ddea.run(
            pop=pop,
            iter_max=iter_max,
            iter_sub_max=iter_sub_max,
            samples=(x_train, y_train),
            lower_bound=lb,
            upper_bound=ub,
            n_group_init=10,
            group_update_gap=8,
            gpu_device=gpu_device,
            lr_individual=0.1,
            surrogate_update_gap=int(8/2),
            n_top_rate=0.1,
            n_random_children_rate=0.1,
            xl = xl,
            xu = xu
        )

    
    save_ys = []
    save_cons = []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)
    
    print('scores:', np.max(save_ys, axis=1))
    
    # save results
    def save_data(algorithm_name, args, offline_x, offline_y, offline_cons, save_xs, save_ys, save_cons,other=None,is_constraint=False, level=2):
        save_name = algorithm_name + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(level):
            current_dir = os.path.dirname(current_dir)
        
        if is_constraint:
            os.makedirs(os.path.join(current_dir, 'results','constraint'), exist_ok=True)
            path = os.path.join(current_dir, 'results','constraint', save_name +'.savedata.pkl')
        else:
            os.makedirs(os.path.join(current_dir, 'results','unconstraint'), exist_ok=True)
            path = os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')

        print('saving to: ', path)
        
        offline_x = np.array(offline_x)
        offline_y = np.array(offline_y).flatten()
        if is_constraint:
            offline_cons = np.array(offline_cons)
        else:
            offline_cons = None
        for i in range(len(save_xs)):
            save_xs[i] = np.array(save_xs[i])
            save_ys[i] = np.array(save_ys[i]).flatten()
            if is_constraint:
                save_cons[i] = np.array(save_cons[i])
        if not is_constraint:
            save_cons = None

        save_data = {
            'algorithm_name': algorithm_name,
            'constraint': is_constraint,
            'args': args,
            'offline_x': offline_x,
            'offline_y': offline_y,
            'offline_cons': offline_cons,
            'xs':save_xs,
            'ys':save_ys,
            'cons':save_cons,
            'other': other
        }
        with open (path, 'wb') as f:
            import pickle
            pickle.dump(save_data, f)
    

    save_data('CCDDEA', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, is_constraint=False, level=2)

if __name__ == '__main__':
    args = get_parser()
    if args.change_optimization_step > 0:
        iter_max = args.change_optimization_step

    save_name = 'CCDDEA' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')):
        print('Already exists:', save_name +'.savedata.pkl')
        exit()

    main(args)
