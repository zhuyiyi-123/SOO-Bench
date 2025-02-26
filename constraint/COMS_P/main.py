from data_c import TaskDataset
import torch
from trainers import ConservativeObjectiveModel
from nets import ForwardModel
import tensorflow as tf
import numpy as np

from soo_bench.Taskdata import *

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from par import get_parser

def stand_scores(data):
    mean = np.mean(data)
    std = np.std(data)
    stand_scores = (data - mean) / std
    return stand_scores

def coms_cleaned(args):
    # export the experiment parameters
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, args.seed)
    num = args.num*task.var_num
    if args.sample_method == 'sample_bound':
        task.sample_bound(num, args.low, args.high)
    elif args.sample_method == 'sample_limit':
        task.sample_limit(num, args.low, args.high)
    x = np.array(task.x).astype(np.float32)
    y = np.array(task.y).astype(np.float32)
    cons = task.cons.astype(np.float32)

    input_shape = x.shape[1:]
    particle_lr = 0.05
    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))

    # make a neural network to predict scores
    forward_model = ForwardModel(
        input_shape, activations=['leaky_relu', 'leaky_relu'],
        hidden_size=64)

    # make a trainer for the forward model
    trainer = ConservativeObjectiveModel(
        forward_model, forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=0.0003, alpha=1.0,
        alpha_opt=tf.keras.optimizers.Adam, alpha_lr=0.01)

    dataset = TaskDataset(x, y, cons)
    train_dataset_size = int(len(dataset) * (1 - 0.2))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [train_dataset_size,
                                                                (len(dataset) - train_dataset_size)])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    validate_dl = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)


    # train the forward model
    trainer.launch(train_dl, validate_dl, 50)


    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=128)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    xt = initial_x
    fast = False
    particle_evaluate_gradient_steps = 100
    if args.change_optimization_step > 0:
        particle_evaluate_gradient_steps = args.change_optimization_step
    particle_train_gradient_steps = 50
    score_max = []

    if not fast:

        save_xs = []
        predictions = []

        solution = xt

    for step in range(1, 1 + particle_evaluate_gradient_steps):

        # update the set of solution particles
        xt = trainer.optimize(xt, 1, training=False)

        if True:

            solution = xt
            # evaluate the solutions found by the model
            prediction = forward_model(xt, training=False).numpy()


        if not fast:
            save_xs.append(np.array(solution.numpy()))
            predictions.append(prediction)


    save_ys, save_cons = [], []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)
    print('scores:', np.max(save_ys, axis=1))
    # save results
    def save_data(algorithm_name, args, offline_x, offline_y, offline_cons, save_xs, save_ys, save_cons,other = None, is_constraint=True, level=2):
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
    

    save_data('PRIME', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, other={'predictions':predictions}, is_constraint=True, level=2)



if __name__ == '__main__':
    args = get_parser()
    save_name = 'PRIME' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(current_dir, 'results','constraint', save_name +'.savedata.pkl')):
        print('Already exists:', save_name +'.savedata.pkl')
        exit()
    def set_seed(seed):
        import random
        import torch
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(args.seed)
    coms_cleaned(args)
