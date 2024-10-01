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
    task.sample_bound(args.num, args.low, args.high)
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
    particle_train_gradient_steps = 50
    score_max = []

    if not fast:

        scores = []
        predictions = []
        constraints = []

        solution = xt

        score, constraint = task.predict(solution.numpy())

    for step in range(1, 1 + particle_evaluate_gradient_steps):

        # update the set of solution particles
        xt = trainer.optimize(xt, 1, training=False)
        final_xt = trainer.optimize(
            xt, particle_train_gradient_steps, training=False)

        if True:

            solution = xt
            # evaluate the solutions found by the model
            score, constraint = task.predict(solution.numpy())
            prediction = forward_model(xt, training=False).numpy()
            final_prediction = forward_model(final_xt, training=False).numpy()
            score_feasible = []
            score_infeasible = []
            q = 0 
            for m in range(len(constraint)):
                k = 0
                for e in range(len(constraint[0])):
                    if constraint[m][e] < 0:
                        k = k + 1
                if k == 0:
                    score_feasible.append(score[m])
                    q = q + 1
                else:
                    score_infeasible.append(score[m])
            if q == 0:
                score_max.append(min(y))
            else:
                score_max.append(max(score_feasible))


        if not fast:

            scores.append(score_max)
            constraints.append(constraint)
            predictions.append(prediction)
            

    print('scores:', scores)
    save_name = 'PRIME' + '-' + args.task +'-'+ str(args.benchmark)+ '-' + str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(2):
        current_dir = os.path.dirname(current_dir)
    os.makedirs(os.path.join(current_dir, 'results','constraint'), exist_ok=True)
    path = os.path.join(current_dir, 'results','constraint', save_name +'.pkl')
    print('saving to: ', path)
    with open (path, 'wb') as f:
        import pickle
        pickle.dump(scores, f)



# run COMs using the command line interface
if __name__ == '__main__':
    args = get_parser()
    def set_seed(seed):
        import random
        import torch
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(args.seed)
    coms_cleaned(args)