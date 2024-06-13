from logger import Logger
from utils import spearman
from data_c import TaskDataset
import torch
from trainers import ConservativeObjectiveModel
from trainers import VAETrainer
from nets import ForwardModel
from nets import SequentialVAE
import tensorflow as tf
import numpy as np
import click
import pandas as pd
import json
import os, sys
from workshop.Benchmark_new.benchmark.Taskdata import OfflineTask
from workshop.Benchmark_new.constraint.par import get_parser
# 标准化scores
def stand_scores(data):
    mean = np.mean(data)
    std = np.std(data)
    stand_scores = (data - mean) / std
    return stand_scores

def coms_cleaned(args):
    # create the logger and export the experiment parameters
    task = OfflineTask(args.task, args.benchmark, args.seed)
    task.sample_bound(args.num, args.low, args.high)
    x = np.array(task.x).astype(np.float32)
    y = np.array(task.y).astype(np.float32)
    cons = task.cons.astype(np.float32)
    # cons_mean = np.mean(cons).astype(np.float64)
    # cons_std = np.std(cons).astype(np.float64)
    # x = stand_scores(ori_x).astype(np.float32)
    # y = stand_scores(ori_y.reshape(len(ori_y), 1)).astype(np.float32)
    # cons = stand_scores(cons).astype(np.float32)

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
    print("111:", len(train_dataset))
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    validate_dl = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)


    # train the forward model
    trainer.launch(train_dl, validate_dl, 50)

    # load the surrogate model
    # dhs_model.load_state_dict(torch.load('model_para_surrogate_gtopx8.pkl'))

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
        prob_k = 6

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
            print(step)
            # print(constraint)
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
            

    # score_2 = []
    # q = 0
    # constraints = constraints[0]
    # for m in range(len(constraints)):
    #     k = 0
    #     for e in range(len(constraints[0])):
    #         if constraints[m][e] < 0:
    #             k = k + 1
    #     if k == 0:
    #         score_2.append(scores[0][m])
    #         q = q + 1
    #     else:
    #         score_2.append(min(initial_y))
    # print("best:", min(score_2))
    print(scores)
    save_name = 'PRIME' + '-' + args.task +'-'+ str(args.benchmark)+ '-' + str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
    with open ('/root/workshop/Benchmark_new/con_results_last/' + save_name + '.pkl', 'wb') as f:
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