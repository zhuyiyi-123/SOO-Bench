from soo_bench.Taskdata import *
from data_ import build_pipeline
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
from trainers import Ensemble
from nets import ForwardModel
import tensorflow as tf
from par import get_parser
import numpy as np
import os,sys


import pickle
def default_name(taskname, benchmarkid, num,seed):
    return f'{taskname}_id{benchmarkid}_num{num}_seed{seed}.pkl'

def load(path):
    with open(path,'rb') as f:
        res = pickle.load(f)
    return res



def cma_es(args):
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, args.seed)
    task.sample_bound(args.num, args.low, args.high)
    first_score = np.max(task.y)
    x, y = task.x.astype(np.float32), task.y.astype(np.float32)
    input_shape = x.shape[1:]
    input_size = np.prod(input_shape)

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        input_shape,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        initial_max_std=args.initial_max_std,
        initial_min_std=args.initial_min_std)
        for b in range(args.bootstraps)]

    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=args.ensemble_lr)

    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, bootstraps=args.bootstraps,
        batch_size=args.ensemble_batch_size,
        val_size=args.val_size)

    # train the model for an additional number of epochs
    ensemble.launch(train_data, val_data, args.ensemble_epochs)

    # select the top 1 initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=args.solver_samples)[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = initial_x

    # create a fitness function for optimizing the expected task score
    def fitness(input_x):
        input_x = tf.reshape(input_x, input_shape)[tf.newaxis]
        if args.optimize_ground_truth:
            value = task.predict(input_x)
        else:
            value = ensemble.get_distribution(input_x).mean()
        return (-value[0].numpy()).tolist()[0]

    import cma
    result = []
    for i in range(args.solver_samples):
        xi = x[i].numpy().flatten().tolist()
        es = cma.CMAEvolutionStrategy(xi, args.cma_sigma)
        step = 0
        while not es.stop() and step < args.cma_max_iterations:
            solutions = es.ask()
            es.tell(solutions, [fitness(x) for x in solutions])
            step += 1
        result.append(
            tf.reshape(es.result.xbest, input_shape))
        print(f"CMA: {i + 1} / {args.solver_samples}")

    # convert the solution found by CMA-ES to a tensor
    x = tf.stack(result, axis=0)
    solution = x


    # save the current solution to the disk
    if args.do_evaluation:

        # evaluate the found solution
        score = task.predict(solution)[0].tolist()
        score.insert(0, first_score)
        print(score)
        print("score:", min(score), max(score))
        save_name = 'CMAES' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(2):
            current_dir = os.path.dirname(current_dir)
        os.makedirs(os.path.join(current_dir, 'results','unconstraint'), exist_ok=True)
        path = os.path.join(current_dir, 'results','unconstraint', save_name +'.pkl')
        print('saving to: ', path)
        with open (path, 'wb') as f:
            import pickle
            pickle.dump(score, f)


if __name__ == '__main__':
    args = get_parser()


    def set_seed(seed):
        import torch
        import random
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    set_seed(args.seed)
    cma_es(args)