import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()

from soo_bench.Taskdata import *

import tensorflow as tf
import numpy as np
import os
import torch

from nets import ForwardModel, Encoder, DiscreteDecoder, ContinuousDecoder
from trainers import CBAS, WeightedVAE, Ensemble
from cbasdata import build_pipeline
import argparse
from tqdm import tqdm
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser(description="pairwise offline")
parser.add_argument("--task", type=str, default="gtopx_data")
parser.add_argument("--low", type=int, default=25)
parser.add_argument("--high", type=int, default=75)
parser.add_argument("--benchmark", type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--num", type=int, default=1000)
parser.add_argument('--sample_method', default="sample_bound", type=str)
parser.add_argument('--change_optimization_step', default=-1, type=int)

parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument("--use_vae", type=bool, default=False)
parser.add_argument("--vae_hidden_size", type=int, default=64)
parser.add_argument("--vae_latent_size", type=int, default=256)
parser.add_argument("--vae_activation", type=str, default="relu")
parser.add_argument("--vae_kernel_size", type=int, default=3)
parser.add_argument("--vae_num_blocks", type=int, default=3)
parser.add_argument("--vae_lr", type=float, default=0.0003) # 0.0003
parser.add_argument("--vae_beta", type=float, default=1) # 1.0
parser.add_argument("--vae_batch_size", type=int, default=100) # 100
parser.add_argument("--val_size", type=int, default=200)
parser.add_argument("--vae_epochs", type=int, default=20)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument('--embedding_size', type=int, default=256)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--initial_max_std", type=float, default=0.2)
parser.add_argument("--initial_min_std", type=float, default=0.1)
parser.add_argument("--bootstraps", type=int, default=5)
parser.add_argument("--ensemble_lr", type=float, default=0.0003) # 0.0003
parser.add_argument("--ensemble_batch_size", type=int, default=100)
parser.add_argument("--ensemble_epochs", type=int, default=100)
parser.add_argument("--solver_samples", type=int, default=128)
parser.add_argument("--optimize_ground_truth", type=bool, default=False)
parser.add_argument("--do_evaluation", type=bool, default=True)
parser.add_argument('--offline_epochs', type=int, default=200)
parser.add_argument('--online_batches', type=int, default=10)
parser.add_argument('--online_epochs', type=int, default=10)
parser.add_argument('--autofocus_epochs', type=int, default=10)
parser.add_argument('--iterations', type=int, default=20)
parser.add_argument('--percentile', type=float, default=80.0)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda:' + str(args.device))


save_name = 'CBAS' + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high)
current_dir = os.path.dirname(os.path.abspath(__file__))
for i in range(2):
    current_dir = os.path.dirname(current_dir)
if os.path.exists(os.path.join(current_dir, 'results','unconstraint', save_name +'.savedata.pkl')):
    print('Already exists:', save_name +'.savedata.pkl')
    exit()

def set_seed(seed):
    import torch
    import random
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(args.seed)

def autofocused_cbas():
    set_use_cache()
    task = OfflineTask(args.task, args.benchmark, args.seed)
    num = args.num*task.var_num
    if args.sample_method == 'sample_bound':
        task.sample_bound(num, args.low, args.high)
    elif args.sample_method == 'sample_limit':
        task.sample_limit(num, args.low, args.high)
    
    
        
    x, y = task.x.astype(np.float32), task.y.astype(np.float32)
    xl = task.xu
    xu = task.xl
    first_score = np.max(task.y)
    input_shape = x.shape[1:]
    task.input_shape = input_shape
    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, w=np.ones_like(y),
        val_size=args.val_size,
        batch_size=args.ensemble_batch_size,
        bootstraps=args.bootstraps)

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task,
        embedding_size=args.embedding_size,
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

    # train the model for an additional number of epochs
    ensemble.launch(train_data,
                    val_data,
                    args.ensemble_epochs)

    decoder = ContinuousDecoder

    # build the encoder and decoder distribution and the p model
    p_encoder = Encoder(task, args.latent_size,
                        args.embedding_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        initial_max_std=args.initial_max_std,
                        initial_min_std=args.initial_min_std)
    p_decoder = decoder(task, args.latent_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        initial_max_std=args.initial_max_std,
                        initial_min_std=args.initial_min_std)
    p_vae = WeightedVAE(p_encoder, p_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=args.vae_lr,
                        vae_beta=args.vae_beta)

    # build a weighted data set
    train_data, val_data = build_pipeline(
        x=x, y=y, w=np.ones_like(task.y),
        batch_size=args.vae_batch_size,
        val_size=args.val_size)

    # train the initial vae fit to the original data distribution
    p_vae.launch(train_data,
                 val_data,
                 args.offline_epochs)

    # build the encoder and decoder distribution and the p model
    q_encoder = Encoder(task, latent_size=args.latent_size,
                        embedding_size=args.embedding_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        initial_max_std=args.initial_max_std,
                        initial_min_std=args.initial_min_std)
    q_decoder = decoder(task, args.latent_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        initial_max_std=args.initial_max_std,
                        initial_min_std=args.initial_min_std)
    q_vae = WeightedVAE(q_encoder, q_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=args.vae_lr,
                        vae_beta=args.vae_beta)

    # create the cbas importance weight generator
    cbas = CBAS(ensemble,
                p_vae,
                q_vae,
                latent_size=args.latent_size)

    # train and validate the q_vae using online samples
    q_encoder.set_weights(p_encoder.get_weights())
    q_decoder.set_weights(p_decoder.get_weights())
    for i in tqdm(range(args.iterations)):
        # generate an importance weighted dataset
        x_t, y_t, w = cbas.generate_data(
            args.online_batches,
            args.vae_batch_size,
            args.percentile)

        # build a weighted data set
        train_data, val_data = build_pipeline(
            x=x_t.numpy(),
            y=y_t.numpy(),
            w=w.numpy(),
            batch_size=args.vae_batch_size,
            val_size=args.val_size)

        # train a vae fit using weighted maximum likelihood
        start_epoch = args.online_epochs * i + \
                      args.offline_epochs
        q_vae.launch(train_data,
                     val_data,
                     args.online_epochs,
                     start_epoch=start_epoch)

        # autofocus the forward model using importance weights
        v = cbas.autofocus_weights(
            x, batch_size=args.ensemble_batch_size)
        train_data, val_data = build_pipeline(
            x=x, y=y, w=v.numpy(),
            bootstraps=args.bootstraps,
            batch_size=args.ensemble_batch_size,
            val_size=args.val_size)

        # train a vae fit using weighted maximum likelihood
        start_epoch = args.autofocus_epochs * i + \
                      args.ensemble_epochs
        ensemble.launch(train_data,
                        val_data,
                        args.autofocus_epochs,
                        start_epoch=start_epoch)
        
    n_steps = 100
    if args.change_optimization_step > 0:
        n_steps = args.change_optimization_step
    
    save_xs = []
    z = tf.random.normal([args.solver_samples, args.latent_size])
    tmp = z
    q_dx = q_decoder.get_distribution(tmp, training=False)
    x_t = q_dx.sample()
    x_t = tf.clip_by_value(x_t, xl, xu)
    for i in range(n_steps):
        save_xs.append(x_t)
    
    save_ys = []
    save_cons = []
    for x in save_xs:
        score, cons = task.predict(x)
        save_ys.append(score)
        save_cons.append(cons)

    print('scores:', np.max(save_ys, axis=1))
    print('offline_best:', np.max(task.y))
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
    save_data('CBAS', args, task.x, task.y, task.cons, save_xs, save_ys, save_cons, is_constraint=False, level=2)

t0 = time.time()
autofocused_cbas()
print('time:', time.time()-t0)