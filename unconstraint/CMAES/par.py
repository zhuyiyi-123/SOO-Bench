import os
import argparse
from itertools import chain
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, default="gtopx_data")
    parser.add_argument("--low", type = int, default=25)
    parser.add_argument("--high", type = int, default=75)
    parser.add_argument("--benchmark", type = int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--num", type = int, default=1000)
    parser.add_argument('--sample_method', default="sample_bound", type=str)
    parser.add_argument('--change_optimization_step', default=-1, type=int)
    
    parser.add_argument("--use_vae", type = bool, default=False)
    parser.add_argument("--vae_hidden_size", type=int, default=64)
    parser.add_argument("--vae_latent_size", type=int, default=256)
    parser.add_argument("--vae_activation", type=str, default ="relu")
    parser.add_argument("--vae_kernel_size", type=int, default=3)
    parser.add_argument("--vae_num_blocks", type=int, default=3)
    parser.add_argument("--vae_lr", type=float, default=0.001)
    parser.add_argument("--vae_beta", type = float, default=0.01)
    parser.add_argument("--vae_batch_size", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--vae_epochs", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type = int, default=1)
    parser.add_argument("--initial_max_std", type =float, default =0.2)
    parser.add_argument("--initial_min_std", type =float, default=0.1)
    parser.add_argument("--bootstraps", type = int, default=5)
    parser.add_argument("--ensemble_lr", type =float, default =0.001)
    parser.add_argument("--ensemble_batch_size", type=int, default =100)
    parser.add_argument("--ensemble_epochs", type =int, default =50)
    parser.add_argument("--solver_samples", type =int, default = 128)
    parser.add_argument("--optimize_ground_truth", type = bool, default = False)
    parser.add_argument("--cma_sigma", type =float, default = 0.5)
    parser.add_argument("--cma_max_iterations", type =int, default = 150)
    parser.add_argument("--do_evaluation", type =bool, default = True)
    

    return parser.parse_args()

