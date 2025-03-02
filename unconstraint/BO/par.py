import os
import argparse
from itertools import chain
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, default="mujoco_data")
    parser.add_argument("--low", type = int, default=25)
    parser.add_argument("--high", type = int, default=75)
    parser.add_argument("--benchmark", type = int, default=2)
    parser.add_argument("--num", type = int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sample_method', default="sample_bound", type=str)
    parser.add_argument('--change_optimization_step', default=150, type=int)
    
    parser.add_argument("--bootstraps", type = int, default=5)
    parser.add_argument("--ensemble_batch_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default = 256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--initial_max_std", type=float, default=0.2)
    parser.add_argument("--initial_min_std", type=float, default=0.1)
    parser.add_argument("--ensemble_lr", type = float, default=0.001)
    parser.add_argument("--ensemble_epochs", type=int, default=100)
    parser.add_argument("--bo_gp_samples", type=int, default=500)
    parser.add_argument("--bo_batch_size", type=int, default=32)
    parser.add_argument("--optimize_ground_truth", type=bool, default=False)
    parser.add_argument("--bo_noise_se", type = float, default=0.1)
    parser.add_argument("--bo_num_restarts", type =int, default =10)
    parser.add_argument("--bo_batch_limit", type = int, default=5)
    parser.add_argument("--bo_maxiter", type = int, default=200)
    parser.add_argument("--bo_iterations", type =int, default =10)
    parser.add_argument("--bo_mc_samples", type=int, default =128)
    parser.add_argument("--solver_samples", type =int, default =128)
    parser.add_argument("--bo_raw_samples", type =int, default =128)
    parser.add_argument("--do_evaluation", type = bool, default = True)

    return parser.parse_args()

