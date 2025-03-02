import os
import argparse
from itertools import chain
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, default="gtopx_data")
    parser.add_argument("--benchmark", type = int, default=2)
    parser.add_argument("--num", type = int, default=100)
    parser.add_argument("--low", type = int, default=25)
    parser.add_argument("--high", type = int, default=75)
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--sample_method', default="sample_bound", type=str)
    parser.add_argument('--change_optimization_step', default=-1, type=int)
    
    parser.add_argument("--random_seed", type = int, default=2)
    parser.add_argument("--save_model", type = bool, default=False)
    parser.add_argument("--e_train", type = bool, default=True)
    parser.add_argument("--init_m", type = float, default=0.02)
    parser.add_argument("--Ld_K", type = int, default=64)
    parser.add_argument("--Ld_K_max", type=int, default=64)
    parser.add_argument("--surrogate_hidden", type=int, default=2048)
    parser.add_argument("--surrogate_lr", type=float, default = 0.0003)
    parser.add_argument("--train_batch", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--train_epoch", type=int, default=64)
    parser.add_argument("--online_solutions_batch", type = int, default=128)
    parser.add_argument("--opt_config", type=str, default={"energy_opt": True, "opt_steps": 100})

    return parser.parse_args()

