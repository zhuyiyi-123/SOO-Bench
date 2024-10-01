import os
import argparse
from itertools import chain
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, default="gtopx_data")
    parser.add_argument("--low", type = int, default=50)
    parser.add_argument("--high", type = int, default=100)
    parser.add_argument("--benchmark", type = int, default=1)
    parser.add_argument("--num", type = int, default=1000)#50000
    parser.add_argument("--seed", type = int, default=0)
    

    return parser.parse_args()

