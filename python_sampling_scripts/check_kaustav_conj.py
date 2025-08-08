import argparse   # to handle command-line arguments
import numpy as np
import torch
import sys
import os
import time

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kaustav_conj.utils import h, H, bK, block_spec, M_to_A
from kaustav_conj.core import build_cost_function, get_b_best

#Store all parameters specified via command line:
parser = argparse.ArgumentParser()

# add all arguments to parser
parser.add_argument("--d", type=int, help="d = length of n-vectors")
parser.add_argument("--N_n", type=int, help="N_n = number of sampled n-vectors per pair (d, lamb)")
parser.add_argument("--N_init", type=int, help="N_init = number of random initializations per n-vector")
parser.add_argument("--N_steps", type=int, help="N_steps = number of gradient descend steps")
parser.add_argument("--learning_rate", type=float, help="learning_rate = number of gradient descend steps")
parser.add_argument("--eps", type=float, help="eps = threshold for inequality checks")
parser.add_argument("--print_more", type=bool, help="print_more = bool, print output for all N_init initializations or only at the end")

args = parser.parse_args() # Parse the arguments provided by the user via the command line
locals().update(vars(args)) # This defines variables d, ... ready to use in the code

print("## INPUT PARAMETERS ##\n\n", 
    f"d: {d}\n", 
    f"N_n: {N_n}\n", 
    f"N_init: {N_init}\n", 
    f"N_steps: {N_steps}\n", 
    f"learning_rate: {learning_rate}\n", 
    f"eps: {eps}\n",
    f"print_more: {print_more}\n")

print("## OUTPUT ##")

# Record the start time
time0 = time.time()/3600

conjecture_holds = True
for lamb in range(int(np.ceil(d/2)), d):
    print(F"\n{'='*40}\n{'='*40}\nCASE d = {d}, lambda = {lamb}\n{'='*40}\n{'='*40}\n")
    for _ in range(N_n):
        n = np.random.rand(d)
        b_best_conj = bK(n, lamb)
        U_best, b_best_num, H_best, conjecture_holds = get_b_best(n, lamb, N_init=N_init, N_steps=N_steps,learning_rate=learning_rate, eps=eps, print_more=print_more)
        if conjecture_holds == False:
            break
    if conjecture_holds == False:
        break

print(f"\n{'='*40}\n{'='*40}\nConjecture holds: {conjecture_holds}")
if not conjecture_holds:
    print(f"!!!!! COUNTEREXAMPLE FOUND within threshold for inequality checks of = {eps}. Exiting code before time.")
    print(f"n: {n}")
    print(f"U_best: {U_best}")
    print(f"b_best_num: {b_best_num}")
print(f"{'='*40}\n{'='*40}\n")
print("## FINISHED RUNNING check_kaustav_conj.py ## \n")
    