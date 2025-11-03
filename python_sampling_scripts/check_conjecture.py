import argparse   # to handle command-line arguments
import numpy as np
import torch
import sys
import os
import time

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# from kaustav_conj.utils import ...
# from kaustav_conj.core import ...

#Store all parameters specified via command line:
parser = argparse.ArgumentParser()

# add all arguments to parser

parser.add_argument("--int_partition", type=str, help="int_partition = comma-separated non-increasing integers (e.g., '2,1,1')")
parser.add_argument("--N_n", type=int, help="N_n = number of sampled n-vectors")
parser.add_argument("--N_init", type=int, help="N_init = number of random initializations per n-vector")
parser.add_argument("--N_steps", type=int, help="N_steps = number of gradient descend steps")
parser.add_argument("--learning_rate", type=float, help="learning_rate = learning rate for gradient descend method")
parser.add_argument("--eps", type=float, help="eps = convergence threshold")
# parser.add_argument("--verbosity", type=int, help="verbosity = int (should be 0, 1, or 2)")

args = parser.parse_args() # Parse the arguments provided by the user via the command line
args.int_partition = [int(x) for x in args.int_partition.split(',')] if args.int_partition else []
locals().update(vars(args)) # This defines variables d, ... ready to use in the code

# Convert the string partition to a list of integers
if isinstance(int_partition, str):
    int_partition = [int(x) for x in int_partition.split(',')]

print("## INPUT PARAMETERS ##\n\n", 
    f"int_partition: {int_partition}\n", 
    f"N_n: {N_n}\n", 
    f"N_init: {N_init}\n", 
    f"N_steps: {N_steps}\n", 
    f"learning_rate: {learning_rate}\n",
    f"eps: {eps}\n"
#    f"verbosity: {verbosity}\n"
)

print("## OUTPUT ##")