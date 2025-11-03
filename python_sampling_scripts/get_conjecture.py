import argparse   # to handle command-line arguments
import numpy as np
import torch
import sys
import os
import numpy as np
from more_itertools import set_partitions
from collections import Counter

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kaustav_conj.core import multi_get_b_best, find_conjecture
from kaustav_conj.utils import list_all_integer_partitions

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

print("## OUTPUT ##\n")

d = sum(int_partition)
output = [] # will contain 4-tuples [n, b_best_num, best_P, rel_err]
discarded_output = [] # will contain discarded 4-tuples
best_P_counter = Counter()
count_discarded_n = 0 # will count the amount of n vectors discarded due to the eps threshold
highest_rel_error = 0.

for _ in range(N_n):
    n = np.sort(np.random.rand(d)) # random spectrum with entries in [0,1]

     # perform optimization of our cost function
    U_best, b_best_num, H_best = multi_get_b_best(n, int_partition, N_init=N_init, N_steps=N_steps,learning_rate=learning_rate, verbosity=0)

    # get best_P, the set partition P of {0,...,d} that minimizes norm(avg(P,n) - b_best_num); and get rel_err = norm(norm(avg(best_P,n)) - b_best_num)/||n||
    rel_err, best_P = find_conjecture(n, b_best_num)

    # store output. If needed, also store best unitaries and H_best values
    output.append([n, b_best_num, best_P, rel_err])

    # print(f"n = {n}\nrel_err = {rel_err}\n") # print some output data, if needed

    if(rel_err < eps): # this check is for convergence. But it also leaves out cases where the true best b vector is not made of averages of n entries (if any such cases exist!)
        # Convert the inner lists of best_P to tuples to make them hashable, then increase by 1 best_P counter
        hashable_best_P = tuple(tuple(block) for block in best_P)
        best_P_counter[hashable_best_P] += 1
    else: # either no convergence, or the true best b is not made of averages of n entries
        count_discarded_n += 1
        discarded_output.append([n, b_best_num, best_P, rel_err])
        # print(...)
    
    # store highest relative error found
    if(rel_err > highest_rel_error):
        highest_rel_error = rel_err

print(f"Set partitions of {{1,...,{d}}} that occurred as best partitions, with counts:")
for key, value in best_P_counter.items():
    print(f"{key}: {value}")

print(f"\nNumber of sampled n vectors discarded because eps check failed: {count_discarded_n} out of {N_n}")
print(f"The highest relative error ||b_best_num - n||/||n|| found is {highest_rel_error}\n")