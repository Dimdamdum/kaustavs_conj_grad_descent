"""
Core functions

This module contains the main implementations for .... TODO
"""

import numpy as np
import torch
from kaustav_conj.utils import H, M_to_A, block_spec, nK

def build_cost_function(n, lamb):
    """
    Creates the cost function to be minimized via gradient descent, for the given n.
    
    Parameters:
    -----------
    n : list (float)
        Enters cost function as a parameter.
    lamb : int
        The partition parameter.
        
    Returns:
    --------
    cost_function
        A function that takes as input a torch square matrix M of size d = len(n),
        converts it into an antihermitian matrix A (using the utility function M_to_A),
        computes the unitary U = exp(A), computes U @ diag(n) @ U.adjoint(),
        computes block diag spectrum b of the former, and returns entropic measure -H(b).
    """
    d = len(n)
    D = torch.diag(torch.tensor(n, dtype=torch.cdouble))
    if (lamb < d/2 or lamb >= d):
        print(f"Warning: build_cost_function called with lambda={lamb}, which is outside [len(n)/2,len(n) - 1]")
        return np.nan  # Return NaN for invalid inputs
    eps = 1e-15/2
    # check all entries of n are >  - eps and <= 1 + eps, and return nan if fail
    if not np.all((np.array(n) > -eps) & (np.array(n) <= 1 + eps)):
        print("Warning: entries of n must be in (-eps, 1+eps]")
        return np.nan
    # define  cost_function
    def cost_function(M):
        # check M is d x d and otherwise print warning and return nan
        if not (isinstance(M, torch.Tensor) and M.shape == (d, d)):
            print(f"Warning: M must be a torch.Tensor of shape ({d}, {d})")
            return np.nan
        A = M_to_A(M) # get antihermitian matrix corresponding to M
        U = torch.matrix_exp(A) # get unitary 
        b = block_spec(U @ D @ U.adjoint(), lamb)
        return -H(torch.real(b))
    return cost_function

def get_b_best(n, lamb, rand_range=1., N_init=1, N_steps=1000, learning_rate=0.01):
    """
    Returns
    
    Parameters:
    -----------
    n : list (float)
        The main parameter/spectrum.
    lamb : int
        The partition parameter.
    rand_range: float, optional
        Parameter for scaling initialization tensor.
    N_init: int, optional
        Number of random initializations for optimization (default: 1).
    N_steps: int, optional
        Number of gradient descent steps per initialization (default: 1000).
    learning_rate: float, optional
        Learning rate for gradient descent (default: 0.01).
        
    Returns:
    --------
    U_best_best : torch.Tensor (square matrix of size d = len(n) )
        Optimal unitary leading to b_best_best, optimized both via gradient descent and over different initializations.
    b_best_best : list (float)
        The block spectrum maximizing the entropic function H. Obtained by conjugating diagonal(n) with U_best_best.
    H_best_best : float
        H(b_best_best)
    """
    d = len(n)

    # build cost function
    cost_function = build_cost_function(n, lamb)

    # useful variables
    U_best = torch.empty(d, d, dtype=torch.cdouble)
    b_best = np.empty(d)
    H_best = 0.0
    D = torch.diag(torch.tensor(n, dtype=torch.cdouble))
    b_best_conj = nK(n, lamb)
    H_best_conj = H(b_best_conj)

    # initialize storage lists
    U_best_list = []
    b_best_list = []
    H_best_list = []

    print(f"\n{'='*40}\nStarting get_b_best optimization\n{'='*40}")
    print(f"Parameters recap:")
    print(f"  n = {n}")
    print(f"  lambda = {lamb}")
    print(f"  rand_range = {rand_range}")
    print(f"  N_init = {N_init}")
    print(f"  N_steps = {N_steps}")
    print(f"  learning_rate = {learning_rate}\n")

    # start cycle for different initializations
    for i in range(N_init):
        print(f"\n{'='*20}\nStarting gradient descent run {i+1}/{N_init}\n{'='*20}")
        # initialize variable to be optimized, with entries in [-rand_range/2, rand_range/2]
        M = rand_range * (torch.rand(d, d) - 0.5 * torch.ones(d,d))
        M.requires_grad_(True)

        # define optimizer
        optimizer = torch.optim.Adam([M], lr=learning_rate)
        for step in range(N_steps):
            optimizer.zero_grad() # clear previous gradient
            loss = cost_function(M)
            loss.backward() # compute gradient
            optimizer.step() # update M
            if step % 100 == 0:
                print(f"Gradient descent, step # {step}")

        # store best unitary, block spec, and H value
        U_best = torch.matrix_exp(M_to_A(M))
        b_best_unsorted = torch.real(block_spec(U_best @ D @ U_best.adjoint(), lamb))
        b_best = torch.cat([
            torch.sort(b_best_unsorted[:lamb], descending=True).values,
            torch.sort(b_best_unsorted[lamb:], descending=True).values
        ]).detach().numpy()
        H_best = H(b_best)

        # print results
        print(f"\n{'='*20}\nResults of gradient descent run {i+1}/{N_init}\n{'='*20}")
        print(f"Numerical b_best: \n {b_best}")
        print(f"Conjectured b_best: \n {b_best_conj}")
        print(f"Norm of difference: \n {np.linalg.norm(b_best - b_best_conj)}")
        print(f"Conjectured H_best - numerical H_best (should be > 0): \n {H_best_conj - H_best}")

        # store results
        U_best_list.append(U_best)
        b_best_list.append(b_best)
        H_best_list.append(H_best)

    print(f"\n{'='*40}\nget_b_best optimization finished\n{'='*40}")
    # Find the index of the highest H_best value
    best_idx = np.argmax(H_best_list)
    U_best_best = U_best_list[best_idx]
    b_best_best = b_best_list[best_idx]
    H_best_best = H_best_list[best_idx]

    # print final results
    print(f"\n{'='*20}\n Final results, optimized over various initializations\n{'='*20}")
    print(f"Numerical b_best: \n {b_best_best}")
    print(f"Conjectured b_best: \n {b_best_conj}")
    print(f"Norm of difference: \n {np.linalg.norm(b_best_best - b_best_conj)}")
    print(f"Conjectured H_best - numerical H_best (should be > 0): \n {H_best_conj - H_best_best}")

    return U_best_best, b_best_best, H_best_best
    