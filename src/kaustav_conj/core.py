"""
Core functions

This module contains the main implementations for .... TODO
"""

import numpy as np
import torch
from kaustav_conj.utils import H, M_to_A, block_spec

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