"""
Utility functions

This module contains helper functions for matrix operations, basic entropic functions, eigenvalue computations.
"""

import numpy as np
from numpy import linalg as LA
import torch

def h(x):
    """
    Binary entropy function: h(x) = -x*log(x) - (1-x)*log(1-x)
    Returns 0 for values at the boundaries to avoid log(0).
    """
    eps = 1e-15  # Good threshold: avoids log(0) while maintaining precision
    if (x < - eps or x > 1 + eps):
        print(f"Warning: h(x) called with x={x}, which is outside [0,1]")
        return np.nan  # Return NaN for invalid inputs
    elif (x < eps or x > 1 - eps):
        return 0.  # Avoid log(0) by returning 0 at boundaries
    else:
        return -x * np.log(x) - (1 - x) * np.log(1 - x)
    
def H(v):
    """
    "Binary" Shannon entropy function: H(v) = sum of h(v_i) for each element v_i in vector v.
    """
    total = 0.0
    for vi in v:
        total += h(vi)
    return total

def nK(n, lamb):
    """
    Returns best block spectrum relative to n, according to Kaustav's conjecture, with each block spectrum already ordered
    """
    d = len(n)
    if (lamb < d/2 or lamb >= d):
        print(f"Warning: nK called with lambda={lamb}, which is outside [len(n)/2,len(n) - 1]")
        return torch.nan  # Return NaN for invalid inputs
    n_sorted = torch.sort(n, descending=True).values
    nK_2 = []
    for i in range(d - lamb):
        nK_2.append((n_sorted[i] + n_sorted[-i-1])/2)
    nK_2 = sorted(nK_2, reverse=True)
    nK_1 = sorted(nK_2 + list(n_sorted[d - lamb:lamb]), reverse=True)
    return torch.tensor(nK_1 + nK_2, dtype=n.dtype)

def block_spec(M, lamb):
    """
    Returns block spectrum of  eigenvalues of M. lamb is the partition parameter. 
    
    Parameters:
    -----------
    M : torch.Tensor (complex)
        Square matrix of size d
    lamb : int
        Partition parameter, in [d/2, d - 1 ]
        
    Returns:
    --------
    b : torch.Tensor (complex)
        Block spectrum.
        [b[0], ..., b[lamb - 1]] is the (unordered) spectrum of upper-left lamb x lambd block of M
        [b[lamb], ..., b[d - 1]] is the (unordered) spectrum of lower-right block of M
    """
    d = M.shape[0]
    if (lamb < d/2 or lamb >= d):
        print(f"Warning: nK called with lambda={lamb}, which is outside [len(n)/2,len(n) - 1]")
        return np.nan  # Return NaN for invalid inputs
    B_1 = M[0:lamb,0:lamb]
    B_2 = M[lamb:d,lamb:d]
    return torch.cat((torch.linalg.eigvals(B_1), torch.linalg.eigvals(B_2)))

# function to pass from M to the corresponding hermitian matrix H
def M_to_A(M):
    """
    Returns a d x d antihermitian matrix A starting from a d x d real matrix M
    """
    d = M.shape[0]
    A = torch.zeros((d, d), dtype=torch.cdouble) # this tensor will have complex entries
    for i in range(d):
        for k in range(d):
            if i == k:
                A[i][k] = M[i][k] * 1.j
            if i < k:
                A[i][k] = M[i][k] + 1.j * M[k][i]
            if i > k:
                A[i][k] = - M[k][i] + 1.j * M[i][k]
    return A


# def check_convergence(residual_norm: float, tol: float, iteration: int, max_iter: int) -> bool:
    """
    Check if the algorithm has converged.
    
    Parameters:
    -----------
    residual_norm : float
        ......
        
    Returns:
    --------
    bool
        True if converged, False otherwise
    """
#    return residual_norm < tol or iteration >= max_iter

    