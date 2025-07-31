"""
Utility functions

This module contains helper functions for matrix operations, convergence checking,... TODO
"""

import numpy as np
from numpy import linalg as LA

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
        return np.nan  # Return NaN for invalid inputs
    n_sorted = np.sort(n)[::-1]
    nK_2 = []
    for i in range(d - lamb):
        nK_2.append((n_sorted[i] + n_sorted[-i-1])/2)
    nK_2.sort(reverse=True)
    nK_1 = sorted(nK_2 + list(n_sorted[d - lamb:lamb]), reverse=True)
    return np.array(nK_1 + nK_2)

def block_spec(M, lamb):
    """
    Returns block spectrum of  eigenvalues of M. lamb is the partition parameter. 
    
    Parameters:
    -----------
    M : np.ndarray (complex)
        Square matrix of size d
    lamb : int
        Partition parameter, in [d/2, d - 1 ]
        
    Returns:
    --------
    b : np.array (complex)
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
    return np.concatenate((LA.eigvals(B_1), LA.eigvals(B_2))) 


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

    