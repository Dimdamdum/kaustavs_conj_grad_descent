"""
Utility functions

This module contains helper functions for matrix operations, convergence checking,... TODO
"""

import numpy as np
# from ... import ...

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

    