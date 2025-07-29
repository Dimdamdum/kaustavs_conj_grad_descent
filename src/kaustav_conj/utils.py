"""
Utility functions

This module contains helper functions for matrix operations, convergence checking,... TODO
"""

import numpy as np
# from ... import ...


def check_convergence(residual_norm: float, tol: float, iteration: int, max_iter: int) -> bool:
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
    return residual_norm < tol or iteration >= max_iter