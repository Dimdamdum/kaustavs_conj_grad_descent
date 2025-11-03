"""
Utility functions

This module contains helper functions for matrix operations, basic entropic functions, eigenvalue computations.
"""

import numpy as np
from numpy import linalg as LA
import torch
from sympy.utilities.iterables import partitions


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
    elif isinstance(x, torch.Tensor):
        return -x * torch.log(x) - (1 - x) * torch.log(1 - x) # needed when using torch gradients
    else:
        return -x * np.log(x) - (1 - x) * np.log(1 - x) # needed because torch.log does not work on floats
    
def H(v):
    """
    "Binary" Shannon entropy function: H(v) = sum of h(v_i) for each element v_i in vector v.
    """
    total = 0.0
    for vi in v:
        total += h(vi)
    return total

def bK(n, lamb):
    """
    Returns best block spectrum relative to n, according to Kaustav's conjecture, with each block spectrum already ordered.
    Accepts n as a list of floats (or numpy array), returns numpy array.
    """
    d = len(n)
    if (lamb < d/2 or lamb >= d):
        print(f"Warning: bK called with lambda={lamb}, which is outside [len(n)/2,len(n) - 1]")
        return float('nan')  # Return NaN for invalid inputs
    n_sorted = sorted(n, reverse=True)
    bK_2 = []
    for i in range(d - lamb):
        bK_2.append((n_sorted[i] + n_sorted[-i-1]) / 2)
    bK_2 = sorted(bK_2, reverse=True)
    bK_1 = sorted(bK_2 + n_sorted[d - lamb:lamb], reverse=True)
    return np.array(bK_1 + bK_2)

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
        print(f"Warning: block_spec called with lambda={lamb}, which is outside [d/2,d - 1]")
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

def majorizes(x, y, eps=1e-14):
    """
    Returns True if sum(x) = sum(y) up to precision eps (default: 1e-14) and
    sum_{i=0}^k x_decreasingly_ordered_k[i] >= sum_{i=0}^k y_decreasingly_ordered_k[i] up to precision eps for k = 0, ..., d - 1
    with d = len(x) = len(y), False otherwise
    """
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        print(f"Possible error: majorizes called with vectors of different lengths (len(x)={len(x)}, len(y)={len(y)})")
        return False
    d = len(x)
    x_sorted = np.sort(x)[::-1]
    y_sorted = np.sort(y)[::-1]
    abs_sum_diff = abs(np.sum(x_sorted) - np.sum(y_sorted))
    if abs_sum_diff > eps:
        print(f"(Function majorizes:) sum of all entries of x and y differ by {abs_sum_diff}")
        return False
    for k in range(d):
        if np.sum(x_sorted[:k+1]) < np.sum(y_sorted[:k+1]) - eps:
            print(f"(Function majorizes:) ({k}-th partial sum of y) - ({k}-th partial sum of x ) = {np.sum(y_sorted[:k+1]) - np.sum(x_sorted[:k+1])}")
            return False
    return True
    
def multi_block_spec(M, int_partition, order=False):
    """
    Returns block spectrum of  eigenvalues of M. lamb is the integer partition. 
    
    Parameters:
    -----------
    M : torch.Tensor (complex)
        Square matrix of size d
    int_partition : list (integer)
        Integer partition of d
    order : boolean
        If True, block spectra will be ordered decreasingly (for each block).
        
    Returns:
    --------
    b : torch.Tensor (complex)
        Block spectrum.
        [b[0], ..., b[int_partition[0] - 1]] is the (un)ordered spectrum of upper-left int_partition[0] x int_partition[0] block of M
        [b[int_partition[0]], ..., b[int_partition[0] + int_partition[1] - 1]] is the (un)ordered spectrum of second block, and so on
    """
    d = M.shape[0]
    int_partition_length = len(int_partition)
    if not (d == sum(int_partition)) and all(int_partition[i] >= int_partition[i+1] for i in range(int_partition_length - 1)):
        print(f"Warning: multi_block_spec called with int_partition={int_partition}, which is not a valid decreasingly ordered integer partition of d={d}")
        return np.nan  # Return NaN for invalid inputs
    blocks = []
    lower_cut = 0
    for i in range(int_partition_length):
        upper_cut = lower_cut + int_partition[i]
        blocks.append(M[lower_cut:upper_cut, lower_cut:upper_cut])
        lower_cut = upper_cut
    if order:
        b = torch.cat([torch.sort(torch.real(torch.linalg.eigvals(B)), descending=True).values for B in blocks])
    else:
        b = torch.cat([torch.linalg.eigvals(B) for B in blocks])
    return b

def list_all_integer_partitions(d):
    """
    Returns a list of sublists. The sublists are the integer partitions (Youg diagrams) of the integer d. The trivial integer partition [d] is omitted.
    """

    # Generate all integer partitions
    all_partitions = list(partitions(d))

    # Convert to list format
    list_partitions = []
    for p in all_partitions:
        partition = []
        for k, v in p.items():
            partition.extend([k] * v)
        list_partitions.append(partition)

    return list_partitions[1:] # remove trivial partition

import numpy as np

def avg(set_partition, n):
    """
    Compute a vector obtained by mixing/averaging the components of n according to the set partition P of the set {0,...,len(n) - 1}.

    Parameters
    ----------
    set_partition : list of lists of int
        Ordered set partition of {0, ..., d-1}, e.g. [[0,1,3], [2]]
    n : list
        d-dimensional vector of real numbers

    Returns
    -------
    np.ndarray
        Vector in R^d whose components are blockwise averages of n.
    """
    n = np.asarray(n, dtype=float)
    d = len(n)
    out = np.zeros(d, dtype=float)

    # Validate that set_partition unrolled is a permutation of 0..d-1
    flat_partition = [item for sublist in set_partition for item in sublist]
    if sorted(flat_partition) != list(range(d)):
        raise ValueError(f"avg: partition P must be a permutation of 0..{d-1}; unrolled P = {sorted(flat_partition)}")

    for block in set_partition:
        avg_val = np.mean(n[block])
        out[block] = avg_val

    return out

def inter_block_sort(int_partition, b):
    """
    Sorts the blocks of the array b as indicated by int_partition in decreasing order.

    Parameters:
    -----------
    int_partition : list of int
        A partition of the length d of b, indicating the sizes of the blocks.
    b : numpy.ndarray
        A d-dimensional numpy array to be permuted.

    Returns:
    --------
    numpy.ndarray
        A permutation of b such that the blocks of b are sorted in decreasing order.

    Example:
    --------
    d = 3
    int_partition = [2, 1]
    b = np.array([2., 4., 5.])
    Output: np.array([4., 2., 5.])
    """
    if len(b) != sum(int_partition):
        raise ValueError(f"inter_block_sort: Length of b ({len(b)}) must equal the sum of int_partition ({sum(int_partition)}).")
    
    sorted_b = []
    start_idx = 0

    for block_size in int_partition:
        # Extract the block
        block = b[start_idx:start_idx + block_size]
        # Sort the block in decreasing order and append to the result
        sorted_b.extend(sorted(block, reverse=True))
        # Update the starting index for the next block
        start_idx += block_size

    return np.array(sorted_b)

