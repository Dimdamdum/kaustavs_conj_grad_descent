"""
Functions for bbest vectors
This module contains two functions: b_best_ana returning analytical values of bbest, and b_best_conj returning conjectured values of bbest. Their implementation goes on as we find out more and more formulas for bbest (either analytically or as conjectures, based on numerical sampling).
"""
import numpy as np
from kaustav_conj.utils import avg, inter_block_sort

def b_best_ana(n, int_partition):
    """"
    Returns the analytically obtained b_best relative to the d-dimensional vector n and the given integer partition. The key underlying idea is a function mapping int_partition of d to a set_partition of {1, ..., d}, possibly depending on the input n ("regimes").

    Integer partitions supported so far:
    d=4 [3,1], [2,2], [2,1,1], [1,1,1,1]
    d=5 [3,1,1]
    """
    # quick validation
    if len(n) != sum(int_partition):
        raise ValueError("Input vector n must have length 4.")
    if not all(n[i] >= n[i+1] for i in range(3)):
        raise ValueError("Input vector n must be decreasingly ordered: n0 >= n1 >= n2 >= n3.")
    
    already_averaged = False # for some cases, we'll need some permutation of the entries of b_best. In those cases, we'll apply avg directly rather than at the end of the function.

    ############
    ##  d = 4 ##
    ############

    if int_partition == [3,1]:
        best_set_partition = [[1,4], [2],[3]]
    elif int_partition == [2,2]:
        best_set_partition = [[1,4], [2,3]]
    elif int_partition == [2,1,1]:
        eta = sum(n)/4
        if eta >= n[1]: # 1st regime
            best_set_partition = [[1,3,4],[2]]
        elif eta >= n[2]: # 2nd regime
            best_set_partition = [[1,2,3,4]]
        else: # 3rd regime
            best_set_partition = [[1,2,4],[3]]
    elif int_partition == [1,1,1,1]:
        best_set_partition = [[1,2,3,4]]

    ############
    ##  d = 5 ##
    ############

    elif int_partition == [3,1,1]:
        eta = (sum(n) - n[2])/4
        if eta >= n[1]: # 1st regime
            best_set_partition = [[1,4,5],[2],[3]]
        elif eta >= n[3]: # 2nd regime
            best_set_partition = [[1,2,4,5],[3]]
        else: # 3rd regime
            best_set_partition = [[1,2,5],[3],[4]] # !!! we can't keep the average of n[0],n[1],n[4] in positions 0,1,4 of b_best! We need one more permutation here.
            b_best = avg(best_set_partition, n)
            b_best[1], b_best[3] = b_best[3], b_best[1]
            already_averaged = True

    else:
        raise ValueError("Sorry, for the integer partition you chose the function has not been implemented yet!")
    
    if already_averaged == False:
        b_best = avg(best_set_partition, n)
    b_best = inter_block_sort(int_partition, b_best)

    return b_best


def b_best_conj(n, int_partition):
    """"
    Returns the analytically obtained b_best relative to the d-dimensional vector n and the given integer partition.

    Partitions supported so far:

    """
    # quick validation
    if len(n) != sum(int_partition):
        raise ValueError("Input vector n must have length 4.")
    if not all(n[i] >= n[i+1] for i in range(3)):
        raise ValueError("Input vector n must be decreasingly ordered: n0 >= n1 >= n2 >= n3.")
    
    #TODO
    b_best = np.array([1.])

    #if int_partition == [1]:
    #    return
    #else:
    raise ValueError("Sorry, for the integer partition you chose the function has not been implemented yet!")

    return b_best

def check_conjecture(n, int_partition, b_best_num, tolerance):
    """"
    Checks whether norm(b_best_num - b_best_conj(n, int_partition)) is below tolerance. Returns boolean.

    """
    if np.linalg.norm(b_best_conj(n, int_partition) - np.array(b_best_num)) < tolerance:
        return True
    else:
        return False