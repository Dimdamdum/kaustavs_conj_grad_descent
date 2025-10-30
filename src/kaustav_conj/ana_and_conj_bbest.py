"""
Functions for bbest vectors
This module contains two functions: b_best_ana returning analytical values of bbest, and b_best_conj returning conjectured values of bbest. Their implementation goes on as we find out more and more formulas for bbest (either analytically or as conjectures, based on numerical sampling).
"""
import numpy as np

def b_best_ana(n, partition):
    """"
    Returns the analytically obtained b_best relative to the d-dimensional vector n and the given partition.

    Partitions supported so far:
    d=4 [3,1], [2,2], [2,1,1], [1,1,1,1]
    d=5 [3,1,1]
    """
    # quick validation
    if len(n) != sum(partition):
        raise ValueError("Input vector n must have length 4.")
    if not all(n[i] >= n[i+1] for i in range(3)):
        raise ValueError("Input vector n must be decreasingly ordered: n0 >= n1 >= n2 >= n3.")
    
    ############
    ##  d = 4 ##
    ############

    #### partition (3,1) ####
    if partition == [3,1]:
        appo = np.array(sorted([(n[0] + n[3]) / 2, n[1], n[2]], reverse=True))
        b_best = np.concatenate((appo, np.array([(n[0] + n[3]) / 2])))

    #### partition (2,2) ####
    elif partition == [2,2]:
        appo = np.array(sorted([(n[0] + n[3]) / 2, (n[1] + n[2]) / 2], reverse=True))
        b_best = np.concatenate((appo, appo))

    #### partition (2,1,1) ####
    elif partition == [2,1,1]:
        eta = sum(n)/4
        if eta >= n[1]: # 1st regime
            appo = (n[0]+n[2]+n[3])/3
            b_best = np.array([appo, n[1], appo, appo])
        elif eta >= n[2]: # 2nd regime
            b_best = np.array([eta,eta,eta,eta])
        else: # 3rd regime
            appo = (n[0]+n[1]+n[3])/3
            b_best = np.array([n[2], appo, appo, appo])

    #### partition (1,1,1,1) ####
    elif partition == [1,1,1,1]:
        eta = sum(n)/4
        b_best = np.array([eta,eta,eta,eta])

    ############
    ##  d = 5 ##
    ############

    #### partition (3,1,1) ####
    elif partition == [3,1,1]:
        n_reduced = np.array([n[0], n[1], n[3], n[4]])
        b_best_red = b_best_ana(n_reduced, [2,1,1])
        appo = np.array(sorted([b_best_red[0], b_best_red[1], n[2]], reverse=True))
        b_best = np.concatenate([appo, np.array([b_best_red[2]]), np.array([b_best_red[3]])])

    else:
        raise ValueError("Sorry, for the partition you chose the function has not been implemented yet!")

    return b_best


def b_best_CONJ(n, partition):
    """"
    Returns the analytically obtained b_best relative to the d-dimensional vector n and the given partition.

    Partitions supported so far:

    """
    # quick validation
    if len(n) != sum(partition):
        raise ValueError("Input vector n must have length 4.")
    if not all(n[i] >= n[i+1] for i in range(3)):
        raise ValueError("Input vector n must be decreasingly ordered: n0 >= n1 >= n2 >= n3.")
    
    #TODO
    b_best = np.array([1.])

    #if partition == [1]:
    #    return
    #else:
    raise ValueError("Sorry, for the partition you chose the function has not been implemented yet!")

    return b_best