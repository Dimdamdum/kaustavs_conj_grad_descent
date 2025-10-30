"""
Tests for core algorithms
"""

import numpy as np
import pytest
import sys
import os
import torch

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kaustav_conj.utils import h, H, M_to_A, bK
from kaustav_conj.core import build_cost_function, get_b_best, multi_build_cost_function, multi_get_b_best

class TestCore:
    """Test core functions."""

    def test_cost_function_builders(self):
        """Test implementation of build_cost_function and multi_build_cost_function"""
        n = [0.2, 0.6]
        lamb = 1
        cost_function = build_cost_function(n, lamb)
        M = torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.double) * np.pi/2    # U = exp(M_to_A(M)) is sigma_x
        H_correct = -H(n) # conjugating a matrix with sigma_x simply permutes its diagonal elements
        assert np.isclose(H_correct, cost_function(M), rtol=1e-10)
        assert np.isnan(build_cost_function(n, 2))
        n = [0.2, 1.001]
        assert np.isnan(build_cost_function(n, lamb))
        assert np.isnan(cost_function(np.zeros((2,2))))
        assert np.isnan(cost_function(torch.tensor([[0.]], dtype=torch.double)))
        # test more general cost function
        n = [0.2, 0.6]
        partition = [1,1]
        multi_cost_function = multi_build_cost_function(n, partition)
        assert np.isclose(H_correct, multi_cost_function(M), rtol=1e-10)


    def test_get_b_best(self):
        """Test whether get_b_best actually gives optimal spectrum"""
        n = [0.2, 0.6]
        lamb = 1
        b_best_conj = bK(n, lamb)
        eps = 1e-14
        U_best, b_best_num, H_best, conjecture_holds = get_b_best(n, lamb, N_init=4, N_steps=300,learning_rate=0.01, eps = eps)
        assert np.allclose(b_best_conj, b_best_num, rtol=1e-5)
        assert conjecture_holds

    def test_multi_get_b_best(self):
        """Test whether multi_get_b_best actually gives optimal spectrum"""
        n = [0.9, 0.3, 0.2, 0.1] # average of n_i's is between n_1 and n_2
        partition = [2,1,1]
        b_best_true = np.array([.4, .3, .4, .4])
        eps = 1e-14
        U_best, b_best_num, H_best = multi_get_b_best(n, partition, N_init=4, N_steps=300,learning_rate=0.01)
        b_best_num = b_best_num.detach().numpy()
        assert np.allclose(b_best_true, b_best_num, rtol=1e-3)