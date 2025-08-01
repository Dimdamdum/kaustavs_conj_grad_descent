"""
Tests for utility functions
"""

import numpy as np
from numpy import linalg as LA
import pytest
import sys
import os
import scipy
from scipy.linalg import block_diag
import torch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kaustav_conj.utils import h, H, nK, block_spec, M_to_A


class TestUtils:
    """Test utility functions."""
    
    def test_h_function(self):
        """Test basic properties of function h(x)."""
        x = 0.5
        assert np.isclose(h(x), np.log(2), rtol=1e-10)  # h(0.5) = log(2)
        x_torch = torch.tensor(x, dtype=torch.double)
        assert torch.isclose(h(x_torch), torch.tensor(np.log(2), dtype=torch.double), rtol=1e-10)
        x = 0.123511
        assert np.isclose(h(x), h(1-x), rtol=1e-10) # h(x) = h(1 - x)
        assert h(0.0) == 0.0
        assert h(1 - 1e-16) == 0.0
        assert np.isnan(h(-0.1))
    
    def test_H_function(self):
        """Test function H(v) is sum of h values."""
        v = [0.1, 0.5, 0.9]
        v_torch = torch.tensor(v, dtype=torch.double)
        expected = h(0.1) + h(0.5) + h(0.9)
        assert np.isclose(H(v), expected, rtol=1e-10)
        assert torch.isclose(H(v_torch), torch.tensor(expected, dtype=torch.double), rtol=1e-10)

    def test_nK_function(self):
        """Test basic properties of function nK(n)."""
        n = [1., 6., 2., 4., 3.3]
        lamb = 3
        nK_correct = [3.5, 3.3, 3., 3.5, 3.]
        assert np.allclose(nK(n, lamb), nK_correct, rtol=1e-10)
        assert np.isnan(nK(n, 2))

    def test_block_spec_function(self):
        """Test we're actually getting spectrum of diagonal blocks (torch version)."""
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
        D1 = torch.diag(torch.tensor([3.14 + 1j], dtype=torch.cdouble))
        D2 = torch.diag(torch.tensor([4. + 0j, 5. + 0j], dtype=torch.cdouble))
        M = torch.block_diag(Y, D1, D2)
        lamb = 3
        spec_1 = torch.tensor([-1. + 0j, 1. + 0j, 3.14 + 1j], dtype=torch.cdouble)
        spec_2 = torch.tensor([4. + 0j, 5. + 0j], dtype=torch.cdouble)
        b = block_spec(M, lamb)
        b_1 = b[0:lamb]
        b_2 = b[lamb:5]
        # Convert tensors to numpy arrays for comparison and sorting (bc of complex entries)
        spec_1_sorted = np.sort(np.array(spec_1))
        b_1_sorted = np.sort(np.array(b_1))
        spec_2_sorted = np.sort(np.array(spec_2))
        b_2_sorted = np.sort(np.array(b_2))
        # Check spec_1 and b_1 are same arrays up to permutation
        assert np.allclose(spec_1_sorted, b_1_sorted, rtol=1e-10)
        # Check spec_2 and b_2 are same arrays up to permutation
        assert np.allclose(spec_2_sorted, b_2_sorted, rtol=1e-10)

    def test_M_to_A_function(self):
        """Test basics of M_to_A."""
        M = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.double)
        A = M_to_A(M)
        A_correct = torch.tensor([[1.j, 2. + 3.j], [-2. + 3.j,  4.j]], dtype=torch.cdouble)
        assert torch.allclose(A, A_correct, rtol=1e-10)
        M = torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.double)
        A = M_to_A(M)
        A_correct = torch.tensor([[1.j, -1.j], [-1.j,  1.j]], dtype=torch.cdouble)


#    def test_check_convergence(self):
#        """Test convergence checking."""
#        assert check_convergence(1e-7, 1e-6, 10, 100) == True  # Tolerance met
#        assert check_convergence(1e-5, 1e-6, 10, 100) == False  # Tolerance not met
#        assert check_convergence(1e-7, 1e-6, 101, 100) == True  # Max iterations reached
    
#    def test_compute_residual(self):
#        """Test residual computation."""
#        # TODO
