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

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kaustav_conj.utils import h, H, nK, block_spec


class TestUtils:
    """Test utility functions."""
    
    def test_h_function(self):
        """Test basic properties of function h(x)."""
        assert np.isclose(h(0.5), np.log(2), rtol=1e-10)  # h(0.5) = log(2)
        x = 0.123511
        assert np.isclose(h(x), h(1-x), rtol=1e-10) # h(x) = h(1 - x)
        assert h(0.0) == 0.0
        assert h(1 - 1e-16) == 0.0
        assert np.isnan(h(-0.1))
    
    def test_H_function(self):
        """Test function H(v) is sum of h values."""
        # Test with mixed values
        v_mixed = [0.1, 0.5, 0.9]
        expected = h(0.1) + h(0.5) + h(0.9)
        assert np.isclose(H(v_mixed), expected, rtol=1e-10)

    def test_nK_function(self):
        """Test basic properties of function nK(n)."""
        n = np.array([1., 6., 2., 4., 3.3])
        lamb = 3
        nK_correct = np.array([3.5, 3.3, 3., 3.5, 3.])
        print(nK(n, lamb))
        assert np.allclose(nK(n, lamb), nK_correct, rtol=1e-10)
        assert np.isnan(nK(n, 2))

    def test_block_spec_function(self):
        """Test we're actually getting spectrum of diagonal blocks."""
        Y = np.array([[0, -1.j], [1.j, 0]])
        D1 = np.diag([[3.14 + 1.j]])
        D2 = np.diag((4. + 0.j,5. + 0.j))
        M = block_diag(Y, D1, D2)
        lamb = 3
        spec_1 = [-1. + 0.j, 1. + 0.j, 3.14 + 1.j]
        spec_2 = [4. + 0.j, 5. + 0.j]
        b = block_spec(M, lamb)
        b_1 = list(b[0:lamb])
        b_2 = list(b[lamb:5])
        # Check spec_1 and b_1 are same lists up to permutation
        assert np.allclose(
            np.sort(np.array(spec_1)), 
            np.sort(np.array(b_1)), 
            rtol=1e-10
        )
        # Check spec_2 and b_2 are same lists up to permutation
        assert np.allclose(
            np.sort(np.array(spec_2)), 
            np.sort(np.array(b_2)), 
            rtol=1e-10
        )




#    def test_check_convergence(self):
#        """Test convergence checking."""
#        assert check_convergence(1e-7, 1e-6, 10, 100) == True  # Tolerance met
#        assert check_convergence(1e-5, 1e-6, 10, 100) == False  # Tolerance not met
#        assert check_convergence(1e-7, 1e-6, 101, 100) == True  # Max iterations reached
    
#    def test_compute_residual(self):
#        """Test residual computation."""
#        # TODO
