"""
Tests for utility functions
"""

import numpy as np
import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kaustav_conj.utils import h


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

#    def test_check_convergence(self):
#        """Test convergence checking."""
#        assert check_convergence(1e-7, 1e-6, 10, 100) == True  # Tolerance met
#        assert check_convergence(1e-5, 1e-6, 10, 100) == False  # Tolerance not met
#        assert check_convergence(1e-7, 1e-6, 101, 100) == True  # Max iterations reached
    
#    def test_compute_residual(self):
#        """Test residual computation."""
#        # TODO
