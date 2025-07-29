"""
Tests for utility functions
"""

import numpy as np
import pytest
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# from kaustav_conj.core import check_convergence, ....


class TestUtils:
    """Test utility functions."""
    
    def test_check_convergence(self):
        """Test convergence checking."""
        #TODO
        # assert check_convergence(1e-7, 1e-6, 10, 100) == True
        # assert check_convergence(1e-5, 1e-6, 10, 100) == False
        # assert check_convergence(1e-7, 1e-6, 101, 100) == True
    
    def test_compute_residual(self):
        """Test residual computation."""
        # TODO