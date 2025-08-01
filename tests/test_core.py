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

from kaustav_conj.utils import h, H, M_to_A
from kaustav_conj.core import build_cost_function

class TestCore:
    """Test core functions."""

    def test_cost_function_builder(self):
        """Test implementation of build_cost_function"""
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