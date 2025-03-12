import pytest
import numpy as np
from repeater_algorithm import RepeaterChainSimulation


def test_iterative_convolution():
    repeater = RepeaterChainSimulation()
    t_trunc = 5
    shift = 
    lambdas = np.array([0.85, 0.05, 0.05, 0.05])
    first_func = np.array([0. ,0. ,0., 0.] * t_trunc)
    func = np.array([lambdas] * t_trunc)
    result = repeater.iterative_convolution(first_func, func)
    print(result)
    assert result.shape == (5, 4), "Output shape mismatch"
    assert np.isclose(np.sum(result), 1, atol=1e-2), "Sum of lambdas should be close to 1"
    assert not np.isnan(result).any(), "Output lambdas should not contain NaN values"
    print("iterative_convolution test passed.")