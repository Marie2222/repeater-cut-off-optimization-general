import numpy as np

from repeater_algorithm import RepeaterChainSimulation

import pytest
import numpy as np

def test_distillation():
    # Initialize the repeater algorithm instance
    repeater = RepeaterChainSimulation()
    
    # Sample input parameters
    lambdas = np.array([0.85, 0.05, 0.05, 0.05])
    p_gen = 0.01
    t_trunc = 4
    cutoff = 3
    cut_type = "memory_time"

    t_list = np.arange(1, t_trunc) # t_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pmf = p_gen * (1 - p_gen)**(t_list - 1) # pmf = [0.1, 0.09, 0.081, 0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969, 0.043046721, 0.0387420489]
    pmf = np.concatenate((np.array([0.]), pmf)) # pmf = [0.0, 0.1, 0.09, 0.081, 0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969, 0.043046721, 0.0387420489]
    pmf1 = np.tile(pmf[:, np.newaxis], 4) # pmf = [[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1], [0.09, 0.09, 0.09, 0.09], [0.081, 0.081, 0.081, 0.081], [0.0729, 0.0729, 0.0729, 0.0729], [0.06561, 0.06561, 0.06561, 0.06561], [0.059049, 0.059049, 0.059049, 0.059049], [0.0531441, 0.0531441, 0.0531441, 0.0531441], [0.04782969, 0.04782969, 0.04782969, 0.04782969], [0.043046721, 0.043046721, 0.043046721, 0.043046721], [0.0387420489, 0.0387420489, 0.0387420489, 0.0387420489]]
    pmf2 = np.tile(pmf[:, np.newaxis], 4)
    lambda_func1 = np.array([lambdas] * t_trunc)
    lambda_func2 = np.array([lambdas] * t_trunc)

    # Run distillation
    pmf_dist, state_out = repeater.distillation(
        pmf1, lambda_func1, pmf2, lambda_func2, 
        cutoff, cut_type, depolar_rate=0.1, dephase_rate=0.05
    )
    
    # Check output shapes
    assert pmf_dist.shape == pmf1.shape, "PMF output shape mismatch"
    assert state_out.shape == lambda_func1.shape, "State output shape mismatch"

    print("Distillation test passed.")
