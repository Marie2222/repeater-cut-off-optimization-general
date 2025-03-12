import pytest
import numpy as np
from protocol_units import get_dist_lambda_out

def test_get_dist_lambda_out():
    # Sample test input
    t1, t2 = 3, 5  # Example time values
    lambdas1 = np.array([0.85, 0.05, 0.05, 0.05])
    lambdas2 = np.array([0.85, 0.05, 0.05, 0.05])
    depolar_rate = 0.00
    dephase_rate = 0.00
    amplitude_damping_rate = 0.0
    bit_phase_flip_rate = 0.0
    
    # Call function
    output_lambdas = get_dist_lambda_out(
        t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate, amplitude_damping_rate, bit_phase_flip_rate
    )
    print(output_lambdas)
    # Check output validity
    assert len(output_lambdas) == 4, "Output lambda array must have length 4"
    assert np.isclose(np.sum(output_lambdas), 1, atol=1e-2), "Sum of lambdas should be close to 1"
    assert not np.isnan(output_lambdas).any(), "Output lambdas should not contain NaN values"
    
    print("get_dist_lambda_out test passed.")
