#!/usr/bin/env python
"""
1. Generate s.d  with no noise
2. Generate s.d. with noise
3. Calculate the delta between the two data sets
4. Max/min value should not exceed specified noise level

Assumptions made:
    - generating gaussian white noise at specified noise level
    - assuming MinMax scaler on all features  - so noise level
    can be interpreted as a percentage. If feature scaling is not MinMax,
    it's not a percentage...it's just an applied delta...

What could make this test fail?
    -
"""

import numpy as np

from synthetic_data.synthetic_data import generate_x_noise, make_tabular_data

np.random.seed(111)
np.set_printoptions(precision=11)
seed = 1234


def test_noise():
    # define expression
    expr = "x1"

    # define mapping from symbols to column of X
    col_map = {"x1": 0, "x2": 1}

    # baseline 2D data, no noise
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    # dummy dist arg to bypass dist requirement
    dist = [{"dist": "norm", "column": col} for col in range(2)]

    n_samples = 3
    X, _, _, _ = make_tabular_data(
        n_samples=n_samples,
        cov=cov,
        col_map=col_map,
        expr=expr,
        p_thresh=0.5,
        seed=seed,
        dist=dist,
    )

    # the noise matrix
    noise_level_x = 0.08
    x_noise = generate_x_noise(X, noise_level_x, seed=seed)

    # 2D data with noise
    X_noise, _, _, _ = make_tabular_data(
        n_samples=n_samples,
        cov=cov,
        col_map=col_map,
        expr=expr,
        p_thresh=0.5,
        noise_level_x=noise_level_x,
        seed=seed,
        dist=dist,
    )
    # delta from noise to no noise
    delta = X_noise - X

    print("delta = ")
    print(delta)

    print("x_nose = ")
    print(x_noise)

    assert np.allclose(delta, x_noise, rtol=1e-05, atol=1e-08)
