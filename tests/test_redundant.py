#!/usr/bin/env python
"""
Test the creation of redundant features (see sklearn source)

The structure of X is columns of [informative, redundant, nuisance] features

If the test fails, and you want to debug, two things must match between the
    main script and this test: the matrix (B/b) and the unscaled inputs to
    the redundant routine, here named x_cont (in main script routine it's x)
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from synthetic_data.synthetic_data import (
    make_tabular_data,
    transform_to_distribution,
    generate_redundant_features,
)

np.random.seed(111)
np.set_printoptions(precision=6)
seed = 1234


def test_redundant():
    # define expression
    expr = "x1"

    # define mapping from symbols to column of X
    col_map = {"x1": 0, "x2": 1}

    # baseline 2D data, no noise
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    # generate synthetic data with 2 redundant columns
    seed = 1234
    generator = np.random.RandomState(seed)

    n_samples = 3
    n_informative = 2
    n_redundant = 2

    dist = [{"dist": "norm", "column": col} for col in range(2)]

    X, _, _, _ = make_tabular_data(
        n_samples=n_samples,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_nuisance=0,
        cov=cov,
        col_map=col_map,
        expr=expr,
        p_thresh=0.5,
        # random_state=generator,
        seed=seed,
        dist=dist,
    )

    means = np.zeros(n_informative)
    mvnorm = stats.multivariate_normal(mean=means, cov=cov, allow_singular=True)
    x = mvnorm.rvs(n_samples, random_state=seed)

    x_cont = np.zeros_like(x)
    for i in range(x.shape[1]):
          x_tmp = x[:, i]
          tmp_norm = stats.norm(loc=x_tmp.mean(), scale=x_tmp.std())
          x_cont[:, i] = tmp_norm.cdf(x_tmp)


    for a_dist in dist:
        col = a_dist["column"]
        x_cont[:, col] = transform_to_distribution(x_cont[:, col], a_dist)

    # this duplicates the generate_redundant_features function
    generator = np.random.RandomState(seed)
    B = 2 * generator.rand(n_informative, n_redundant) - 1
#    print("in test - B")
#    print(B)

    # x_cont = X[:, :n_informative]
    print("in test - x_cont")
    print(x_cont)

    x_redundant = np.dot(x_cont, B)

    # now loop over the redundant columns and use MinMax scaler...
    for col in range(n_redundant):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_redundant[:, col] = (
            scaler.fit_transform(x_redundant[:, col].reshape(-1, 1)).flatten()
        )

    x_slice_redundant = X[:, -n_redundant:]
#    print("in test script - x_slice_redundant")
#    print(x_slice_redundant)

#    print("in test script - x_redundant")
#    print(x_redundant)

    # check that they match
    assert np.allclose(x_redundant, x_slice_redundant, rtol=1e-05, atol=1e-08)
