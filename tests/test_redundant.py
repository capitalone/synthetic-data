#!/usr/bin/env python
"""
Test the creation of redundant features (see sklearn source)

The structure of X is columns of [informative, redundant, nuisance] features
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from synthetic_data.synthetic_data import (
    generate_redundant_features,
    make_tabular_data,
    transform_to_distribution,
)

np.random.seed(111)
np.set_printoptions(precision=11)
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
    )
    print("in test results for X - ")
    print(X)

    # replicate the redundant features
    # replicate the random state - initialize, run multivariate...
    generator = np.random.RandomState(seed)
    means = np.zeros(n_informative)
    mvnorm = stats.multivariate_normal(mean=means, cov=cov)
    x = mvnorm.rvs(n_samples, random_state=seed)
    norm = stats.norm()
    x_cont = norm.cdf(x)

    # this duplicates the generate_redundant_features function
    B = 2 * generator.rand(n_informative, n_redundant) - 1
    print("in test - B")
    print(B)
    # x_cont = X[:, :n_informative]
    print("in test - x")
    print(x_cont)

    x_redundant = np.dot(x_cont, B)

    scaler = MinMaxScaler(feature_range=[-1, 1])
    x_redundant_scaled = scaler.fit_transform(x_redundant)
    print(" - scaled - ")
    print(x_redundant_scaled)

    x_slice_redundant = X[:, -n_redundant:]

    # print("in test script - x_redundant")
    # print(x_redundant)

    # check that they match
    assert np.allclose(
        x_redundant_scaled, x_slice_redundant, rtol=1e-05, atol=1e-08
    )
