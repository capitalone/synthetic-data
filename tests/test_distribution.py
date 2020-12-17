#!/usr/bin/env python

import numpy as np
from scipy import stats

from synthetic_data.synthetic_data import transform_to_distribution

np.random.seed(seed=3)


def test_distribution():
    """ Test feature generation - statistics & shape"""
    mu = 0.0
    sigma = 0.1
    n_samples = 1000
    seed = 1234

    adict = {"col": [], "dist": "norm", "kwds": {"loc": mu, "scale": sigma}}

    x = stats.uniform(0, 1).rvs(n_samples, random_state=seed)
    x_test = transform_to_distribution(x, adict)
    print("shape - ", x_test.shape)
    print("mean - ", np.mean(x_test))
    print("std - ", np.std(x_test))
    print("diff on mean - ", mu - np.mean(x_test))

    assert mu - np.mean(x_test) < 0.01
    assert sigma - np.std(x_test, ddof=1) < 0.01
    assert x_test.shape[0] == n_samples
    assert x_test.shape == (n_samples,)
