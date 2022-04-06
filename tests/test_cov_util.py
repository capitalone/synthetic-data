#!/usr/bin/env python

import numpy as np
from scipy import stats

from synthetic_data.utils import tuples_to_cov

np.random.seed(seed=3)


def test_cov_util():
    """ Test covariance utility """
    col_map = {"x1": 0, "x2": 1, "x3": 2}

    cov_list = [("x1", "x2", -1), ("x1", "x3", 0.1)]
    cov = tuples_to_cov(cov_list, col_map)

    cov_true = np.array([[1, -1, 0.1], [-1, 1, 0], [0.1, 0, 1]])

    print(cov_true)
    print(cov)
    assert np.array_equal(cov, cov_true)
