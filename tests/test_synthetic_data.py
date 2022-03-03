import numpy as np
import pytest
# from sympy import symbols

from synthetic_data.synthetic_data import (pre_data_generation_checks,
                                           resolve_covariant)


def test_resolve_covariant_provided_no_covariant():
    covariant = resolve_covariant(n_total=2, covariant=None)

    assert covariant[0][0] == 1.0
    assert covariant[1][1] == 1.0


def test_resolve_covariant_provied_a_covariant():
    cov = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    covariant = resolve_covariant(n_total=3, covariant=cov)

    assert covariant[0][0] == 1
    assert covariant[1][1] == 5
    assert covariant[2][2] == 9


def test_pre_data_generation_checks():
    # x1, x2 = symbols("x1 x2")
    col_map = {"x1": 0, "x2": 1}

    # This assertion checks that nothing is returned by the function, meaning no exceptions
    # were thrown and all asserts passed
    assert not pre_data_generation_checks(n_informative=2, n_total=1, col_map=col_map)


def test_pre_data_generation_checks_0_n_total():
    # x1, x2 = symbols("x1 x2")
    col_map = {"x1": 0, "x2": 1}

    with pytest.raises(Exception) as exec_info:
        pre_data_generation_checks(n_informative=2, n_total=-1, col_map=col_map)
    err = 'total number of samples (n_informative + n_nuisance) must be greater than 0'
    assert err in str(exec_info.value)


def test_pre_data_generation_checks_col_and_n_informative_mismatch():
    # x1, x2 = symbols("x1 x2")
    col_map = {"x1": 0, "x2": 1}

    with pytest.raises(Exception) as exec_info:
        pre_data_generation_checks(n_informative=3, n_total=3, col_map=col_map)
    err = 'number of dictionary keys in col_map not equal to n_informative.'
    assert err in str(exec_info.value)
