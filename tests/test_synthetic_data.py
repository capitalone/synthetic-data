import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from synthetic_data.synthetic_data import (
    make_tabular_data,
    pre_data_generation_checks,
    resolve_covariant,
)


def test_resolve_covariant_provided_no_covariant():
    covariant = resolve_covariant(n_total=2, covariant=None)

    assert covariant[0][0] == 1.0
    assert covariant[1][1] == 1.0


def test_resolve_covariant_provied_a_covariant():
    cov = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(Exception) as exec_info:
        covariant = resolve_covariant(n_total=3, covariant=cov)

    err = "Assertion error - please check covariance matrix is symmetric."
    assert err in str(exec_info.value)


def test_pre_data_generation_checks():
    col_map = {"x1": 0, "x2": 1}

    # This assertion checks that nothing is returned by the function, meaning no exceptions
    # were thrown and all asserts passed
    assert not pre_data_generation_checks(n_informative=2, n_total=1, col_map=col_map)


def test_pre_data_generation_checks_0_n_total():
    col_map = {"x1": 0, "x2": 1}

    with pytest.raises(Exception) as exec_info:
        pre_data_generation_checks(n_informative=2, n_total=-1, col_map=col_map)
    err = "total number of samples (n_informative + n_nuisance) must be greater than 0"
    assert err in str(exec_info.value)


def test_pre_data_generation_checks_col_and_n_informative_mismatch():
    col_map = {"x1": 0, "x2": 1}

    with pytest.raises(Exception) as exec_info:
        pre_data_generation_checks(n_informative=3, n_total=3, col_map=col_map)
    err = "number of dictionary keys in col_map not equal to n_informative."
    assert err in str(exec_info.value)


def test_data_scaling():
    col_map = {"x1": 0, "x2": 1, "x3": 2, "x4": 3}
    expr = "x1 + x2 + x3 + x4"

    dist = [{"dist": "norm", "column": col} for col in range(4)]

    x_final, _, _, _ = make_tabular_data(
        n_informative=4, expr=expr, col_map=col_map, scaler=StandardScaler(), dist=dist
    )
    assert np.all(np.isclose(x_final.mean(axis=0), np.zeros(4)))

    x_final, _, _, _ = make_tabular_data(
        n_informative=4,
        expr=expr,
        col_map=col_map,
        scaler=MinMaxScaler(feature_range=(0, 1)),
        dist=dist,
    )
    assert (x_final.max() == 1) and (x_final.min() == 0)

    x_final, _, _, _ = make_tabular_data(
        n_informative=4, expr=expr, col_map=col_map, scaler=None, dist=dist
    )

    with pytest.raises(Exception) as exec_info:
        x_final, _, _, _ = make_tabular_data(
            n_informative=4, expr=expr, col_map=col_map, scaler=lambda x: x, dist=dist
        )
    err = "Please provide a valid sklearn scaler."
    assert err in str(exec_info.value)


def test_marginal_dist_check():
    col_map = {"x1": 0, "x2": 1, "x3": 2, "x4": 3}

    # should skip dist enforcement
    data = make_tabular_data(n_informative=4, col_map=col_map)

    with pytest.raises(Exception) as exec_info:
        make_tabular_data(n_informative=4, col_map=col_map, dist=[{}])
    err = (
        "When providing a marginal distribution list, ensure the length of the "
        "list is equal to n_informative columns."
    )
    assert err in str(exec_info.value)
