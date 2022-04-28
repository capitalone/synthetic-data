#!/usr/bin/env python
"""
Generate synthetic data for a binary classification problem.

Inspired by sklearn.datasets.make_classification.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

Generate tabular data to provide ground truth data for post-hoc explainability of black box models.
With user specified control over:
    1. marginal distribution for features
    2. correlation structure
    3. nonlinearity & interactions (otherwise, why use advanced ML techniques?)
    4. Noise / overlap
    5. categorical features (stretch goal)
    6. outliers (stretch)

    TODO:
        - class_sep - spread out our class clusters
        - add noise to X & y (see e.g. flip_y)
        - add repeated? add redundant?
        - add weights for balanced/unbalanced
        - scale - set min/max?
        - shuffle
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from synthetic_data.parser import MathParser


def transform_to_distribution(x, adict):
    """
    Input:
        x - input uniform distributed random variable
        adict - dictionary corresponding to the desired distribution (name & params)
        e.g. {'col':[list], 'dist':'normal', 'kwargs':{'loc':0.0, 'scale':1.0, 'size'=n_samples}}
    Output:
        x_samples - the transformed vector with desired distribution
    """

    if "args" not in adict.keys():
        adict["args"] = {}

    if "kwargs" not in adict.keys():
        adict["kwargs"] = {}

    method_gen = getattr(stats, adict["dist"])
    method_specific = method_gen(**adict["args"], **adict["kwargs"])
    x_samples = method_specific.ppf(x)

    return x_samples


def eval_expr_for_sample(x, col_map, expr):
    """
    Inputs:
        x - 1D array with shape = number of symbols
        col_map - dictionary with keys = symbols, values = columns of X
        expr - str expression that gives y = f(X)
    Output:
        evaluated expression
    """
    # print(type(x), type(col_map), type(expr))
    # print(x.shape)

    # exit clause when expr not provided
    if not expr:
        return

    # for a sample, build a dictionary of symbol:value
    my_sub = {}

    for i, key_symbol in enumerate(col_map.keys()):
        my_sub[key_symbol] = x[i]

    parser = MathParser(my_sub)
    out_value = parser.parse(expr)

    return out_value


def sigmoid(x, k=1.0, x0=None):
    """sigmoid/logistic function"""

    if x0 is None:
        x0 = x.mean()
    sig = 1.0 / (1.0 + np.exp(-k * (x - x0)))

    return sig


# this is an example call from script....
# X, y_reg, y_prob, y_labels = make_tabular_data(
#    n_samples=1000, cov=cov, col_map=col_map, expr=expr, p_thresh=0.5)


def generate_x_noise(X, noise_level_x, seed=None):
    """
    inputs - X (used to determine shape of output matrix)
            noise_level_x : strength of the noise (range 0 to 1)
    outputs:
        x_noise - array with the same dimension as X with gaussian white noise
    """
    n_samples = X.shape[0]
    n_total = X.shape[1]

    # generate covariance matrix - 1's on diagonal - everything else is 0
    cov = np.zeros((n_total, n_total))
    np.fill_diagonal(cov, 1.0)

    # generate our gaussian white noise
    means = np.zeros(n_total)
    mvnorm = stats.multivariate_normal(mean=means, cov=cov)
    x_noise = noise_level_x * mvnorm.rvs(n_samples, random_state=seed)
    # print("in noise x_noise - ", x_noise.shape)

    return x_noise


def resolve_covariant(n_total, covariant=None):
    """Resolves a covariant in the following cases:
        - If a covariant is not provided a diagonal matrix of 1s is generated, and symmetry is checked via a comparison with the datasets transpose
        - If a covariant is provided, the symmetry is checked

    args:
        n_total {int} -- total number of informative features
        covariant {[type]} -- [description] (default: {None})
    returns:
        covariant {np_array}
    """

    if covariant is None:
        print("No covariant provided, generating one.")
        covariant = np.diag(np.ones(n_total))

    # test for symmetry on covariance matrix by comparing the matrix to its transpose
    assert np.all(
        covariant == covariant.T
    ), "Assertion error - please check covariance matrix is symmetric."

    return covariant


def pre_data_generation_checks(n_informative, col_map, n_total):
    """This function is used to ensure input, and input combinations are correct before
    generating synthetic data

    args:
        n_informative {int} -- n_informative - number of informative features - need to appear at least once in expression
        col_map {dict} -- dictionary mapping str symbols to columns
        n_total {int} -- total number of samples in the dataset
    """
    assert n_informative == len(
        col_map
    ), "number of dictionary keys in col_map not equal to n_informative."

    assert (
        n_total > 0
    ), "total number of samples (n_informative + n_nuisance) must be greater than 0"


def generate_redundant_features(x, n_informative, n_redundant, seed):
    generator = np.random.RandomState(seed)
    B = 2 * generator.rand(n_informative, n_redundant) - 1
    # B = 2 * random_state.rand(n_informative, n_redundant) - 1
    # print("in main script - b")
    # print(B)
    # print("in main script - x")
    # print(x)
    x_redundant = np.dot(x, B)
    # print("in synthetic_data - ")
    # print(x_redundant)

    return x_redundant


def scaler_check(scaler):

    if (
        not (
            issubclass(scaler.__class__, BaseEstimator)
            and issubclass(scaler.__class__, TransformerMixin)
        )
        and scaler is not None
    ):
        raise TypeError("Please provide a valid sklearn scaler.")


def make_tabular_data(
    n_samples=1000,
    n_informative=2,
    n_redundant=0,
    n_nuisance=0,
    n_classes=2,
    dist=[],
    cov=None,
    col_map={},
    expr=None,
    sig_k=1.0,
    sig_x0=None,
    p_thresh=0.5,
    noise_level_x=0.0,
    noise_level_y=0.0,
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    seed=None,
):
    """
    Use copulas and marginal distributions to build the joint probability of X.
    args:
        n_samples - number of samples to generate
        n_informative - number of informative features - need to appear at least once in expression
        n_redundant - number of redundant features
        n_nuisance - number of nuiscance features with no signal (noise)
        n_classes - number of classes for labeling (default is binary classification)
        dist - list of dicts for marginal distributions to apply to columns
            each dict specifies column, distribution and dictionary of args & kwds
            suport for distributions available in  scipy stats:
            https://docs.scipy.org/doc/scipy/reference/stats.html
        cov - a symmetric matrix specifying covariance amongst features
        col_map - dictionary mapping str symbols to columns
        expr - str expression holding y = f(x)
        p_thresh - probability threshold for assigning class labels
        noise_level_x (float) - level of white noise (jitter) added to x
        noise_level_y (float) - level of white noise added to y (think flip_y)
        scaler (sklearn scaler) - sklearn style scaler. Defaults to MinMaxScaler(feature_range = (-1,1)).
                              If None, no feature scaling is performed.
        seed - numpy random state object for repeatability




    returns X: array of shape [n_samples, n_total] where n_total = n_inform + n_nuisance, etc.
            y: array of shape [n_samples] with our labels
            y_reg: array of shape [n_samples] with regression values which get split for labels
    """

    n_total = n_informative + n_redundant + n_nuisance
    x_final = np.zeros((n_samples, n_total))

    pre_data_generation_checks(
        n_informative=n_informative, col_map=col_map, n_total=n_total
    )
    scaler_check(scaler)

    # generate covariance matrix if not handed one
    cov = resolve_covariant(n_informative, covariant=cov)

    # initialize X array
    means = np.zeros(n_informative)
    mvnorm = stats.multivariate_normal(mean=means, cov=cov)
    x = mvnorm.rvs(n_samples, random_state=seed)
    # x_cont = np.zeros_like(x)

    # now tranform marginals back to uniform distribution
    norm = stats.norm()
    x_cont = norm.cdf(x)

    # print("x_cont.shape - ", x.shape)
    # print("x_cont - ")
    # print(x_cont)
    # at this point x_cont has columns with correlation & uniform dist

    # apply marginal distributions

    for a_dist in dist:
        col = a_dist["column"]
        # method = getattr(stats, a_dist["dist"])
        # x_cont[:, col] = method.ppf(x_unif[:, col])
        x_cont[:, col] = transform_to_distribution(x_cont[:, col], a_dist)
    # print(x_cont.max())
    x_final[:, :n_informative] = x_cont

    # add redundant - lines 224-228
    # https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/datasets/_samples_generator.py#L37

    if n_redundant > 0:
        x_redundant = generate_redundant_features(
            x_cont, n_informative, n_redundant, seed
        )
        x_final[:, n_informative : n_informative + n_redundant] = x_redundant

    if n_nuisance > 0:
        x_nuis = np.random.rand(n_samples, n_nuisance)
        # x_final = np.concatenate((x_final, x_nuis), axis=1)
        x_final[:, -n_nuisance:] = x_nuis
    #    else:
    #        x_final = x_cont

    # Rescale to feature range
    if scaler is not None:
        x_final = scaler.fit_transform(x_final)

    # apply expression to each sample of X[mapped_cols,:]
    #    y_reg, y_prob, y_labels = calculate_y(X, p_thresh=0.5)

    # extract the columns we need from X
    col_list = list(col_map.values())
    x_filt = x_final[:, col_list]  # same as [:,:n_informative] - or should be
    y_reg = np.apply_along_axis(eval_expr_for_sample, 1, x_filt, col_map, expr)
    y_reg = y_reg.astype("float32")

    y_prob = sigmoid(y_reg, k=sig_k, x0=sig_x0)
    y_labels = y_prob >= p_thresh

    #
    # post processing steps - e.g. add noise
    #

    if noise_level_x > 0.0:
        x_noise = generate_x_noise(
            x_final[:, :n_informative], noise_level_x, seed=seed
        )
        x_final[:, :n_informative] = x_final[:, :n_informative] + x_noise

    return x_final, y_reg, y_prob, y_labels


def make_data_from_report(
    report: dict,
    n_samples: int = None,
    noise_level: float = 0.0,
    seed=None,
) -> pd.DataFrame:
    """
    Use a DataProfiler report to generate a synthetic data set to mimic the report.
    args:
        report (dict) - DataProfiler report
        n_samples (int) - number of samples to generate
        noise_level (float) - level of white noise (jitter) added to x
        seed - numpy random state object for repeatability

    returns X: DataFrame of shape [n_samples, n_total]
    """

    # make sure correlation matrix was generated
    if report["global_stats"]["correlation_matrix"] is None:
        raise Exception("The report must have the correlation matrix enabled")

    # make sure no non-numerical columns exist
    for stat in report["data_stats"]:
        if stat["data_type"] not in ["int", "float"]:
            raise Exception("The function only supports numerical variables")

    # if n_samples not provided, generate same samples as original dataset
    if not n_samples:
        n_samples = report["global_stats"]["samples_used"]

    n_informative = len(report["data_stats"])

    # build covariance matrix
    R = report["global_stats"]["correlation_matrix"]

    stddevs = [stat["statistics"]["stddev"] for stat in report["data_stats"]]
    D = np.diag(stddevs)

    cov = D @ R @ D
    cov = cov.round(decimals=8)  # round to avoid failing symmetry check

    # create col_map of appropriate length to pass pre_data_generation_checks
    col_map = {}
    for i in range(n_informative):
        col_map[f"x{i+1}"] = i

    x_final, _, _, _ = make_tabular_data(
        n_samples=n_samples,
        n_informative=n_informative,
        cov=cov,
        col_map=col_map,
        noise_level_x=noise_level,
        seed=seed,
    )

    # generate scalers by range of values in original data
    scalers = {}
    for col, stat in enumerate(report["data_stats"]):
        _min = stat["statistics"]["min"]
        _max = stat["statistics"]["max"]
        scalers[col] = MinMaxScaler(feature_range=(_min, _max))

    # rescale to feature range
    for col in scalers:
        x_final[:, col] = (
            scalers[col]
            .fit_transform(x_final[:, col].reshape(-1, 1))
            .flatten()
        )

    # find number of decimals for each column and round the data to match
    precisions = [
        stat["samples"][0][::-1].find(".") for stat in report["data_stats"]
    ]

    for i, precision in enumerate(precisions):
        x_final[:, i] = np.around(
            x_final[:, i], precision if precision > 0 else 0
        )

    # return x_final in a DataFrame with the original column names
    return pd.DataFrame(
        x_final, columns=[stat["column_name"] for stat in report["data_stats"]]
    )
