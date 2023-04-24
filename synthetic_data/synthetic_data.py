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
import scipy.interpolate as interpolate

from synthetic_data.marginal_dist import detect_dist
from synthetic_data.null_replication import replicate_null
from synthetic_data.parser import MathParser


def multinomial_ppf(x_uniform, my_dist, categories_info):
    """
    input:
        x_uniform - uniform distribution random variable
        my_dist - scipy.stats.multinomial object
    output:
        x_sampled - vector of samples from the multinomial
    """
    discrete_cdf = np.cumsum(my_dist.p)
    #    print("pmf - ", my_dist.p)
    #    print("cdf - ", discrete_cdf)
    x_sampled = np.zeros_like(x_uniform)
    # print(
    #    "test uniformity - ",
    #    x_uniform.min(),
    #    x_uniform.max(),
    #    x_uniform.mean(),
    #    x_uniform.std(),
    # )
    # search discrete CDF
    mapping_order = categories_info["mapping_order"]
    category_mapping = categories_info["category_mapping"]
    for j, x_u in enumerate(x_uniform):
        if x_u <= discrete_cdf[0]:
            x_sampled[j] = category_mapping[mapping_order[0]]
        else:
            for i, _ in enumerate(discrete_cdf[:-1]):
                if (discrete_cdf[i] < x_u) and (x_u <= discrete_cdf[i + 1]):
                    x_sampled[j] = category_mapping[mapping_order[i + 1]]
                    break

    return x_sampled


def transform_to_distribution(x, adict):
    """
    Input:
        x - input uniform distributed random variable
        adict - dictionary corresponding to the desired distribution (name & params)
        e.g. {'col':[list], 'dist':'normal', 'kwargs':{'loc':0.0, 'scale':1.0, 'size'=n_samples}}
    Output:
        x_samples - the transformed vector with desired distribution
    """

    if "args" not in adict:
        adict["args"] = []

    if "kwargs" not in adict:
        adict["kwargs"] = {}

    # DP categoricals will be multinomial
    if adict["dist"] in ["multinomial", "norm", "skewnorm"]:
        method_gen = getattr(stats, adict["dist"])
        method_specific = method_gen(*adict["args"], **adict["kwargs"])
        if adict["dist"] == "multinomial":
            x_samples = multinomial_ppf(x, method_specific, adict['categories_info'])
        else:
            x_samples = method_specific.ppf(x)
    else:
        x_samples = adict["args"].ppf(x)

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
    np.testing.assert_almost_equal(
        covariant, covariant.T, 1e-8
        , "Assertion error - please check covariance matrix is symmetric.")

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


def marginal_dist_check(dist, num_cols):
    """
    Checks if dist argument passed to make_tabular_data is valid.
    args:
        dist - list of dicts for marginal distributions to apply to columns
    """
    if len(dist) != num_cols and len(dist) > 0:
        raise ValueError(
            "When providing a marginal distribution list, ensure the length of "
            "the list is equal to n_informative columns."
        )


def make_tabular_data(
    n_samples=1000,
    n_informative=2,
    n_redundant=0,
    n_nuisance=0,
    n_classes=2,
    dist=None,
    cov=None,
    col_map=None,
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
        scaler (sklearn scaler) - sklearn style scaler.
            Defaults to MinMaxScaler(feature_range = (-1,1)).
            If None, no feature scaling is performed.
        seed - numpy random state object for repeatability




    returns X: array of shape [n_samples, n_total] where n_total = n_inform + n_nuisance, etc.
            y: array of shape [n_samples] with our labels
            y_reg: array of shape [n_samples] with regression values which get split for labels
    """
    if dist is None:
        dist = []

    n_total = n_informative + n_redundant + n_nuisance
    x_final = np.zeros((n_samples, n_total))

    pre_data_generation_checks(
        n_informative=n_informative, col_map=col_map, n_total=n_total
    )
    scaler_check(scaler)
    marginal_dist_check(dist, n_informative)

    # generate covariance matrix if not handed one
    cov = resolve_covariant(n_informative, covariant=cov)

    # initialize X array
    means = np.zeros(n_informative)
    # if coming from make_data_from_report - that data won't be standardized...

    for i, a_dist in enumerate(dist):
        if a_dist.get("mean") is not None:
            means[i] = a_dist["mean"]

    mvnorm = stats.multivariate_normal(mean=means, cov=cov, allow_singular=True)
    x = mvnorm.rvs(n_samples, random_state=seed)

    # now tranform marginals back to uniform distribution
    x_cont = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_tmp = x[:, i]
        # print(i, x_tmp.mean(), x_tmp.std())
        tmp_norm = stats.norm(loc=x_tmp.mean(), scale=x_tmp.std())
        x_cont[:, i] = tmp_norm.cdf(x_tmp)

    # print("x_cont.shape - ", x.shape)
    # print("x_cont - ")
    # print(x_cont)
    # at this point x_cont has columns with correlation & uniform dist

    # apply marginal distributions
    for a_dist in dist:
        col = a_dist["column"]
        x_cont[:, col] = transform_to_distribution(x_cont[:, col], a_dist)
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
        x_noise = generate_x_noise(x_final[:, :n_informative], noise_level_x, seed=seed)
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
    # cov = cov.round(decimals=8)  # round to avoid failing symmetry check

    # create col_map of appropriate length to pass pre_data_generation_checks
    col_map = {}
    for i in range(n_informative):
        col_map[f"x{i+1}"] = i

    dist = detect_dist(report)

    x_final, _, _, _ = make_tabular_data(
        n_samples=n_samples,
        n_informative=n_informative,
        cov=cov,
        col_map=col_map,
        noise_level_x=noise_level,
        seed=seed,
        dist=dist,
        scaler=None,
    )

    # Approximate the original data format given its precision / # of digits
    for i, col_stat in enumerate(report["data_stats"]):
        digits = 0
        if col_stat['data_type'] not in ['int', 'float']:
            continue
        if col_stat['data_type'] in ['float']:
            precision = col_stat.get('statistics', {}).get('precision', {}).get('max', 0)
            digits = precision - np.ceil(np.log10(np.abs(x_final[:, i])))
            digits = int((digits[np.isfinite(digits)]).max())
        x_final[:, i] = np.around(x_final[:, i], digits)

    # replicate null values if null replication metrics exist in the original report
    col_to_null_metrics = {}
    for col_id, col_data_stats in enumerate(report["data_stats"]):
        if "null_replication_metrics" not in col_data_stats:
            continue
        col_to_null_metrics[col_id] = col_data_stats["null_replication_metrics"]
    x_final = replicate_null(x_final, col_to_null_metrics, cov)

    # return x_final in a DataFrame with the original column names
    return pd.DataFrame(
        x_final, columns=[stat["column_name"] for stat in report["data_stats"]]
    )
