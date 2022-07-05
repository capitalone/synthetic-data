from synthetic_data.multivar_skew_norm import MultivariateSkewNorm
import numpy as np
from scipy.stats import skew, skewnorm
from scipy.special import kl_div
from collections import Counter

# Covariance matrices must be positive semi-definite
cov = np.asarray([[1.5,-0.9],[-0.9,1.5]])

FIRST_VAR_SKEW = 7
SECOND_VAR_SKEW = -3
skews = np.asarray([FIRST_VAR_SKEW, SECOND_VAR_SKEW])

mvsn = MultivariateSkewNorm(cov, skews)

def test_array_shapes():
    n = mvsn.dim
    assert mvsn.skew.shape[0] == n
    assert mvsn.cov.shape[0] == n
    assert mvsn.delta.shape[0] == n
    assert mvsn.capital_delta.shape[0] == n
    assert mvsn.omega.shape[0] == n   
    assert mvsn.alpha is None, "alpha array should be None on initialization"

def test_repeatability():
    X = mvsn.rvs(1000, 1)
    X_repeat = mvsn.rvs(1000, 1)
    assert np.all(X == X_repeat), "The same samples must be generated if given the same seed"

def test_marginal_dist():
    X = mvsn.rvs(10000)
    X0, X1 = X[:,0], X[:,1]
    Y0, Y1 = skewnorm.rvs(FIRST_VAR_SKEW, size=10000), skewnorm.rvs(SECOND_VAR_SKEW, size=10000)

    # The skew, mean and variance of the marginal distributions of the mvsn should be similar to those of the univariate skew normal distributions
    assert np.abs(skew(X0) - skew(Y0)) <= 0.2 and np.abs(skew(X1) - skew(Y1)) <= 0.2, "skew difference exceeds threshold"
    assert np.abs(np.mean(X0) - np.mean(Y0)) <= 0.2 and np.abs(np.mean(X1) - np.mean(Y1)) <= 0.2, "mean difference exceeds threshold"
    assert np.abs(np.std(X0) - np.std(Y0)) <= 0.2 and np.abs(np.std(X1) - np.std(Y1)) <= 0.2, "standard deviation difference exceeds threshold"

def test_kl_divergence():
    PRECISION = 2
    SAMPLE_SIZE = pow(10, 6)

    X = mvsn.rvs(SAMPLE_SIZE)
    X = np.round(X, PRECISION)

    freq = Counter([tuple(x) for x in X])
    empirical_prob = [freq[x] / SAMPLE_SIZE for x in freq]

    X = [x for x in freq]

    # Discretize Probability Density Functions to (approximate) Probability Mass Functions by multiplying PDF with dx
    # i.e For the univariate case, Riemann Sum approximation of area under a PDF to get probability
    dx = pow(10, (-1 * 2 * PRECISION))
    groundtruth_prob = mvsn.pdf(X)
    groundtruth_prob *= dx

    assert sum(kl_div(groundtruth_prob, empirical_prob)) < 0.5