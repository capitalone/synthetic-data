import dataprofiler as dp
import numpy as np
import pandas as pd
from sklearn import datasets
from synthetic_data.marginal_dist import (
    detect_dist,
    _detect_dist_discrete,
    _detect_dist_continuous,
    _gen_rv_hist_continuous,
)
from scipy import stats


def test_marginal_dist_detection():

    np.random.seed(0)
    data = datasets.load_iris(as_frame=True).frame

    profile_options = dp.ProfilerOptions()
    profile_options.set(
        {
            "data_labeler.is_enabled": False,
            "correlation.is_enabled": True,
            "structured_options.multiprocess.is_enabled": False,
        }
    )

    profile = dp.Profiler(data, options=profile_options)
    report = profile.report()
    marginal_dist_list = detect_dist(report)

    assert len(marginal_dist_list) == len(
        report["data_stats"]
    ), "Length of distributions list must be equal to number of columns"

    for col_num, col in enumerate(report["data_stats"]):
        dist_name = marginal_dist_list[col_num]["dist"]

        assert hasattr(
            stats, dist_name
        ), "The detected distribution must be defined in scipy.stats"
        dist_method = getattr(stats, dist_name)
        if col["data_type"] == "float":
            assert issubclass(
                dist_method, stats.rv_histogram
            ), "Detected distribution must be continuous for columns with continuous random variables"
        else:
            assert isinstance(
                dist_method, type(stats.multinomial)
            ), "Detected distribution must be discrete for columns with discrete random variables"


def test_discrete_dist_detection():

    np.random.seed(0)
    data = {
        "randint": stats.randint.rvs(0, 5, size=1000),
        "randint_nonzero_min": stats.randint.rvs(2, 7, size=1000),
        "binomial": stats.binom.rvs(5, 0.25, size=1000),
    }
    data = pd.DataFrame(data)

    profile_options = dp.ProfilerOptions()
    profile_options.set(
        {
            "data_labeler.is_enabled": False,
            "correlation.is_enabled": True,
            "structured_options.multiprocess.is_enabled": False,
        }
    )

    profile = dp.Profiler(data, options=profile_options)
    report = profile.report()
    for col in report["data_stats"]:
        col_name, col_stats = col["column_name"], col["statistics"]
        detected_dist = _detect_dist_discrete(col_stats)
        #if col_name == "randint" or col_name == "randint_nonzero_min":
        #    assert detected_dist["dist"] == "randint"
        #elif col_name == "binomial":
        #    assert detected_dist["dist"] == "binom"
        assert detected_dist["dist"] == 'multinomial'


def test_gen_rv_hist_continuous():

    np.random.seed(0)
    data = {
        "uniform": stats.uniform.rvs(size=1000),
        "normal": stats.norm.rvs(size=1000),
        "normal_nonzero_mean": stats.norm.rvs(5, 2, size=1000)
    }
    data = pd.DataFrame(data)

    profile_options = dp.ProfilerOptions()
    profile_options.set(
        {
            "data_labeler.is_enabled": False,
            "correlation.is_enabled": True,
            "structured_options.multiprocess.is_enabled": False,
        }
    )

    profile = dp.Profiler(data, options=profile_options)
    report = profile.report()
    for col in report["data_stats"]:
        col_name, col_stats = col["column_name"], col["statistics"]

        detected_dist = _gen_rv_hist_continuous(col_stats)
        assert detected_dist["dist"] == "rv_histogram"
        assert isinstance(detected_dist["args"],  stats.rv_histogram)


def test_continuous_dist_detection():

    np.random.seed(0)
    data = {
        "uniform": stats.uniform.rvs(size=1000),
        "normal": stats.norm.rvs(size=1000),
        "normal_nonzero_mean": stats.norm.rvs(5, 2, size=1000)
    }
    data = pd.DataFrame(data)

    profile_options = dp.ProfilerOptions()
    profile_options.set(
        {
            "data_labeler.is_enabled": False,
            "correlation.is_enabled": True,
            "structured_options.multiprocess.is_enabled": False,
        }
    )

    profile = dp.Profiler(data, options=profile_options)
    report = profile.report()
    for col in report["data_stats"]:
        col_name, col_stats = col["column_name"], col["statistics"]

        detected_dist = _detect_dist_continuous(col_stats)
        #if col_name == "uniform":
        #    assert detected_dist["dist"] == "uniform"
        #elif col_name == "normal" or col_name == "normal_nonzero_mean":
        #    assert detected_dist["dist"] == "norm"
        assert detected_dist["dist"] == "skewnorm"
