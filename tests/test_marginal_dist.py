import dataprofiler as dp
import numpy as np
import pandas as pd
from sklearn import datasets
from synthetic_data.marginal_dist import detect_dist
from scipy import stats


def test_marginal_dist_detection():

    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    data.target = data.target.astype(int)

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
            assert isinstance(
                dist_method, stats.rv_continuous
            ), "Detected distribution must be continuous for columns with continuous random variables"
        else:
            assert isinstance(
                dist_method, stats.rv_discrete
            ), "Detected distribution must be discrete for columns with discrete random variables"
