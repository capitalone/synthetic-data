"""
Methods that support detecting and translating marginal distributions from a
DataProfiler report into scipy.stats distributions used by synthetic_data
"""
import numpy as np
from scipy import stats


def _gen_rv_hist_continuous(col_stats):
    """
    Detects type of continuous distribution based on Kolmogorov-Smirnov Goodness-of-fit test
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.
    Args:
        col_stats (dict): Column data statistics. The column data must be
        a continuous numerical random variable.
    Returns:
        dist (dict): Dictionary stating distribution type along with other parameters
        for the distribution.
    """

    bin_counts, bin_edges = (
        col_stats["histogram"]["bin_counts"],
        col_stats["histogram"]["bin_edges"],
    )

    # Create a continuous distribution from the histogram and sample data from it
    hist_dist = stats.rv_histogram((bin_counts, bin_edges))
    observed_samples = hist_dist.rvs(size=1000)
    dist = {'dist': 'rv_histogram', 'args': hist_dist}
    return dist


def _detect_dist_continuous(col_stats):
    """
    Detects type of continuous distribution based on Kolmogorov-Smirnov Goodness-of-fit test, https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.
    Args:
        col_stats (dict): Column data statistics. The column data must be of a continuous numerical random variable.
    Returns:
        dist (dict): Dictionary stating distribution type along with other parameters for the distribution.
    """
    # Distributions to test against (must be a continuous distribution from scipy.stats)
    # Distribution name -> list of positional arguments for the distribution
    dist = {}
    dist["dist"] = "skewnorm"
    dist["args"] = (col_stats["skewness"], col_stats["mean"], col_stats["stddev"])
    dist["mean"] = col_stats["mean"]
    dist["std"] = col_stats["stddev"]

    return dist


def _detect_dist_discrete(col_stats):
    """
    Constructs a scipy.stats.multinomial distribution for a feature based on
    the columns stats provided by a DataProfiler report.

    Args:
        col_stats (dict): Column data statistics. The column data must be of a
        discrete numerical random variable.
    Returns:
        dist (dict): Dictionary stating distribution type along with other parameters
        for the distribution.
    """
    # Convert strings to ints, ideally DataProfiler should be handling this
    categories = col_stats["categories"]

    # can sort if all numeric here
    try:
        mapping_order = np.argsort(list(map(int, categories)))
    except:
        mapping_order = list(range(len(categories)))

    category_mapping = {i: categories[i] for i in mapping_order}
    observed_freq = [col_stats["categorical_count"][categories[i]] for i in mapping_order]
    p = np.array(observed_freq) / sum(observed_freq)

    assert p.sum() <= 1.00000000000005, f"ppf is too big, {p}"
    assert p.sum() > 0.995, f"ppf is too small, {p}"

    dist = {}
    dist["dist"] = "multinomial"
    dist["args"] = (1, p)

    # categories need not be continuous
    # think of this as labels to be applied to samples
    dist["categories_info"] = {
        "category_mapping": category_mapping,
        "mapping_order": mapping_order,
    }


    return dist


def detect_dist(report):
    """
    Detects type of distribution modeled by each column of a DataProfiler report.
    Type of distribution selected for each column based on goodness-of-fit test
    (Chi-Squared test for discrete variables, Kolmogorov-Smirnov test for continuous variables).

    Args:
        report (dict): DataProfiler report
    Returns:
        dist_list (list): List of dictionaries stating distribution type for each column
    """
    dist_list = []

    for col_num, col_dict in enumerate(report["data_stats"]):
        col_stats = col_dict["statistics"]
        is_continuous = col_dict["data_type"] == "float" or not col_dict["categorical"]
        print(col_dict["column_name"], is_continuous)

        print(col_num, col_dict["data_type"])
        dist = (
            _gen_rv_hist_continuous(col_stats)
            if is_continuous
            else _detect_dist_discrete(col_stats)
        )
        dist["column"] = col_num
        dist["column_name"] = col_dict["column_name"]
        # print("marginal_dist line 121 ", dist)
        dist_list.append(dist)

    return dist_list
