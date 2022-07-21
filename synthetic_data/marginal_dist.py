from collections import Counter

from scipy import stats


def _detect_dist_continuous(col_stats):
    """
    Detects type of continuous distribution based on Kolmogorov-Smirnov Goodness-of-fit test, https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.
    Args:
        col_stats (dict): Column data statistics. The column data must be of a continuous numerical random variable.
    Returns:
        dist (dict): Dictionary stating distribution type along with other parameters for the distribution.
    """
    bin_counts, bin_edges = (
        col_stats["histogram"]["bin_counts"],
        col_stats["histogram"]["bin_edges"],
    )

    # Create a continuous distribution from the histogram and sample data from it
    hist_dist = stats.rv_histogram((bin_counts, bin_edges))
    hist_mean = hist_dist.mean()
    observed_samples = hist_dist.rvs(size=1000)

    # Center the distribution around 0
    observed_samples -= hist_mean

    # Distributions to test against (must be a continuous distribution from scipy.stats)
    # Distribution name -> list of positional arguments for the distribution
    # If the observed histogram is centered around 0, means of distributions set to 0
    test_dists = (
        # norm(loc, scale)
        ("norm", (0, col_stats["stddev"])),
        # skewnorm(a, loc, scale)
        ("skewnorm", (col_stats["skewness"], 0, col_stats["stddev"])),
        # uniform(loc, scale)
        ("uniform", (col_stats["min"], col_stats["max"] - col_stats["min"])),
    )

    dist = {}
    max_p = 0

    for dist_name, dist_args in test_dists:

        # overfitting on purpose for testing
        # method = getattr(stats, dist_name)
        # dist_args = method.fit(observed_samples)

        p = stats.kstest(observed_samples, dist_name, dist_args)[1]
        if p > max_p:
            dist["dist"] = dist_name
            dist["args"] = dist_args
            max_p = p

    return dist


def _detect_dist_discrete(col_stats):
    """
    Detects type of discrete distribution based on Pearson's Chi-Squared Goodness-of-fit test, https://en.wikipedia.org/wiki/Chi-squared_test.
    Args:
        col_stats (dict): Column data statistics. The column data must be of a discrete numerical random variable.
    Returns:
        dist (dict): Dictionary stating distribution type along with other parameters for the distribution.
    """
    # Convert strings to ints, ideally DataProfiler should be handling this
    categories = [int(category) for category in col_stats["categories"]]
    categorical_count = {int(k): v for k, v in col_stats["categorical_count"].items()}

    categories.sort()
    observed_freq = [categorical_count[category] for category in categories]
    sample_size = col_stats["sample_size"]

    # Shift categories so they start at 0
    categories = list(range(len(categories)))

    # Distributions to test against (must be a discrete distribution from scipy.stats)
    # Distribution name -> list of positional arguments for the distribution
    test_dists = (
        # binom(n, p)
        *[("binom", (categories[-1], p * 0.25)) for p in range(1, 4)],
        # randint(low, high)
        ("randint", (0, categories[-1] + 1)),
    )

    dist = {}
    max_p = 0

    for dist_name, args in test_dists:
        dist_method = getattr(stats, dist_name)
        dist_method_specific = dist_method(*args)
        test_samples = dist_method_specific.rvs(size=sample_size)
        test_samples_count = Counter(test_samples)
        expected_freq = [test_samples_count[category] for category in categories]
        p = stats.chisquare(observed_freq, expected_freq)[1]
        if p > max_p:
            dist["dist"] = dist_name
            dist["args"] = args
            max_p = p

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
        is_continuous = col_dict["data_type"] == "float"
        dist = (
            _detect_dist_continuous(col_stats)
            if is_continuous
            else _detect_dist_discrete(col_stats)
        )
        dist["column"] = col_num
        dist["column_name"] = col_dict["column_name"]
        dist_list.append(dist)
    return dist_list
