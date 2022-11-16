import numpy as np


def replicate_null(data_array, null_metrics, cov):
    """
    Incorporates null values into synthetic data to replicate null values from original data

    Args:
        data_array (np.ndarray): data to incorporate null values into
        null_metrics (dict): dict mapping column id to their 'null_replication_metrics' (if exists)
        cov (np.ndarray): covariance matrix from original data report
    Returns:
        data_array (np.ndarray): synthetic data with null values
    """
    col_null_indices = dict()
    for col_id, col_null_metrics in null_metrics.items():
        class_prior = col_null_metrics["class_prior"]
        class_mean = col_null_metrics["class_mean"]
        shared_cov = np.delete(cov, col_id, 0)
        shared_cov = np.delete(shared_cov, col_id, 1)
        X = np.delete(data_array, col_id, 1)
        preds = _lda_predict(X, class_prior, class_mean, shared_cov)
        col_null_indices[col_id] = preds

    for col_id, null_indices in col_null_indices.items():
        data_array[:, col_id][null_indices == 1] = None

    return data_array


def _lda_predict(X, priors, means, cov):
    """
    Linear Discriminant Analysis based binary classifier to determine whether column values should be null (1) or not (0).

    Posterior calculation formula based on: 'https://scikit-learn.org/stable/modules/lda_qda.html'.

    P(x) omitted from above formula as the value remains same among different classes and therefore do not need it
    for relative comparison of posterior values.

    Args:
        X (np.ndarray): data array
        priors (list): 'class_prior' field of a column's 'null_replication_metrics'
        means (list): 'class_mean' field of a column's 'null_replication_metrics'
        cov (np.ndarray): covariance matrix from original data report with the values of column being processed removed
    Returns:
        preds (np.ndarray): Binary array indicating if a column value should be changed to null (1) or kept unchanged (0)
    """
    preds = list()
    for x in X:
        posts = list()
        for c in [0, 1]:
            prior = np.log(priors[c])
            inv_cov = np.linalg.inv(cov)
            diff = x - means[c]
            likelihood = -0.5 * diff.T @ inv_cov @ diff
            posterior = prior + likelihood
            posts.append(posterior)
        pred = np.argmax(posts)
        preds.append(pred)
    return np.array(preds)
