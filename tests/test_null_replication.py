import numpy as np

from synthetic_data.null_replication import replicate_null


def test_null_replication():

    NUM_COLS = 5
    # dataset to incorporate null values into
    np.random.seed(111)
    test_data = np.random.rand(100, NUM_COLS)

    cov = np.asarray(
        [
            [0.07898725, 0.00412165, -0.00022342, 0.00725404, -0.01581922],
            [0.00412165, 0.10266713, 0.00146469, -0.00324403, 0.0048762],
            [-0.00022342, 0.00146469, 0.09017485, -0.00573846, 0.00635938],
            [0.00725404, -0.00324403, -0.00573846, 0.08524252, -0.01228459],
            [-0.01581922, 0.0048762, 0.00635938, -0.01228459, 0.09593231],
        ]
    )

    # replicate null values only in column at index 4
    null_metrics = {
        4: {
            "class_prior": [0.74, 0.26],
            "class_mean": [
                [
                    0.4561036075608999,
                    0.3535834542017815,
                    0.45888137614441876,
                    0.5267180870547946,
                ],
                [
                    0.7816550671278726,
                    0.7112568841047568,
                    0.46924282509845355,
                    0.49031505156389993,
                ],
            ],
        }
    }

    test_data = replicate_null(test_data, null_metrics, cov)

    # get null counts of each column
    null_counts = [
        np.count_nonzero(np.isnan(test_data[:, col_id])) for col_id in range(NUM_COLS)
    ]
    null_counts = np.asarray(null_counts)

    assert np.all(
        null_counts[:4] == 0
    ), "Columns with no null values in original data cannot have null values in synthetic data"

    assert (
        null_counts[4] > 0
    ), "Columns with null values in original data must have non-zero number of null values in synthetic data"
