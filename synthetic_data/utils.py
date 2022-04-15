import pathlib

import numpy as np


def resolve_output_path(file_path):
    """Resolves and creates the output path for a given examples artifacts

    Args:
        file_path (str): file path string

    Returns:
        [str]: path to the examples output
    """

    output_path = f"{file_path}/output"

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    return output_path


def tuples_to_cov(list_of_tuples, col_map):
    """
    Alternative form for specifying correlation between features, intended
    to be an easier specification, and less error prone (using symbolic names rather than indices)

    Input - list of tuples where each tuple is of the form:
            (sym_i, sym_j, correlation)
    Output - symmetric (n_features x n_features) array of cov
    """

    n_feat = len(col_map.keys())
    cov = np.zeros((n_feat, n_feat))

    for a_tuple in list_of_tuples:
        # map symbols to indices
        i = col_map[a_tuple[0]]
        j = col_map[a_tuple[1]]
        cov[i, j] = cov[j, i] = a_tuple[2]

    for i in range(n_feat):
        cov[i, i] = 1.0

    return cov
