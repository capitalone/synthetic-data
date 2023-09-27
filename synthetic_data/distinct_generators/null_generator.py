"""Contains a random Null generator."""
import numpy as np


def null_generation(num_rows: int = 1) -> np.array:
    """
    Randomly generates an array of integers between the given min and max values.

    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of null values
    """
    return np.array([None] * num_rows)
