"""Contains a random int generator."""
import numpy as np
from numpy.random import Generator


def random_integers(
    rng: Generator, min: int = -1e6, max: int = 1e6, num_rows: int = 1
) -> np.array:
    """
    Randomly generates an array of integers between the given min and max values.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min: the minimum integer that can be returned
    :type min: int, optional
    :param max: the maximum integer that can be returned
    :type max: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of integers
    """
    # rng.integers has an exclusive max length.
    # Need to ensure that the max of the data is n-1 the max param value.
    max += 1

    return rng.integers(min, max, (num_rows,))
