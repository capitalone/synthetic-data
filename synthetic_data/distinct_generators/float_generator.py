"""Contains a random float generator."""
import numpy as np
from numpy.random import Generator


def random_floats(
    rng: Generator,
    min: int = -1e6,
    max: int = 1e6,
    precision: int = 3,
    num_rows: int = 1,
) -> np.array:
    """
    Randomly generates an array of floats between the given min and max values.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min: the minimum float that can be returned
    :type min: int, optional
    :param max: the maximum float that can be returned
    :type max: int, optional
    :param sig_figs: restricts float to a number of sig_figs after decimal
    :type sig_figs: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of floats
    """
    if precision < 0:
        raise ValueError("precision should be greater than or equal to 0")
    if not isinstance(precision, int):
        raise ValueError("precision should be an int")
    return np.around(rng.uniform(min, max, num_rows), precision)
