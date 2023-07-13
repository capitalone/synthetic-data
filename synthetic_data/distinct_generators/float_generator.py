import numpy as np
from numpy.random import Generator


def random_floats(
    rng: Generator,
    min_value: int = -1e6,
    max_value: int = 1e6,
    sig_figs: int = 3,
    num_rows: int = 1,
) -> np.array:
    """
    Randomly generates an array of floats between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min_value: the minimum float that can be returned
    :type min_value: int, optional
    :param max_value: the maximum float that can be returned
    :type max_value: int, optional
    :param sig_figs: restricts float to a number of sig_figs after decimal
    :type sig_figs: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of floats
    """
    if sig_figs < 0:
        raise ValueError("sig_figs should be greater than or equal to 0")
    if not isinstance(sig_figs, int):
        raise ValueError("sig_figs should be an int")
    return np.around(rng.uniform(min_value, max_value, num_rows), sig_figs)
