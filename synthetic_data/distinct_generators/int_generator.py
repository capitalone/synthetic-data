import numpy as np
from ..base_generator import BaseGenerator
from numpy.random import Generator

def random_integers(rng: Generator,
                    min_value: int = -1e6,
                    max_value: int = 1e6,
                    num_rows: int = 1
                    ) -> np.array:
    """
    Randomly generates an array of integers between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param min_value: the minimum integer that can be returned
    :type min_value: int, optional
    :param max_value: the maximum integer that can be returned
    :type max_value: int, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of integers
    """
    return rng.integers(min_value, max_value, (num_rows,))