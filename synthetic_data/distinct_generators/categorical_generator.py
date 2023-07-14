"""Contains a categorical generator."""
from typing import List, Optional

import numpy as np
from numpy.random import Generator


def random_categorical(
    rng: Generator, categories: Optional[List[str]] = None, num_rows: int = 1
) -> np.array:
    """
    Randomly generates an array of categorical chosen out of categories.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param categories: a list of values that are allowed in a categorical \
        or defaults to ["A", "B", "C", "D", "E"] if None
    :type categories: string, None, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of categories
    """
    if categories is None:
        categories = ["A", "B", "C", "D", "E"]
    if num_rows > len(categories):
        raise ValueError("num_rows exceeds number of categories")
    return rng.choice(categories, (num_rows,))
