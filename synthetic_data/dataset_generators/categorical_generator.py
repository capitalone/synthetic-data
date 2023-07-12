import numpy as np
import pandas as pd
from numpy.random import Generator
from typing import List, Optional


def random_categorical(
    rng: Generator, categories: Optional[List[str]] = None, num_rows: int = 1
) -> np.array:
    """
    Randomly generates an array of categorical chosen out of categories

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param categories: a list of values that are allowed in a categorical or None
    :type categories: string, None, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional

    :return: np array of categories
    """
    if categories is None:
        categories = ["A", "B", "C", "D", "E"]

    return rng.choice(categories, (num_rows,))
