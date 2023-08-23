"""Contains a categorical generator."""
from typing import List, Optional

import numpy as np
from numpy.random import Generator


def random_categorical(
    rng: Generator,
    categories: Optional[List[str]] = None,
    probabilities: Optional[List[float]] = None,
    num_rows: int = 1,
) -> np.array:
    """
    Randomly generates an array of categorical values chosen out of categories.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param categories: a list of values that are allowed in categorical values \
        or defaults to ["A", "B", "C", "D", "E"] if None
    :type categories: string, None, optional
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional
    :param probabilities: a list of floats that respresent the
        probability each category will be chosen
    :type probabilities: float, optional

    :return: np array of categories
    """
    if categories is None:
        categories = ["A", "B", "C", "D", "E"]
    if probabilities is None:
        return rng.choice(categories, size=num_rows)

    if len(categories) != len(probabilities):
        raise ValueError("categories and probabilities must be of the same length")
    if not np.isclose(sum(probabilities), 1):
        raise ValueError("Probabilities must sum to 1")
    return rng.choice(categories, size=num_rows, p=probabilities)
