"""Contains a random text generator."""
import string
from typing import List, Optional

import numpy as np
from numpy.random import Generator


def random_text(
    rng: Generator,
    vocab: Optional[List[str]] = None,
    min: int = 1,
    max: int = 1000,
    num_rows: int = 1,
) -> np.array:
    """
    Randomly generates an array of text with lengths between the min and max values.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param vocab: a list of values that are allowed in a string or None
    :type vocab: List[str], None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional
    :param min: the minimum length a string can be
    :type min: int, optional
    :param max: the maximum length a string can be
    :type max: int (one above the max), optional

    :return: numpy array of strings
    :rtype: numpy array
    """
    if vocab is None:
        vocab = list(
            string.ascii_uppercase
            + string.ascii_lowercase
            + string.digits
            + " "
            + string.punctuation
        )
    text_list = []

    # rng.integers has an exclusive max length.
    # Need to ensure that the max of the data is n-1 the max param value.
    max += 1

    for _ in range(num_rows):
        length = rng.integers(min, max)
        string_entry = "".join(rng.choice(vocab, (length,)))
        text_list.append(string_entry)

    return np.array(text_list)
