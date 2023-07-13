import numpy as np
import string
from numpy.random import Generator
from typing import List, Optional


def random_string(
    rng: Generator,
    chars: Optional[List[str]] = None,
    num_rows: int = 1,
    str_len_min: int = 1,
    str_len_max: int = 256,
) -> np.array:
    """
    Randomly generates an array of strings with length between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param chars: a list of values that are allowed in a string or None
    :type chars: List[str], None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional
    :param str_len_min: the minimum length a string can be
    :type str_len_min: int, optional
    :param str_len_max: the maximum length a string can be
    :type str_len_max: int, optional

    :return: numpy array of strings
    """
    if chars is None:
        chars = list(
            string.ascii_uppercase
            + string.ascii_lowercase
            + string.digits
            + " "
            + string.punctuation
        )
    string_list = []

    for _ in range(num_rows):
        length = rng.integers(str_len_min, str_len_max)
        string_entry = "".join(rng.choice(chars, (length,)))
        string_list.append(string_entry)

    return np.array(string_list)


def random_text(
    rng: Generator,
    chars: Optional[str] = None,
    num_rows: int = 1,
    str_len_min: int = 256,
    str_len_max: int = 1000,
) -> np.array:
    """
    Randomly generates an array of text with length between a min and max value

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param chars: a list of values that are allowed in a string or None
    :type chars: List[str], None
    :param num_rows: the number of rows in np array generated
    :type num_rows: int, optional
    :param str_len_min: the minimum length a string can be (must be larger than 255)
    :type str_len_min: int, optional
    :param str_len_max: the maximum length a string can be
    :type str_len_max: int, optional

    :return: numpy array of text
    """
    if str_len_min < 256:
        raise ValueError(
            f"str_len_min must be > 255. " f"Value provided: {str_len_min}."
        )

    return random_string(rng, chars, num_rows, str_len_min, str_len_max)
