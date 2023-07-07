from numpy.random import Generator
import string
import numpy as np

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