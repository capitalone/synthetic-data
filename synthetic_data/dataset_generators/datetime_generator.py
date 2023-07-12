from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import Generator


def generate_datetime(
    rng: Generator,
    date_format: str,
    start_date: pd.Timestamp = pd.Timestamp(1920, 1, 1),
    end_date: pd.Timestamp = pd.Timestamp(2049, 12, 31),
) -> str:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, None, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to pd.Timestamp(1920, 1, 1)
    :type start_date: pd.Timestamp, None, optional
    :param end_date: the latest date that datetimes can be generated at,
        defaults to pd.Timestamp(2049, 12, 31)
    :type end_date: pd.Timestamp, None, optional

    :return: generated datetime
    :rtype: str
    """
    t = rng.random()
    ptime = start_date + t * (end_date - start_date)

    return ptime.strftime(date_format)


def random_datetimes(
    rng: Generator,
    date_format_list: Optional[str] = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    num_rows: int = 1,
) -> np.array:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, None, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to pd.Timestamp(1920, 1, 1)
    :type start_date: pd.Timestamp, None, optional
    :param end_date: the latest date that datetimes can be generated at,
        defaults to pd.Timestamp(2049, 12, 31)
    :type end_date: pd.Timestamp, None, optional

    :return: array of generated datetimes
    :rtype: numpy array
    """
    date_list = [""] * num_rows
    if not date_format_list:
        date_format_list = ["%B %d %Y %H:%M:%S"]

    for i in range(num_rows):
        date_format = rng.choice(date_format_list)
        datetime = generate_datetime(
            rng, date_format=date_format, start_date=start_date, end_date=end_date
        )
        date_list[i] = datetime

    return np.array(date_list)
