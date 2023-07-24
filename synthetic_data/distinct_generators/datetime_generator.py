"""Contains a datetime generator."""
from datetime import datetime
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

    # this will return the datetime object instead of a string for sorting
    return datetime.strptime(ptime, date_format)


def random_datetimes(
    rng: Generator,
    date_format: Optional[list[str]] = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    num_rows: int = 1,
) -> np.array:
    """
    Datetime needs to sort the order already. A column has multiple formats.
    Dataset_generator will call this to generate datetimes and then will see if
    sorts, but this will always already be sorted.

    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format_list: the format that the generated datatime will follow,
        defaults to None
    :type date_format: List, None, optional
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
    if not date_format:
        date_format = ["%B %d %Y %H:%M:%S"]

    for i in range(num_rows):
        datetime = generate_datetime(
            rng, date_format=date_format, start_date=start_date, end_date=end_date
        )
        date_list[i] = datetime

    # sort and then convert into string format
    sorted_data = sorted(date_list)
    sorted_data_strings = np.array([dt.strftime(date_format) for dt in sorted_data])

    return sorted_data_strings
