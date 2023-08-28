"""Contains a datetime generator."""
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator


def generate_datetime(
    rng: Generator,
    date_format: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
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

    :return: list of generated datetime
    :rtype: list[str, datetime]
    """
    if start_date is None:
        start_date: pd.Timestamp = pd.Timestamp(1920, 1, 1)
    if end_date is None:
        end_date: pd.Timestamp = pd.Timestamp(2049, 12, 31)
    t = rng.random()
    ptime = start_date + t * (end_date - start_date)
    date_string = ptime.strftime(date_format)
    return [date_string, datetime.strptime(date_string, date_format)]


def random_datetimes(
    rng: Generator,
    format: Optional[List[str]] = None,
    min: pd.Timestamp = None,
    max: pd.Timestamp = None,
    num_rows: int = 1,
) -> np.array:
    """
    Generate datetime given the random_state, format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param format: the format that the generated datatime will follow,
        defaults to None
    :type format: str, None, optional
    :param min: the earliest date that datetimes can be generated at,
        defaults to pd.Timestamp(1920, 1, 1)
    :type min: pd.Timestamp, None, optional
    :param max: the latest date that datetimes can be generated at,
        defaults to pd.Timestamp(2049, 12, 31)
    :type max: pd.Timestamp, None, optional

    :return: array of generated datetimes
    :rtype: numpy array
    """
    date_list = [""] * num_rows
    if not format:
        format = ["%B %d %Y %H:%M:%S"]

    if not isinstance(format, list):
        raise Exception("format must be of type `list`")

    for i in range(num_rows):
        date_format = rng.choice(format)
        datetime = generate_datetime(
            rng, date_format=date_format, start_date=min, end_date=max
        )
        date_list[i] = datetime

    return np.array(date_list)
