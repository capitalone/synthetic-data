import numpy as np
import pandas as pd
from numpy.random import Generator
from typing import Optional


def generate_datetime(
    rng: Generator,
    date_format: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Generate datetime given the random_state, date_format, and start/end dates.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param date_format: the format that the generated datatime will follow,
        defaults to None
    :type date_format: str, None, optional
    :param start_date: the earliest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional
    :param start_date: the latest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional

    :return: generated datetime
    :rtype: str
    """
    if not start_date:
        # 100 years in past
        start_date = pd.Timestamp(1920, 1, 1)
    if not end_date:
        # protection of 30 years in future
        end_date = pd.Timestamp(2049, 12, 31)
    t = rng.random()
    ptime = start_date + t * (end_date - start_date)

    return ptime.strftime(date_format)


def random_datetimes(
    rng: Generator,
    date_format_list: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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
        defaults to None
    :type start_date: pd.Timestamp, None, optional
    :param start_date: the latest date that datetimes can be generated at,
        defaults to None
    :type start_date: pd.Timestamp, None, optional

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





