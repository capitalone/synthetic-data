"""Contains tests for dataset_generator"""

import unittest
from collections import OrderedDict
from datetime import datetime

import dataprofiler as dp
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data import dataset_generator as dg
from synthetic_data.distinct_generators import datetime_generator as dategen


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%B %d %Y %H:%M:%S"]

    def test_get_ordered_column_datetime_ascending(self):
        data = dategen.random_datetimes(
            rng=self.rng, start_date=self.start_date, end_date=self.end_date, num_rows=5
        )

        ordered_data = np.array(
            [
                [
                    "October 02 2006 22:34:32",
                    datetime.strptime(
                        "October 02 2006 22:34:32", self.date_format_list[0]
                    ),
                ],
                [
                    "August 19 2008 16:53:49",
                    datetime.strptime(
                        "August 19 2008 16:53:49", self.date_format_list[0]
                    ),
                ],
                [
                    "March 13 2010 17:18:44",
                    datetime.strptime(
                        "March 13 2010 17:18:44", self.date_format_list[0]
                    ),
                ],
                [
                    "March 11 2016 15:15:39",
                    datetime.strptime(
                        "March 11 2016 15:15:39", self.date_format_list[0]
                    ),
                ],
                [
                    "September 27 2018 18:24:03",
                    datetime.strptime(
                        "September 27 2018 18:24:03", self.date_format_list[0]
                    ),
                ],
            ]
        )

        ordered_data = ordered_data[:, 0]
        output_data = dg.get_ordered_column(data, "datetime", "ascending")

        self.assertTrue(np.array_equal(output_data, ordered_data))

    def test_get_ordered_column_datetime_descending(self):
        data = dategen.random_datetimes(
            rng=self.rng, start_date=self.start_date, end_date=self.end_date, num_rows=5
        )

        ordered_data = np.array(
            [
                [
                    "September 27 2018 18:24:03",
                    datetime.strptime(
                        "September 27 2018 18:24:03", self.date_format_list[0]
                    ),
                ],
                [
                    "March 11 2016 15:15:39",
                    datetime.strptime(
                        "March 11 2016 15:15:39", self.date_format_list[0]
                    ),
                ],
                [
                    "March 13 2010 17:18:44",
                    datetime.strptime(
                        "March 13 2010 17:18:44", self.date_format_list[0]
                    ),
                ],
                [
                    "August 19 2008 16:53:49",
                    datetime.strptime(
                        "August 19 2008 16:53:49", self.date_format_list[0]
                    ),
                ],
                [
                    "October 02 2006 22:34:32",
                    datetime.strptime(
                        "October 02 2006 22:34:32", self.date_format_list[0]
                    ),
                ],
            ]
        )

        ordered_data = ordered_data[:, 0]
        output_data = dg.get_ordered_column(data, "datetime", "descending")

        self.assertTrue(np.array_equal(output_data, ordered_data))

    def test_get_ordered_column_custom_datetime_ascending(self):
        custom_date_format = ["%Y %m %d"]
        data = dategen.random_datetimes(
            rng=self.rng,
            date_format_list=custom_date_format,
            start_date=self.start_date,
            end_date=self.end_date,
            num_rows=5,
        )

        ordered_data = np.array(
            [
                [
                    "2006 10 02",
                    datetime.strptime("2006 10 02", custom_date_format[0]),
                ],
                [
                    "2008 08 19",
                    datetime.strptime("2008 08 19", custom_date_format[0]),
                ],
                [
                    "2010 03 13",
                    datetime.strptime("2010 03 13", custom_date_format[0]),
                ],
                [
                    "2016 03 11",
                    datetime.strptime("2016 03 11", custom_date_format[0]),
                ],
                [
                    "2018 09 27",
                    datetime.strptime("2018 09 27", custom_date_format[0]),
                ],
            ]
        )

        ordered_data = ordered_data[:, 0]
        output_data = dg.get_ordered_column(data, "datetime", "ascending")

        self.assertTrue(np.array_equal(output_data, ordered_data))

    def test_get_ordered_column_custom_datetime_descending(self):
        custom_date_format = ["%Y %m %d"]
        data = dategen.random_datetimes(
            rng=self.rng,
            date_format_list=custom_date_format,
            start_date=self.start_date,
            end_date=self.end_date,
            num_rows=5,
        )

        ordered_data = np.array(
            [
                [
                    "2018 09 27",
                    datetime.strptime("2018 09 27", custom_date_format[0]),
                ],
                [
                    "2016 03 11",
                    datetime.strptime("2016 03 11", custom_date_format[0]),
                ],
                [
                    "2010 03 13",
                    datetime.strptime("2010 03 13", custom_date_format[0]),
                ],
                [
                    "2008 08 19",
                    datetime.strptime("2008 08 19", custom_date_format[0]),
                ],
                [
                    "2006 10 02",
                    datetime.strptime("2006 10 02", custom_date_format[0]),
                ],
            ]
        )

        ordered_data = ordered_data[:, 0]
        output_data = dg.get_ordered_column(data, "datetime", "descending")

        self.assertTrue(np.array_equal(output_data, ordered_data))

    def test_get_ordered_column(self):

        data = OrderedDict(
            {
                "int": np.array([5, 4, 3, 2, 1]),
                "float": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                "string": np.array(["abcde", "bcdea", "cdeab", "deabc", "eabcd"]),
                "categorical": np.array(["E", "D", "C", "B", "A"]),
            }
        )
        ordered_data = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array(["abcde", "bcdea", "cdeab", "deabc", "eabcd"]),
            np.array(["A", "B", "C", "D", "E"]),
        ]
        ordered_data = np.asarray(ordered_data)

        output_data = []
        for data_type in data.keys():
            output_data.append(dg.get_ordered_column(data[data_type], data_type))
        output_data = np.asarray(output_data)

        self.assertTrue(np.array_equal(output_data, ordered_data))
