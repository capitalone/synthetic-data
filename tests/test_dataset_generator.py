import unittest
from collections import OrderedDict
from datetime import datetime
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data import dataset_generator as dg
from synthetic_data.distinct_generators import datetime_generator as dategen


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.dataset_length = 10
        self.columns_to_gen = [
            {"generator": "integer", "name": "int", "min_value": 4, "max_value": 88},
            {
                "generator": "datetime",
                "name": "dat",
                "date_format_list": ["%Y-%m-%d"],
                "start_date": pd.Timestamp(2001, 12, 22),
                "end_date": pd.Timestamp(2022, 12, 22),
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": ["0", "1"],
                "str_len_min": 2,
                "str_len_max": 5,
            },
            {
                "generator": "categorical",
                "name": "cat",
                "categories": ["X", "Y", "Z"],
                "probabilities": [0.1, 0.5, 0.4],
            },
            {
                "generator": "float",
                "name": "flo",
                "min_value": 3,
                "max_value": 10,
                "sig_figs": 3,
            },
        ]

    def test_generate_dataset_with_invalid_generator(self):
        columns_to_gen = [{"generator": "non existent generator"}]
        with self.assertRaisesRegex(
            ValueError, "generator: non existent generator is not a valid generator."
        ):
            dg.generate_dataset(
                self.rng,
                columns_to_generate=columns_to_gen,
                dataset_length=self.dataset_length,
            )

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_none_columns(self, mock_warning):
        empty_dataframe = pd.DataFrame()
        df = dg.generate_dataset(self.rng, None, self.dataset_length)
        mock_warning.assert_called_once_with(
            "columns_to_generate is empty, empty dataframe will be returned."
        )
        self.assertEqual(empty_dataframe.empty, df.empty)

    def test_generate_custom_dataset(self):
        expected_data = [
            np.array([62, 23, 70, 30, 21, 70, 57, 60, 87, 36]),
            np.array(
                [
                    "2008-12-17",
                    "2014-07-16",
                    "2005-11-23",
                    "2016-02-07",
                    "2021-10-01",
                    "2007-03-10",
                    "2021-11-24",
                    "2015-12-26",
                    "2003-12-27",
                    "2011-04-02",
                ]
            ),
            np.array(
                ["10", "0001", "0100", "10", "000", "100", "00", "01", "1110", "1111"]
            ),
            np.array(["Z", "Y", "Z", "Y", "Y", "Y", "Z", "Y", "Z", "Y"]),
            np.array(
                [5.379, 4.812, 5.488, 3.035, 7.4, 4.977, 3.477, 7.318, 4.234, 5.131]
            ),
        ]
        expected_df = pd.DataFrame.from_dict(
            dict(zip(["int", "dat", "txt", "cat", "flo"], expected_data))
        )
        df = dg.generate_dataset(
            self.rng,
            columns_to_generate=self.columns_to_gen,
            dataset_length=self.dataset_length,
        )
        self.assertTrue(df.equals(expected_df))


class TestGetOrderedColumn(unittest.TestCase):
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

        np.testing.assert_array_equal(output_data, ordered_data)

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

        np.testing.assert_array_equal(output_data, ordered_data)

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

        np.testing.assert_array_equal(output_data, ordered_data)

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

        np.testing.assert_array_equal(output_data, ordered_data)

    def test_get_ordered_column(self):

        data = OrderedDict(
            {
                "int": np.array([5, 4, 3, 2, 1]),
                "float": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                "string": np.array(["abcde", "bcdea", "cdeab", "deabc", "eabcd"]),
                "categorical": np.array(["E", "D", "C", "B", "A"]),
                "datetime": np.array(
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
                ),
            }
        )

        ordered_data = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array(["abcde", "bcdea", "cdeab", "deabc", "eabcd"]),
            np.array(["A", "B", "C", "D", "E"]),
            np.array(
                [
                    "October 02 2006 22:34:32",
                    "August 19 2008 16:53:49",
                    "March 13 2010 17:18:44",
                    "March 11 2016 15:15:39",
                    "September 27 2018 18:24:03",
                ]
            ),
        ]
        ordered_data = np.array(ordered_data, dtype=object)

        output_data = []
        for data_type in data.keys():
            output_data.append(dg.get_ordered_column(data[data_type], data_type))
        output_data = np.array(output_data)

        np.testing.assert_array_equal(output_data, ordered_data)
