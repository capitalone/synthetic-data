import unittest
from collections import OrderedDict
from datetime import datetime
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data import dataset_generator
from synthetic_data.distinct_generators import datetime_generator


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

    def test_get_ordered_column_integration(self):
        columns_to_gen = [
            {
                "generator": "integer",
                "name": "int",
                "min_value": 4,
                "max_value": 88,
                "order": "ascending",
            },
            {
                "generator": "datetime",
                "name": "dat",
                "date_format_list": ["%Y-%m-%d"],
                "start_date": pd.Timestamp(2001, 12, 22),
                "end_date": pd.Timestamp(2022, 12, 22),
                "order": "ascending",
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": ["0", "1"],
                "str_len_min": 2,
                "str_len_max": 5,
                "order": "ascending",
            },
            {
                "generator": "categorical",
                "name": "cat",
                "categories": ["X", "Y", "Z"],
                "probabilities": [0.1, 0.5, 0.4],
                "order": "ascending",
            },
            {
                "generator": "float",
                "name": "flo",
                "min_value": 3,
                "max_value": 10,
                "sig_figs": 3,
                "order": "ascending",
            },
        ]
        expected_data = [
            np.array([21, 23, 30, 36, 57, 60, 62, 70, 70, 87]),
            np.array(
                [
                    "2003-12-27",
                    "2005-11-23",
                    "2007-03-10",
                    "2008-12-17",
                    "2011-04-02",
                    "2014-07-16",
                    "2015-12-26",
                    "2016-02-07",
                    "2021-10-01",
                    "2021-11-24",
                ]
            ),
            np.array(
                ["00", "000", "0001", "01", "0100", "10", "10", "100", "1110", "1111"]
            ),
            np.array(["Y", "Y", "Y", "Y", "Y", "Y", "Z", "Z", "Z", "Z"]),
            np.array(
                [3.035, 3.477, 4.234, 4.812, 4.977, 5.131, 5.379, 5.488, 7.318, 7.4]
            ),
        ]
        expected_df = pd.DataFrame.from_dict(
            dict(zip(["int", "dat", "txt", "cat", "flo"], expected_data))
        )
        actual_df = dataset_generator.generate_dataset(
            self.rng,
            columns_to_generate=columns_to_gen,
            dataset_length=self.dataset_length,
        )

        self.assertTrue(actual_df.equals(expected_df))

    def test_generate_dataset_with_invalid_generator(self):
        columns_to_gen = [{"generator": "non existent generator"}]
        with self.assertRaisesRegex(
            ValueError, "generator: non existent generator is not a valid generator."
        ):
            dataset_generator.generate_dataset(
                self.rng,
                columns_to_generate=columns_to_gen,
                dataset_length=self.dataset_length,
            )

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_invalid_sorting_type(self, mock_warning):
        columns_to_gen = [
            {
                "generator": "integer",
                "name": "int",
                "min_value": 4,
                "max_value": 88,
                "order": "random",
            }
        ]
        unsupported_sort_types = ["cheese", "random"]

        for type in unsupported_sort_types:
            columns_to_gen[0]["order"] = type
            dataset_generator.generate_dataset(
                self.rng,
                columns_to_generate=columns_to_gen,
                dataset_length=self.dataset_length,
            )
            mock_warning.assert_called_with(
                f"""{columns_to_gen[0]["name"]} is passed with sorting type of {columns_to_gen[0]["order"]}.
                Ascending and descending are the only supported options.
                No sorting action will be taken."""
            )
        self.assertEqual(mock_warning.call_count, 2)

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_valid_sorting_type(self, mock_warning):
        columns_to_gen = [
            {
                "generator": "integer",
                "name": "int",
                "min_value": 4,
                "max_value": 88,
                "order": "ascending",
            }
        ]
        supported_sort_types = ["ascending", "descending", None]

        for type in supported_sort_types:
            columns_to_gen[0]["order"] = type
            dataset_generator.generate_dataset(
                self.rng,
                columns_to_generate=columns_to_gen,
                dataset_length=self.dataset_length,
            )

        self.assertEqual(mock_warning.call_count, 0)

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_none_columns(self, mock_warning):
        empty_dataframe = pd.DataFrame()
        df = dataset_generator.generate_dataset(self.rng, None, self.dataset_length)
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
        actual_df = dataset_generator.generate_dataset(
            self.rng,
            columns_to_generate=self.columns_to_gen,
            dataset_length=self.dataset_length,
        )
        self.assertTrue(actual_df.equals(expected_df))


class TestGetOrderedColumn(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%B %d %Y %H:%M:%S"]

    def test_get_ordered_column_datetime_ascending(self):
        data = datetime_generator.random_datetimes(
            rng=self.rng, start_date=self.start_date, end_date=self.end_date, num_rows=5
        )

        expected = np.array(
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

        actual = dataset_generator.get_ordered_column(data, "datetime", "ascending")

        np.testing.assert_array_equal(actual, expected[:, 0])

    def test_get_ordered_column_datetime_descending(self):
        data = datetime_generator.random_datetimes(
            rng=self.rng, start_date=self.start_date, end_date=self.end_date, num_rows=5
        )

        expected = np.array(
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

        actual = dataset_generator.get_ordered_column(data, "datetime", "descending")

        np.testing.assert_array_equal(actual, expected[:, 0])

    def test_get_ordered_column_custom_datetime_ascending(self):
        custom_date_format = ["%Y %m %d", "%B %d %Y %H:%M:%S"]
        data = datetime_generator.random_datetimes(
            rng=self.rng,
            date_format_list=custom_date_format,
            start_date=self.start_date,
            end_date=self.end_date,
            num_rows=5,
        )

        expected = np.array(
            [
                "November 25 2005 02:50:42",
                "August 19 2008 16:53:49",
                "December 21 2008 00:15:47",
                "March 13 2010 17:18:44",
                "2018 09 27",
            ]
        )

        actual = dataset_generator.get_ordered_column(data, "datetime", "ascending")

        np.testing.assert_array_equal(actual, expected)

    def test_get_ordered_column_custom_datetime_descending(self):
        custom_date_format = ["%Y %m %d"]
        data = datetime_generator.random_datetimes(
            rng=self.rng,
            date_format_list=custom_date_format,
            start_date=self.start_date,
            end_date=self.end_date,
            num_rows=5,
        )

        expected = np.array(
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

        actual = dataset_generator.get_ordered_column(data, "datetime", "descending")

        np.testing.assert_array_equal(actual, expected[:, 0])
