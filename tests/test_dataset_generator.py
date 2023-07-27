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
                "str_len_min": 300,
                "str_len_max": 301,
            },
            {
                "generator": "string",
                "name": "str",
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
                path=None,
            )

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_none_columns(self, mock_warning):
        empty_dataframe = pd.DataFrame()
        df = dg.generate_dataset(self.rng, None, self.dataset_length, None)
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
                [
                    "010100011010001000001100000001111101111111110110001011101100001010000100101100001010101000100010101010101011000110010110110000011101001011101110011010000011111010001100000000110001100100000000000111001110110110101010110010010000101101110001000101010110101101001011101011110110100110100000111101010101",
                    "000111111101111101100010110011100000110010011100001110011100001010011001100010111000101101001010101010101111000010110111110111001000110100010100001111011111011110000100111000000100100011010110111001010011001011100110110010100000000001011111110110001001100001010100100011001110011000100101000011011111",
                    "111100000010101010101001100110110011000010100100101110111110001001010001000010000100110011000101000000010011011001101011011100101000100001001011011110111010010100001110001101101110110011010110111010000110000000011000100011111101001110010011110000001011100100100111010001011000101110011100110111001001",
                    "101010000111110001001111100000101111000100000111001100001001001110101111111010111011011100010101100001001010111010010110110100101010010101000100001001110000001010111100010100101110011100011100111000110110011110111110001000111011110010000111100000110010001110101100101110111111000011001100111111000011",
                    "011110101101010111010100100001101000000101000001000100110011100011000100011111101100100101111000111000101111101101000010100011110010010111110011011010000000001111111101101111110001110110110100010111111001000000101101101101000000001000001001101000010001011111011001111011101011011000100010001001010111",
                    "111100110001011101000101011000110001001100101100101100000110010011001011010110001010111010010111111111100011101110010011001101101000000011000100101001110010110101010001011111001110110111001111111110001000110110101000010001111001100111100110110101101100110100010011011011100110010110100001010100001000",
                    "111000010000010000001100001101000011001001010110100100111000101100101110110111000010011101010110101011111101011011110110111100111011100110011011111011111001110001101000101001100000101010000010111100101110100001000011011011000001101010000010000001110111010010001101011100100101101110001111101001000111",
                    "010001100011001001101011000010000111011010011000110000111110000000000000101101111011101000011001010111110100000010100000000100110001001000110010010011110001111011101111101001011111000000000011000110100000011010111001000001000110111011111000011111010011011000111100000001111100011000011111000000001000",
                    "000011011010101010010011011001001111001000000001111110111101010000011101101000000110111001000101110001011100101110001000100001110101001011110110101000110101000100100010011011000010000111101001111000000011011000100011010100001111111111110010011101110010101010010010110011110011001010100000111111110001",
                    "101111001110001101001010001110100100010010001011110100110000100001000100010100110001110010000111100100010010011010011111000101001110111001111000011011011011111100101010110111100110000111010000100001100111111111010001011100010010100111101010010100011011110110101110111111111000000100110001110011010000",
                ]
            ),
            np.array(
                ["01", "010", "110", "0111", "1001", "001", "0100", "0111", "11", "101"]
            ),
            np.array(["Z", "Y", "Z", "Y", "Y", "Z", "Y", "Y", "Y", "X"]),
            np.array(
                [7.919, 4.878, 9.382, 4.071, 6.537, 8.611, 7.977, 3.159, 8.304, 9.674]
            ),
        ]
        expected_df = pd.DataFrame.from_dict(
            dict(zip(["int", "dat", "txt", "str", "cat", "flo"], expected_data))
        )
        df = dg.generate_dataset(
            self.rng,
            columns_to_generate=self.columns_to_gen,
            dataset_length=self.dataset_length,
            path=None,
        )
        self.assertTrue(df.equals(expected_df))

    @mock.patch("synthetic_data.dataset_generator.pd.DataFrame.to_csv")
    def test_path_to_csv(self, to_csv):
        """
        Ensure csv creation is triggered at the appropiate time.

        :param to_csv: mock of Pandas to_csv()
        :type to_csv: func
        """
        to_csv.return_value = "assume Pandas to_csv for a dataframe runs correctly"
        path = "testing_path"
        dg.generate_dataset(
            self.rng,
            columns_to_generate=self.columns_to_gen,
            dataset_length=4,
            path=path,
        )
        to_csv.assert_called_once_with(path, index=False, encoding="utf-8")


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
