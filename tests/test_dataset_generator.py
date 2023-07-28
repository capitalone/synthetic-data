import unittest
from unittest import mock

import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data.dataset_generator import generate_dataset


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
            generate_dataset(
                self.rng,
                columns_to_generate=columns_to_gen,
                dataset_length=self.dataset_length,
            )

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_none_columns(self, mock_warning):
        empty_dataframe = pd.DataFrame()
        df = generate_dataset(self.rng, None, self.dataset_length)
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
            dict(zip(["int", "dat", "str", "cat", "flo"], expected_data))
        )
        df = generate_dataset(
            self.rng,
            columns_to_generate=self.columns_to_gen,
            dataset_length=self.dataset_length,
        )
