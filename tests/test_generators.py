import os
import unittest
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd

from synthetic_data import Generator
from synthetic_data.generators import (
    GraphGenerator,
    TabularGenerator,
    UnstructuredGenerator,
)

test_dir = os.path.dirname(os.path.realpath(__file__))


class TestTabularGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile_options = dp.ProfilerOptions()
        cls.profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )
        dp.set_seed(0)

        # create dataset and profile for tabular
        cls.tab_data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        cls.tab_profile = dp.Profiler(
            cls.tab_data, profiler_type="structured", options=cls.profile_options
        )
        cls.uncorrelated_generator = Generator(
            profile=cls.tab_profile, is_correlated=False, seed=42
        )

    def test_synthesize_tabular(self):
        generator = Generator(profile=self.tab_profile, seed=42)
        self.assertIsInstance(generator, TabularGenerator)
        synthetic_data = generator.synthesize(100)
        self.assertEqual(len(synthetic_data), 100)

        generator = Generator(data=self.tab_data, seed=42)
        synthetic_data_2 = generator.synthesize(100)
        self.assertEqual(len(synthetic_data_2), 100)

        # asserts that both  methods create the same results
        # if this ever fails may need to start setting seeds
        np.testing.assert_array_equal(synthetic_data, synthetic_data_2)

    def test_is_not_correlated(self):
        self.assertFalse(self.uncorrelated_generator.is_correlated)
        synthetic_data = self.uncorrelated_generator.synthesize(100)
        self.assertEqual(len(synthetic_data), 100)

        # test is not correlated when starting from data
        generator = Generator(
            data=self.tab_data.data[:10], is_correlated=False, seed=42
        )
        self.assertFalse(self.uncorrelated_generator.is_correlated)
        synthetic_data = self.uncorrelated_generator.synthesize(100)
        self.assertEqual(len(synthetic_data), 100)

    @mock.patch("synthetic_data.generators.make_data_from_report")
    def test_synthesize_correlated_method(self, mock_make_data):
        profile = dp.Profiler(
            data=self.tab_data,
            options=self.profile_options,
            samples_per_update=len(self.tab_data),
        )
        instance = TabularGenerator(profile)
        instance.method = "correlated"
        instance.synthesize(num_samples=10)
        mock_make_data.assert_called_once()

    @mock.patch("synthetic_data.generators.generate_dataset")
    def test_uncorrelated_synthesize_columns_to_generate(self, mock_generate_dataset):
        data = dp.Data(os.path.join(test_dir, "data/tabular.csv"))
        profile = dp.Profiler(
            data=data,
            options=self.profile_options,
            samples_per_update=len(data),
        )
        generator = TabularGenerator(profile=profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        expected_columns_to_generate = [
            {
                "generator": "datetime",
                "name": "dat",
                "date_format_list": ["%m/%d/%y %H:%M"],
                "start_date": pd.Timestamp("2013-03-03 21:53:00"),
                "end_date": pd.Timestamp("2013-03-03 22:21:00"),
                "order": "random",
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": [
                    "o",
                    "h",
                    "n",
                    "z",
                    "k",
                    "l",
                    "t",
                    "c",
                    "-",
                    "i",
                    "a",
                    "g",
                    "p",
                    "r",
                    "s",
                    "u",
                    "y",
                    "e",
                ],
                "str_len_min": 10.0,
                "str_len_max": 17.0,
                "order": "random",
            },
            {
                "generator": "integer",
                "name": "int",
                "min_value": 1007884304.0,
                "max_value": 3730416887.0,
                "order": "random",
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": ["D", "P", "U", "T", "C"],
                "str_len_min": 3.0,
                "str_len_max": 3.0,
                "order": "random",
            },
            {
                "generator": "integer",
                "name": "int",
                "min_value": 2489.0,
                "max_value": 56577.0,
                "order": "random",
            },
            {
                "generator": "integer",
                "name": "int",
                "min_value": 22.0,
                "max_value": 5060.0,
                "order": "random",
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": ["6", "2", "7", "5", "9", "3", "0", ".", "4", "1", "8"],
                "str_len_min": 11.0,
                "str_len_max": 15.0,
                "order": "random",
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": [
                    "o",
                    "b",
                    "x",
                    "S",
                    "h",
                    "n",
                    "W",
                    "l",
                    "I",
                    "c",
                    "T",
                    "H",
                    "t",
                    "i",
                    "a",
                    "J",
                    "g",
                    "p",
                    " ",
                    "s",
                    "r",
                    "L",
                    "u",
                    "O",
                    "e",
                ],
                "str_len_min": 5.0,
                "str_len_max": 13.0,
                "order": "random",
            },
            {
                "generator": "text",
                "name": "txt",
                "chars": ["6", "2", "R", "W", "5", "3", "A", "4", "1", "L", "I", "O"],
                "str_len_min": 2.0,
                "str_len_max": 2.0,
                "order": "random",
            },
            {
                "generator": "integer",
                "name": "int",
                "min_value": 60661.0,
                "max_value": 98168.0,
                "order": "descending",
            },
            {
                "generator": "float",
                "name": "flo",
                "min_value": 25.0392,
                "max_value": 51.0,
                "sig_figs": 6,
                "order": "random",
            },
            {
                "generator": "float",
                "name": "flo",
                "min_value": -122.9117,
                "max_value": 127.02,
                "sig_figs": 7,
                "order": "random",
            },
            {
                "generator": "integer",
                "name": "int",
                "min_value": 664.0,
                "max_value": 9464.0,
                "order": "random",
            },
        ]
        generator.synthesize(20)

        # cannot call this because the report will generate the char lists in different orders thus making this fail
        # (even though the elements in the lists are the same)
        # mock_generate_dataset.assert_called_once_with(generator.rng, expected_columns_to_generate, 20)

        mock_generate_dataset.assert_called_once()
        for i in range(len(expected_columns_to_generate)):
            for key in generator.col_data[i].keys():
                if isinstance(generator.col_data[i][key], list):
                    self.assertTrue(
                        set(generator.col_data[i][key]).issubset(
                            expected_columns_to_generate[i][key]
                        )
                    )
                else:
                    self.assertEqual(
                        generator.col_data[i][key], expected_columns_to_generate[i][key]
                    )

    def test_uncorrelated_synthesize_output(self):
        data = dp.Data(os.path.join(test_dir, "data/tabular.csv"))
        profile = dp.Profiler(
            data=data,
            options=self.profile_options,
            samples_per_update=len(data),
        )
        generator = TabularGenerator(profile=profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        actual_synthetic_data = generator.synthesize(20)

        self.assertEqual(len(actual_synthetic_data), 20)
        self.assertIsInstance(actual_synthetic_data, pd.DataFrame)
        np.testing.assert_array_equal(
            actual_synthetic_data.columns.values,
            np.array(["dat", "txt", "int", "flo"], dtype="object"),
        )

        # for column in actual_synthetic_data_columns:
        #     print(actual_synthetic_data.loc[:,column].values, expected.loc[:,column].values)
        #     self.assertTrue(set(actual_synthetic_data.loc[:,column].values).issubset(expected.loc[:,column].values))

        # for i in range(len(actual_synthetic_data)):
        #     print(actual_synthetic_data[i], "YTEESH", expected[i])
        # self.assertTrue(set(actual_synthetic_data[i]).issubset(expected[i]))
        #     self.assertTrue(set(generator.col_data[i]).issubset(expected_columns_to_generate[i]))
        # np.testing.assert_array_equal(actual_synthetic_data.values, expected_dataframe.values)

        # expected_data = {
        #         'dat': ['03/03/13 22:14', '03/03/13 22:05', '03/03/13 22:17', '03/03/13 22:12', '03/03/13 21:55',
        #                     '03/03/13 22:20', '03/03/13 22:14', '03/03/13 22:15', '03/03/13 21:56', '03/03/13 22:05',
        #                     '03/03/13 22:03', '03/03/13 22:18', '03/03/13 22:11', '03/03/13 22:16', '03/03/13 22:05',
        #                     '03/03/13 21:59', '03/03/13 22:08', '03/03/13 21:54', '03/03/13 22:16', '03/03/13 22:10'],
        #         'txt': ['OI', 'I2', 'II', '4R', '5A', '64', '6L', '34', '61', '25', 'O3', 'W2', 'RO', '5L', '41', 'OW', '5O', 'OA', 'W2', '51'],
        #         'int': [8608, 793, 4506, 7943, 6878, 8201, 4317, 3939, 6904, 2015, 2770, 5951, 6546, 1717, 5137, 3875, 5079, 9098, 1338, 9424],
        #         'flo': [-32.771999, 5.238091, 61.214405, 98.628480, 107.289691, 2.962133, 7.121544, 77.001272, -44.320504, 86.376697,
        #                 0.589962, -93.955432, -104.901835, 87.529095, -109.023516, -52.778007, -39.402011, -79.674904, -44.459797, 62.710716]
        #         }

        # expected = pd.DataFrame(expected_data)

        # actual_synthetic_data_columns = actual_synthetic_data.columns

        # print(actual_synthetic_data, "columns", actual_synthetic_data_columns)

    def test_invalid_config(self):
        with self.assertRaises(
            ValueError,
            msg="Warning: profile doesn't match user setting.",
        ):
            Generator(config=1)

    def test_no_profile_or_data(self):
        with self.assertRaisesRegex(
            ValueError,
            "No profile object or dataset was passed in kwargs. "
            "If you want to generate synthetic data from a "
            "profile, pass in a profile object through the "
            'key "profile" or data through the key "data" in kwargs.',
        ):
            Generator()

    def test_invalid_profile(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Profile object is invalid. The supported profile types are: \[.+\].",
        ):
            Generator(profile=1)

    def test_invalid_data(self):
        with self.assertRaisesRegex(
            ValueError, "data is not in an acceptable format for profiling."
        ):
            Generator(data=1)
