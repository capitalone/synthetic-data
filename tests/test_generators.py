import os
import unittest
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd

from synthetic_data import Generator
from synthetic_data.generators import TabularGenerator

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
        cls.tab_csv_data = dp.Data(os.path.join(test_dir, "data/tabular.csv"))
        cls.profile = dp.Profiler(
            data=cls.tab_csv_data,
            options=cls.profile_options,
            samples_per_update=len(cls.tab_csv_data),
        )

    @mock.patch("synthetic_data.generators.make_data_from_report")
    def test_synthesize_correlated_method(self, mock_make_data):
        tab_data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        profile = dp.Profiler(
            data=tab_data,
            options=self.profile_options,
            samples_per_update=len(tab_data),
        )
        correlated_tabular_generator = TabularGenerator(profile)
        self.assertTrue(correlated_tabular_generator.is_correlated)
        correlated_tabular_generator.synthesize(num_samples=10)
        mock_make_data.assert_called_once()

    @mock.patch("synthetic_data.generators.generate_dataset")
    def test_uncorrelated_synthesize_columns_to_generate(self, mock_generate_dataset):

        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
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
                "generator": "categorical",
                "name": "cat",
                "categories": [
                    "groucho-oregon",
                    "groucho-singapore",
                    "groucho-tokyo",
                    "groucho-sa",
                    "zeppo-norcal",
                    "groucho-norcal",
                    "groucho-us-east",
                    "groucho-eu",
                ],
                "probabilities": [
                    0.3333333333333333,
                    0.2,
                    0.13333333333333333,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                ],
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
                "generator": "categorical",
                "name": "cat",
                "categories": ["TCP", "UDP"],
                "probabilities": [0.8823529411764706, 0.11764705882352941],
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
                "chars": ["4", "3", "0", "7", ".", "5", "1", "8", "2", "6", "9"],
                "str_len_min": 11.0,
                "str_len_max": 15.0,
                "order": "random",
            },
            {
                "generator": "categorical",
                "name": "cat",
                "categories": [
                    "Seoul",
                    "Jiangxi Sheng",
                    "Taipei",
                    "Oregon",
                    "Illinois",
                    "Henan Sheng",
                    "Sichuan Sheng",
                    "Hebei",
                    "Liaoning",
                    "Washington",
                ],
                "probabilities": [
                    0.3333333333333333,
                    0.13333333333333333,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                    0.06666666666666667,
                ],
                "order": "random",
            },
            {
                "generator": "categorical",
                "name": "cat",
                "categories": ["11", "36", "OR", "IL", "41", "51", "13", "21", "WA"],
                "probabilities": [
                    0.35714285714285715,
                    0.14285714285714285,
                    0.07142857142857142,
                    0.07142857142857142,
                    0.07142857142857142,
                    0.07142857142857142,
                    0.07142857142857142,
                    0.07142857142857142,
                    0.07142857142857142,
                ],
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
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        actual_synthetic_data = generator.synthesize(20)

        self.assertEqual(len(actual_synthetic_data), 20)
        self.assertIsInstance(actual_synthetic_data, pd.DataFrame)

        np.testing.assert_array_equal(
            actual_synthetic_data.columns.values,
            np.array(["dat", "cat", "int", "txt", "flo"], dtype="object"),
        )
