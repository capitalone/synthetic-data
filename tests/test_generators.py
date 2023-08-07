import os
import unittest
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data import Generator
from synthetic_data.distinct_generators import datetime_generator
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
        cls.data = dp.Data(os.path.join(test_dir, "data/tabular.csv"))
        cls.profile = dp.Profiler(
            data=cls.data,
            options=cls.profile_options,
            samples_per_update=len(cls.data),
        )

    def test_synthesize_tabular(self):
        tab_data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        tab_profile = dp.Profiler(
            tab_data, profiler_type="structured", options=self.profile_options
        )
        generator = Generator(profile=tab_profile, seed=42)
        self.assertIsInstance(generator, TabularGenerator)
        synthetic_data = generator.synthesize(100)
        self.assertEqual(len(synthetic_data), 100)

        generator = Generator(data=tab_data, seed=42)
        synthetic_data_2 = generator.synthesize(100)
        self.assertEqual(len(synthetic_data_2), 100)

        # asserts that both  methods create the same results
        # if this ever fails may need to start setting seeds
        np.testing.assert_array_equal(synthetic_data, synthetic_data_2)

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

    @mock.patch(
        "synthetic_data.generators.TabularGenerator.generate_uncorrelated_column_data"
    )
    @mock.patch("synthetic_data.generators.generate_dataset")
    def test_uncorrelated_synthesize_columns_to_generate(
        self, mock_generate_dataset, mock_col_data
    ):

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
            for key in mock_col_data[i].keys():
                if isinstance(mock_col_data[i][key], list):
                    self.assertTrue(
                        set(mock_col_data[i][key]).issubset(
                            expected_columns_to_generate[i][key]
                        )
                    )
                else:
                    self.assertEqual(
                        mock_col_data[i][key], expected_columns_to_generate[i][key]
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

    def test_generate_uncorrelated_column_data(self):
        pass

    def test_get_ordered_column(self):
        pass


@mock.patch("generate_uncorrelated_column_data.TabularGenerator", spec=TabularGenerator)
class TestDatasetGenerator(unittest.TestCase):
    @staticmethod
    def setup_tabular_generator_mock(mock_generator):
        mock_DataLabeler = mock_generator.return_value

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

    def test_get_ordered_column_integration(self, mocked_tabular_generator):
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
        np.testing.assert_array_equal(actual_df.values, expected_df.values)

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
        expected_dataframe = pd.DataFrame()
        actual_df = dataset_generator.generate_dataset(
            self.rng, None, self.dataset_length
        )
        mock_warning.assert_called_once_with(
            "columns_to_generate is empty, empty dataframe will be returned."
        )
        self.assertEqual(expected_dataframe.empty, actual_df.empty)

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
                "October 02 2006 22:34:32",
                "August 19 2008 16:53:49",
                "March 13 2010 17:18:44",
                "March 11 2016 15:15:39",
                "September 27 2018 18:24:03",
            ]
        )

        actual = dataset_generator.get_ordered_column(data, "datetime", "ascending")

        np.testing.assert_array_equal(actual, expected)

    def test_get_ordered_column_datetime_descending(self):
        data = datetime_generator.random_datetimes(
            rng=self.rng, start_date=self.start_date, end_date=self.end_date, num_rows=5
        )

        expected = np.array(
            [
                "September 27 2018 18:24:03",
                "March 11 2016 15:15:39",
                "March 13 2010 17:18:44",
                "August 19 2008 16:53:49",
                "October 02 2006 22:34:32",
            ]
        )

        actual = dataset_generator.get_ordered_column(data, "datetime", "descending")

        np.testing.assert_array_equal(actual, expected)

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
                "2018 09 27",
                "2016 03 11",
                "2010 03 13",
                "2008 08 19",
                "2006 10 02",
            ]
        )

        actual = dataset_generator.get_ordered_column(data, "datetime", "descending")

        np.testing.assert_array_equal(actual, expected)
