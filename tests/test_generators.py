import os
import unittest
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd

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

    def test_synthesize_uncorrelated_output(self):
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        actual_synthetic_data = generator.synthesize(20)

        self.assertEqual(len(actual_synthetic_data), 20)
        self.assertIsInstance(actual_synthetic_data, pd.DataFrame)

        np.testing.assert_array_equal(
            actual_synthetic_data.columns.values,
            np.array(
                [
                    "datetime",
                    "host",
                    "src",
                    "proto",
                    "type",
                    "srcport",
                    "destport",
                    "srcip",
                    "locale",
                    "localeabbr",
                    "postalcode",
                    "latitude",
                    "longitude",
                    "owner",
                    "comment",
                    "int_col",
                ],
                dtype="object",
            ),
        )


class TestGenerateUncorrelatedColumnData(unittest.TestCase):
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
        cls.dataset_length = 10
        cls.rng = np.random.Generator(np.random.PCG64(12345))
        cls.columns_to_gen = [
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

    @mock.patch("synthetic_data.generators.random_integers")
    @mock.patch("synthetic_data.generators.random_floats")
    @mock.patch("synthetic_data.generators.random_categorical")
    @mock.patch("synthetic_data.generators.random_datetimes")
    @mock.patch("synthetic_data.generators.random_text")
    @mock.patch(
        "synthetic_data.generators.TabularGenerator._generate_uncorrelated_column_data"
    )
    def test_generate_uncorrelated_column_data(
        self,
        mock_generate_uncorrelated_cols,
        mock_random_text,
        mock_random_datetimes,
        mock_random_categorial,
        mock_random_floats,
        mock_random_integers,
    ):
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        expected_calls = [
            # datetime calls
            [
                mock.call(
                    rng=generator.rng,
                    format=["%m/%d/%y %H:%M"],
                    min=pd.Timestamp("2013-03-03 21:53:00"),
                    max=pd.Timestamp("2013-03-03 22:21:00"),
                    num_rows=20,
                )
            ],
            # categorical calls
            [
                mock.call(
                    rng=generator.rng,
                    categories=[
                        "groucho-oregon",
                        "groucho-singapore",
                        "groucho-tokyo",
                        "groucho-sa",
                        "zeppo-norcal",
                        "groucho-norcal",
                        "groucho-us-east",
                        "groucho-eu",
                    ],
                    probabilities=[
                        0.3333333333333333,
                        0.2,
                        0.13333333333333333,
                        0.06666666666666667,
                        0.06666666666666667,
                        0.06666666666666667,
                        0.06666666666666667,
                        0.06666666666666667,
                    ],
                    num_rows=20,
                ),
                mock.call(
                    rng=generator.rng,
                    categories=["TCP", "UDP"],
                    probabilities=[0.8823529411764706, 0.11764705882352941],
                    num_rows=20,
                ),
                mock.call(
                    rng=generator.rng,
                    categories=[
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
                    probabilities=[
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
                    num_rows=20,
                ),
                mock.call(
                    rng=generator.rng,
                    categories=["11", "36", "OR", "IL", "41", "51", "13", "21", "WA"],
                    probabilities=[
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
                    num_rows=20,
                ),
            ],
            # integer calls
            [
                mock.call(
                    rng=generator.rng, min=1007884304.0, max=3730416887.0, num_rows=20
                ),
                mock.call(rng=generator.rng, min=2489.0, max=56577.0, num_rows=20),
                mock.call(rng=generator.rng, min=22.0, max=5060.0, num_rows=20),
                mock.call(rng=generator.rng, min=60661.0, max=98168.0, num_rows=20),
            ],
            # text calls
            [
                mock.call(
                    rng=generator.rng,
                    vocab=["4", "3", "0", "7", ".", "5", "1", "8", "2", "6", "9"],
                    min=11.0,
                    max=15.0,
                    num_rows=20,
                )
            ],
            # float calls
            [
                mock.call(
                    rng=generator.rng,
                    min=25.0392,
                    max=51.0,
                    precision=6,
                    num_rows=20,
                ),
                mock.call(
                    rng=generator.rng,
                    min=-122.9117,
                    max=127.02,
                    precision=7,
                    num_rows=20,
                ),
                mock.call(rng=generator.rng, min=664.0, max=9464.0, num_rows=20),
            ],
        ]

        generator.synthesize(20)

        mock_generate_uncorrelated_cols.assert_called_once()

        random_generators = [
            mock_random_text,
            mock_random_datetimes,
            mock_random_categorial,
            mock_random_floats,
            mock_random_integers,
        ]

        for i, generator in enumerate(random_generators):
            call_args_list = generator.call_args_list
            for j, key in enumerate(call_args_list):
                if isinstance(call_args_list[key], list):
                    self.assertCountEqual(
                        call_args_list[key], expected_calls[i][0][key]
                    )

                else:
                    self.assertEqual(call_args_list[key], expected_calls[j][key])

    @mock.patch("dataprofiler.profilers.StructuredProfiler.report")
    def test_get_ordered_column_integration(self, mock_report):
        mock_report.return_value = {
            "data_stats": [
                {
                    "data_type": "int",
                    "column_name": "test_column_1",
                    "order": "ascending",
                    "statistics": {
                        "min": 1.0,
                        "max": 4.0,
                    },
                },
                {
                    "data_type": "string",
                    "column_name": "test_column_2",
                    "categorical": False,
                    "order": "ascending",
                    "statistics": {
                        "min": 4.0,
                        "max": 5.0,
                        "vocab": ["q", "p", "a", "w", "e", "r", "i", "s", "d", "f"],
                    },
                },
                {
                    "data_type": "string",
                    "column_name": "test_column_3",
                    "categorical": True,
                    "order": "ascending",
                    "statistics": {
                        "min": 10,
                        "max": 13,
                        "categorical_count": {
                            "red": 1,
                            "blue": 2,
                            "yellow": 1,
                            "orange": 3,
                        },
                        "categories": ["blue", "yellow", "red", "orange"],
                    },
                },
                {
                    "data_type": "float",
                    "column_name": "test_column_4",
                    "order": "ascending",
                    "statistics": {"min": 2.11234, "max": 8.0, "precision": {"max": 6}},
                },
                {
                    "data_type": "datetime",
                    "column_name": "test_column_5",
                    "order": "ascending",
                    "statistics": {
                        "format": ["%Y-%m-%d"],
                        "min": "2000-12-09",
                        "max": "2030-04-23",
                    },
                },
                {
                    "column_name": "test_column_6",
                    "data_type": None,
                },
            ]
        }
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)

        expected_array = [
            [1, "arif", "blue", 2.246061, "2003-06-02", None],
            [1, "daips", "blue", 2.628393, "2003-10-08", None],
            [1, "dree", "orange", 2.642511, "2006-02-17", None],
            [1, "drqs", "orange", 2.807119, "2006-11-18", None],
            [1, "dwdaa", "orange", 3.009102, "2008-12-07", None],
            [2, "fswfe", "orange", 3.061853, "2009-12-03", None],
            [2, "fwqe", "orange", 3.677692, "2013-02-24", None],
            [2, "ipdpd", "orange", 3.887541, "2013-08-18", None],
            [3, "pdis", "red", 4.24257, "2014-02-19", None],
            [3, "peii", "red", 4.355663, "2014-04-29", None],
            [3, "pepie", "red", 4.739156, "2017-12-13", None],
            [3, "qrdq", "red", 4.831716, "2018-02-03", None],
            [3, "qrps", "yellow", 5.062321, "2019-05-13", None],
            [3, "rrqp", "yellow", 5.82323, "2020-01-09", None],
            [4, "sasr", "yellow", 6.212038, "2021-12-29", None],
            [4, "sspwe", "yellow", 6.231978, "2022-01-25", None],
            [4, "sssi", "yellow", 6.365346, "2023-03-20", None],
            [4, "wpfsi", "yellow", 7.461754, "2023-10-23", None],
            [4, "wqfed", "yellow", 7.775666, "2026-02-04", None],
            [4, "wsde", "yellow", 7.818521, "2027-06-13", None],
        ]
        expected_column_names = [
            "test_column_1",
            "test_column_2",
            "test_column_3",
            "test_column_4",
            "test_column_5",
            "test_column_6",
        ]

        expected_data = [
            dict(zip(expected_column_names, item)) for item in expected_array
        ]
        expected_df = pd.DataFrame(expected_data)

        actual_df = generator.synthesize(20)

        pd.testing.assert_frame_equal(expected_df, actual_df)

    @mock.patch("dataprofiler.profilers.StructuredProfiler.report")
    @mock.patch("synthetic_data.generators.logging.warning")
    def test_generate_dataset_with_invalid_sorting_type(
        self, mock_warning, mock_report
    ):
        mock_report.return_value = {
            "data_stats": [
                {
                    "data_type": "int",
                    "order": "cheese",
                    "statistics": {
                        "min": 1.0,
                        "max": 4.0,
                    },
                },
                {
                    "data_type": "int",
                    "order": "random",
                    "statistics": {
                        "min": 1.0,
                        "max": 4.0,
                    },
                },
            ]
        }
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        generator._generate_uncorrelated_column_data(num_samples=20)
        self.assertEqual(mock_warning.call_count, 2)

    @mock.patch("dataprofiler.profilers.StructuredProfiler.report")
    @mock.patch("synthetic_data.generators.logging.warning")
    def test_generate_dataset_with_valid_sorting_type(self, mock_warning, mock_report):
        mock_report.return_value = {
            "data_stats": [
                {
                    "data_type": "int",
                    "order": "ascending",
                    "statistics": {
                        "min": 1.0,
                        "max": 4.0,
                    },
                },
                {
                    "data_type": "int",
                    "order": "descending",
                    "statistics": {
                        "min": 1.0,
                        "max": 4.0,
                    },
                },
            ]
        }
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        generator._generate_uncorrelated_column_data(num_samples=20)

        self.assertEqual(mock_warning.call_count, 0)

    @mock.patch("dataprofiler.profilers.StructuredProfiler.report")
    def test_generate_dataset_with_none_columns(self, mock_report):
        expected_dataframe = pd.DataFrame()
        mock_report.return_value = {"data_stats": []}
        generator = TabularGenerator(profile=self.profile, is_correlated=False, seed=42)
        self.assertFalse(generator.is_correlated)
        actual_df = generator._generate_uncorrelated_column_data(num_samples=20)
        self.assertEqual(expected_dataframe.empty, actual_df.empty)


class TestGetOrderedColumn(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.Generator(np.random.PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%B %d %Y %H:%M:%S"]
        self.profile_options = dp.ProfilerOptions()
        self.profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )
        dp.set_seed(0)
        self.data = dp.Data(os.path.join(test_dir, "data/tabular.csv"))
        self.profile = dp.Profiler(
            data=self.data,
            options=self.profile_options,
            samples_per_update=len(self.data),
        )
        self.generator = TabularGenerator(
            profile=self.profile, is_correlated=False, seed=42
        )

    def test_get_ordered_column_datetime_ascending(self):
        data = datetime_generator.random_datetimes(
            rng=self.rng, min=self.start_date, max=self.end_date, num_rows=5
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

        actual = self.generator.get_ordered_column(data, "datetime", "ascending")
        np.testing.assert_array_equal(actual, expected)

    def test_get_ordered_column_datetime_descending(self):
        data = datetime_generator.random_datetimes(
            rng=self.rng, min=self.start_date, max=self.end_date, num_rows=5
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

        actual = self.generator.get_ordered_column(data, "datetime", "descending")

        np.testing.assert_array_equal(actual, expected)

    def test_get_ordered_column_custom_datetime_ascending(self):
        custom_date_format = ["%Y %m %d", "%B %d %Y %H:%M:%S"]
        data = datetime_generator.random_datetimes(
            rng=self.rng,
            format=custom_date_format,
            min=self.start_date,
            max=self.end_date,
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

        actual = self.generator.get_ordered_column(data, "datetime", "ascending")

        np.testing.assert_array_equal(actual, expected)

    def test_get_ordered_column_custom_datetime_descending(self):
        custom_date_format = ["%Y %m %d"]
        data = datetime_generator.random_datetimes(
            rng=self.rng,
            format=custom_date_format,
            min=self.start_date,
            max=self.end_date,
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

        actual = self.generator.get_ordered_column(data, "datetime", "descending")

        np.testing.assert_array_equal(actual, expected)
