import unittest
from unittest import mock

import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data.dataset_generator import generate_dataset_by_class


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.dataset_length = 10

    def test_generate_dataset_with_invalid_generator(self):
        columns_to_gen = [{"generator": "non existent generator"}]
        with self.assertRaisesRegex(
            ValueError, "generator: non existent generator is not a valid generator."
        ):
            generate_dataset_by_class(
                self.rng,
                columns_to_generate=columns_to_gen,
                dataset_length=self.dataset_length,
                path=None,
            )

    @mock.patch("synthetic_data.dataset_generator.logging.warning")
    def test_generate_dataset_with_none_columns(self, mock_warning):
        empty_dataframe = pd.DataFrame()
        df = generate_dataset_by_class(self.rng, None, self.dataset_length, None)
        mock_warning.assert_called_once_with(
            "columns_to_generate is empty, empty dataframe will be returned."
        )
        self.assertEqual(empty_dataframe.empty, df.empty)

    def test_generate_custom_dataset(self):
        columns_to_gen = [
            {"generator": "integer", "name": "int", "min_value": 4, "max_value": 88},
            {
                "generator": "datetime",
                "name": "dat",
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
        expected_names = ["int", "dat", "txt", "str", "cat", "flo"]
        df = generate_dataset_by_class(
            self.rng,
            columns_to_generate=columns_to_gen,
            dataset_length=self.dataset_length,
            path=None,
        )
        # test column names
        self.assertListEqual(list(df.columns), expected_names)
        # test ints
        self.assertGreaterEqual(df["int"].min(), 4)
        self.assertLessEqual(df["int"].max(), 88)
        # test floats
        self.assertGreaterEqual(df["flo"].min(), 3)
        self.assertLessEqual(df["flo"].max(), 10)
        # test dates
        for date_str in df["dat"]:
            date_obj = pd.to_datetime(date_str, format="%B %d %Y %H:%M:%S")
            self.assertTrue(
                pd.Timestamp(2001, 12, 22) <= date_obj <= pd.Timestamp(2022, 12, 22)
            )
        # test categorical
        self.assertTrue(set(df["cat"]).issubset(["X", "Y", "Z"]))
        # test string and text
        chars_set = {"0", "1"}
        for s in df["str"]:
            for char in s:
                self.assertIn(char, chars_set)
        for s in df["txt"]:
            for char in s:
                self.assertIn(char, chars_set)

    @mock.patch("synthetic_data.dataset_generator.pd.DataFrame.to_csv")
    def test_path_to_csv(self, to_csv):
        """
        Ensure csv creation is triggered at the appropiate time.

        :param to_csv: mock of Pandas to_csv()
        :type to_csv: func
        """
        columns_to_gen = [
            {"generator": "integer"},
            {"generator": "datetime"},
            {"generator": "text"},
        ]
        to_csv.return_value = "assume Pandas to_csv for a dataframe runs correctly"
        path = "testing_path"
        generate_dataset_by_class(
            self.rng,
            columns_to_generate=columns_to_gen,
            dataset_length=4,
            path=path,
        )
        to_csv.assert_called_once_with(path, index=False, encoding="utf-8")
