import unittest
from unittest import mock

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

    def test_generate_dataset_with_none_columns(self):
        with self.assertRaisesRegex(
            ValueError, "columns_to_generate is a required parameter"
        ):
            generate_dataset_by_class(self.rng, None, self.dataset_length, None)

    def test_generate_custom_datasets(self):
        columns_to_gen = [
            {"generator": "integer"},
            {"generator": "datetime"},
            {"generator": "text"},
        ]
        expected_columns = ["integer", "datetime", "text"]
        df = generate_dataset_by_class(
            self.rng,
            columns_to_generate=columns_to_gen,
            dataset_length=self.dataset_length,
            path=None,
        )
        self.assertEqual(self.dataset_length, len(df))
        self.assertListEqual(list(df.columns), expected_columns)

        columns_to_gen = [
            {"generator": "string"},
            {"generator": "categorical"},
            {"generator": "float"},
        ]
        expected_columns = ["string", "categorical", "float"]
        df = generate_dataset_by_class(
            self.rng,
            columns_to_generate=columns_to_gen,
            dataset_length=4,
            path=None,
        )
        self.assertEqual(4, len(df))
        self.assertListEqual(list(df.columns), expected_columns)

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
