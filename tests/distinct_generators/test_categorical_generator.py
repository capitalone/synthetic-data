import unittest

import numpy as np
from numpy.random import PCG64, Generator

from synthetic_data.distinct_generators.categorical_generator import random_categorical


class TestRandomsCategories(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.categories = ["People", "Cats", "Dogs"]

    def test_default_return_validity(self):
        result = random_categorical(self.rng, num_rows=5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(set(result).issubset({"A", "B", "C", "D", "E"}))
        self.assertEqual(result.shape[0], 5)

        result = random_categorical(self.rng, num_rows=4)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(set(result).issubset({"A", "B", "C", "D", "E"}))
        self.assertEqual(result.shape[0], 4)

    def test_custom_return_validity(self):
        result = random_categorical(self.rng, categories=self.categories, num_rows=2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(set(result).issubset(self.categories))
        self.assertEqual(result.shape[0], 2)
