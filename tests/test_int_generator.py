import os
import unittest
import pandas as pd
import numpy as np

from synthetic_data.distinct_generators.int_generator import random_integers

class TestIntGenerator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)     

    def test_return_type(self):
        result = random_integers(self.rng)
        self.assertIsInstance(result, np.ndarray)
        for num in result:
            self.assertIsInstance(num, np.int64)

    def test_size(self):
        num_rows = 5
        result = random_integers(self.rng, num_rows=num_rows)
        self.assertEqual(result.shape[0], num_rows)

    def test_values_range(self):
        min_value, max_value = -1,1
        result = random_integers(self.rng, min_value, max_value)
        for x in result:
            self.assertGreaterEqual(x, min_value)
            self.assertLessEqual(x, max_value)