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
        num_rows = [5,20,100]
        for nr in num_rows:
            result = random_integers(self.rng, num_rows=nr)
            self.assertEqual(result.shape[0], nr)
        result = random_integers(self.rng)
        self.assertEqual(result.shape[0], 1)

    def test_values_range(self):
        ranges = [(-1,1), (-10,10), (-100, 100)]
        for range in ranges:
            result = random_integers(self.rng, range[0], range[1])
            for x in result:
                self.assertGreaterEqual(x, range[0])
                self.assertLessEqual(x, range[1])