import unittest

import numpy as np
import pandas as pd

from synthetic_data.distinct_generators.text_generator import random_string


class TestTextGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_return_type(self):
        str_arr = random_string(self.rng)
        for x in str_arr:
            self.assertIsInstance(x, np.str_)

    def test_str_length(self):
        str_arr = random_string(self.rng, str_len_min=4, str_len_max=5)
        self.assertLessEqual(len(str_arr[0]), 5)
        self.assertGreaterEqual(len(str_arr[0]), 4)

    def test_num_rows(self):
        num_rows = [1, 5, 10]
        for nr in num_rows:
            str_arr = random_string(self.rng, num_rows=nr)
            self.assertEqual(str_arr.size, nr)

    def test_chars(self):
        chars_set = {"0", "1"}
        str_arr = random_string(self.rng, chars=["0", "1"])
        for s in str_arr:
            for char in s:
                self.assertIn(char, chars_set)
