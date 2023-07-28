import unittest

import numpy as np
import pandas as pd

from synthetic_data.distinct_generators.text_generator import random_text


class TestTextGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_return_type(self):
        text_arr = random_text(self.rng)
        for x in text_arr:
            self.assertIsInstance(x, np.str_)

    def test_text_length(self):
        text_arr = random_text(self.rng, str_len_min=4, str_len_max=5)
        self.assertLessEqual(len(text_arr[0]), 5)
        self.assertGreaterEqual(len(text_arr[0]), 4)

    def test_num_rows(self):
        num_rows = [1, 5, 10]
        for nr in num_rows:
            text_arr = random_text(self.rng, num_rows=nr)
            self.assertEqual(text_arr.size, nr)

    def test_chars(self):
        chars_set = {"0", "1"}
        text_arr = random_text(self.rng, chars=["0", "1"])
        for s in text_arr:
            for char in s:
                self.assertIn(char, chars_set)
