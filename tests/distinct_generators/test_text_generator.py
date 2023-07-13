import unittest
from unittest import mock
import pandas as pd
import numpy as np
from synthetic_data.distinct_generators.text_generator import random_string, random_text


class TestTextGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_return_type(self):
        str_arr = random_string(self.rng)
        txt_arr = random_text(self.rng)
        for x in str_arr:
            self.assertIsInstance(x, np.str_)
        for x in txt_arr:
            self.assertIsInstance(x, np.str_)

    def test_str_length(self):
        str_arr = random_string(self.rng, str_len_min=1, str_len_max=256)
        txt_arr = random_text(self.rng, str_len_min=256, str_len_max=1000)
        with self.assertRaises(ValueError):
            random_text(self.rng, str_len_min=255)

        self.assertLessEqual(len(str_arr[0]), 256)
        self.assertGreaterEqual(len(str_arr[0]), 1)
        self.assertLessEqual(len(txt_arr[0]), 1000)
        self.assertGreaterEqual(len(txt_arr[0]), 256)

    def test_num_rows(self):
        num_rows = [1, 5, 10]
        for nr in num_rows:
            str_arr = random_string(self.rng, num_rows=nr)
            txt_arr = random_text(self.rng, num_rows=nr)
            self.assertEqual(str_arr.size, nr)
            self.assertEqual(txt_arr.size, nr)

    def test_chars(self):
        chars_set = {"0", "1"}
        str_arr = random_string(self.rng, chars=["0", "1"])
        txt_arr = random_text(self.rng, chars=["0", "1"])
        for s in str_arr:
            for char in s:
                self.assertIn(char, chars_set)
        for s in txt_arr:
            for char in s:
                self.assertIn(char, chars_set)
