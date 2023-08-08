import unittest

import numpy as np
import pandas as pd

from synthetic_data.distinct_generators.text_generator import random_text


class TestTextGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_return_type(self):
        text_arr = random_text(self.rng, vocab=None, min=1, max=1,)
        for x in text_arr:
            self.assertIsInstance(x, np.str_)

    def test_text_length_range(self):
        text_arr = random_text(self.rng, min=3, max=5, vocab=["0", "1"])
        for text in text_arr:
            self.assertLessEqual(len(text), 5)
            self.assertGreaterEqual(len(text), 3)

    def test_text_equal_length_range(self):
        text_arr = random_text(self.rng, min=5, max=5, vocab=["0", "1"])
        for text in text_arr:
            self.assertEqual(len(text), 5)

    def test_num_rows(self):
        num_rows = [1, 5, 10]
        for nr in num_rows:
            text_arr = random_text(self.rng, min=1, max=1, vocab=None, num_rows=nr)
            self.assertEqual(text_arr.size, nr)

    def test_chars(self):
        vocab_set = {"0", "1"}
        text_arr = random_text(self.rng, min=1, max=1, vocab=["0", "1"])
        for t in text_arr:
            for char in t:
                self.assertIn(char, vocab_set)
