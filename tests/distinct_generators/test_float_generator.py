import unittest

import numpy as np
from numpy.random import PCG64, Generator

from synthetic_data.distinct_generators.float_generator import random_floats


class TestRandomFloats(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))

    def test_return_type(self):
        result = random_floats(self.rng)
        self.assertIsInstance(result, np.ndarray)

    def test_size(self):
        num_rows = 5
        result = random_floats(self.rng, num_rows=num_rows)
        self.assertEqual(result.shape[0], num_rows)

    def test_values_range(self):
        min_value, max_value = -1, 1
        result = random_floats(self.rng, min_value, max_value)
        for x in result:
            self.assertGreaterEqual(x, min_value)
            self.assertGreaterEqual(max_value, x)

    def test_sig_figs(self):
        sig_figs = 1
        result = random_floats(
            self.rng,
            min_value=0.817236764,
            max_value=1.92847927,
            sig_figs=sig_figs,
            num_rows=10,
        )
        for x in result:
            self.assertGreaterEqual(sig_figs, len(str(x).split(".")[1]))

        sig_figs = 5
        result = random_floats(
            self.rng,
            min_value=0.817236764,
            max_value=1.92847927,
            sig_figs=sig_figs,
            num_rows=10,
        )
        for x in result:
            self.assertGreaterEqual(sig_figs, len(str(x).split(".")[1]))
