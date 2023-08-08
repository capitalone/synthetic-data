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
        min, max = -1, 1
        result = random_floats(self.rng, min, max)
        for x in result:
            self.assertGreaterEqual(x, min)
            self.assertGreaterEqual(max, x)

    def test_sig_figs(self):
        precision = 1
        result = random_floats(
            self.rng,
            min=0.817236764,
            max=1.92847927,
            precision=precision,
            num_rows=10,
        )
        for x in result:
            self.assertGreaterEqual(precision, len(str(x).split(".")[1]))

        precision = 5
        result = random_floats(
            self.rng,
            min=0.817236764,
            max=1.92847927,
            precision=precision,
            num_rows=10,
        )
        for x in result:
            self.assertGreaterEqual(precision, len(str(x).split(".")[1]))
