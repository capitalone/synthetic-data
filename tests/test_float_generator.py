
import unittest
import numpy as np 
from numpy.random import Generator, PCG64
from synthetic_data.dataset_generators.float_generator import random_floats


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
        sig_figs = 2
        result = random_floats(self.rng, sig_figs=sig_figs)
        for x in result:
            self.assertTrue(len(str(x).split(".")[1]) <= sig_figs)
        

