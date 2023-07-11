import os
import unittest

import dataprofiler as dp
import numpy as np
import pandas as pd
import numpy as np

from synthetic_data.distinct_generators.int_generator import IntGenerator

test_dir = os.path.dirname(os.path.realpath(__file__))

class TestIntGenerator(unittest.TestCase):
    def setUp(self):
        profile_options = dp.ProfilerOptions()
        profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )

        # create dataset and profile for tabular
        tab_data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        tab_profile = dp.Profiler(
            tab_data, profiler_type="structured", options=profile_options
        )
        self.rng = np.random.default_rng(12345)        
        self.generator = IntGenerator(tab_profile, self.rng)
        self.np_array = self.generator.synthesize()

    def test_return_type(self):
        self.assertIsInstance(self.np_array, np.ndarray)
        for num in self.np_array:
            self.assertIsInstance(num, np.int64)

    def test_size(self):
        self.assertEqual(self.np_array.shape[0], self.generator.num_rows)

    def test_values_range(self):
        min_value, max_value = self.generator.min, self.generator.max
        for x in self.np_array:
            print(x, min_value)
            print(x, max_value)
            self.assertTrue(x >= min_value)
            self.assertTrue(x <= max_value)