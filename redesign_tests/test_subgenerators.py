import unittest

import dataprofiler as dp
import pandas as pd

from redesign.generator_builder import Generator


class TestSubgenerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = dp.Data("redesign_tests/iris.csv")

        profile_options = dp.ProfilerOptions()
        profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )
        cls.tab_profile = dp.Profiler(
            data, profiler_type="structured", options=profile_options
        )

        unstruct_data = pd.Series(["first string", "second string"])
        cls.unstruct_profile = dp.Profiler(
            unstruct_data, profiler_type="unstructured", options=profile_options
        )

    def test_synthesize_tab(self):
        result = Generator(profile=self.tab_profile).synthesize(100)
        self.assertEqual(len(result), 100)

    def test_synthesize_unstruct(self):
        with self.assertRaises(NotImplementedError):
            Generator(profile=self.unstruct_profile).synthesize()
