import unittest

import dataprofiler as dp
import pandas as pd

from redesign.generator_builder import Generator


class TestSubgenerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile_options = dp.ProfilerOptions()
        profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )

        data = dp.Data("redesign_tests/iris.csv")
        cls.tab_profile = dp.Profiler(
            data, profiler_type="structured", options=profile_options
        )

        data = pd.Series(["first string", "second string"])
        cls.unstruct_profile = dp.Profiler(
            data, profiler_type="unstructured", options=profile_options
        )

        data = dp.Data("redesign_tests/graph.csv")
        cls.graph_profile = dp.Profiler(
            data, profiler_type="graph", options=profile_options
        )

    def test_synthesize_tab(self):
        result = Generator(profile=self.tab_profile).synthesize(100)
        self.assertEqual(len(result), 100)

    def test_synthesize_unstruct(self):
        with self.assertRaises(NotImplementedError):
            Generator(profile=self.unstruct_profile).synthesize()

    def test_synthesize_graph(self):
        with self.assertRaises(NotImplementedError):
            Generator(profile=self.graph_profile).synthesize()

    def test_invalid_config(self):
        with self.assertRaises(
            ValueError,
            msg="Warning: profile doesn't match user setting.",
        ):
            Generator(config=1).synthesize()

    def test_no_profile(self):
        with self.assertRaises(
            ValueError,
            msg="No profile object was passed in kwargs. "
            "If you want to generate synthetic data from a "
            "profile, pass in a profile object through the "
            'key "profile" in kwargs.',
        ):
            Generator().synthesize()

    def test_invalid_data(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Profile object is invalid. The supported profile types are: \[.+\].",
        ):
            Generator(profile=1).synthesize()
