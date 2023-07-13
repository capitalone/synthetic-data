import os
import unittest
from unittest import mock

import dataprofiler as dp
import numpy as np
import pandas as pd

from synthetic_data import Generator
from synthetic_data.generators import (
    GraphGenerator,
    TabularGenerator,
    UnstructuredGenerator,
)

test_dir = os.path.dirname(os.path.realpath(__file__))


class TestGeneratorBuilder(unittest.TestCase):
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

        # create dataset and profile for tabular
        cls.tab_data = data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        cls.tab_profile = dp.Profiler(
            data, profiler_type="structured", options=profile_options
        )

        # create dataset and prfile for unstructured
        data = pd.Series(["first string", "second string"])
        cls.unstruct_profile = dp.Profiler(
            data, profiler_type="unstructured", options=profile_options
        )

        # create dataset and prfile for graph datasets
        data = dp.Data(os.path.join(test_dir, "data/graph.csv"))
        cls.graph_profile = dp.Profiler(
            data, profiler_type="graph", options=profile_options
        )

    def test_synthesize_tabular(self):
        generator = Generator(profile=self.tab_profile, seed=42)
        self.assertIsInstance(generator, TabularGenerator)
        synthetic_data = generator.synthesize(100)
        self.assertEqual(len(synthetic_data), 100)

        generator = Generator(data=self.tab_data, seed=42)
        synthetic_data_2 = generator.synthesize(100)
        self.assertEqual(len(synthetic_data_2), 100)

        # asserts that both  methods create the same results
        # if this ever fails may need to start setting seeds
        np.testing.assert_array_equal(synthetic_data, synthetic_data_2)

    def test_synthesize_unstruct(self):
        with self.assertRaises(NotImplementedError):
            generator = Generator(profile=self.unstruct_profile)
            self.assertIsInstance(generator, UnstructuredGenerator)
            synthetic_data = generator.synthesize()

    def test_synthesize_graph(self):
        generator = Generator(profile=self.graph_profile)
        self.assertIsInstance(generator, GraphGenerator)

        synthetic_data = generator.synthesize()
        self.assertEqual(synthetic_data.number_of_nodes(), 278)

    def test_invalid_config(self):
        with self.assertRaises(
            ValueError,
            msg="Warning: profile doesn't match user setting.",
        ):
            Generator(config=1)

    def test_no_profile_or_data(self):
        with self.assertRaisesRegex(
            ValueError,
            "No profile object or dataset was passed in kwargs. "
            "If you want to generate synthetic data from a "
            "profile, pass in a profile object through the "
            'key "profile" or data through the key "data" in kwargs.',
        ):
            Generator()

    def test_invalid_profile(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Profile object is invalid. The supported profile types are: \[.+\].",
        ):
            Generator(profile=1)

    def test_invalid_data(self):
        with self.assertRaisesRegex(
            ValueError, "data is not in an acceptable format for profiling."
        ):
            Generator(data=1)
