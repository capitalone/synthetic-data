from synthetic_data.generators import TabularGenerator, UnstructuredGenerator, GraphGenerator
from synthetic_data import Generator
from synthetic_data.distinct_generators import int_generator
import dataprofiler as dp
import numpy as np
import pandas as pd
import os
import unittest

test_dir = os.path.dirname(os.path.realpath(__file__))


class TestRandomInts(unittest.TestCase):
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

    def test_tabular_generator(self):
        generator = Generator(profile=self.tab_profile, seed=42)
        self.assertIsInstance(generator, TabularGenerator)
        synthetic_data = generator.synthesize(100)
        self.assertEqual(len(synthetic_data), 100)
        random_seed = 0
        rng = np.random.default_rng(seed=random_seed)
        for col in synthetic_data:
            col = int_generator.random_integers(rng)
        
        for col in synthetic_data:
            self.assertIsInstance(type(col), int)

    # def test_unstructured_generator(self):
    #     generator = Generator(profile=self.tab_profile, seed=42)
    #     self.assertIsInstance(generator, UnstructuredGenerator)
    
    # def test_graph_generator(self):
    #     generator = Generator(profile=self.tab_profile, seed=42)
    #     self.assertIsInstance(generator, GraphGenerator)
    #     synthetic_data = generator.synthesize()
    #     self.assertEqual(synthetic_data.number_of_nodes(), 278)
    #     random_seed = 0
    #     rng = np.random.default_rng(seed=random_seed)
    #     for col in synthetic_data:
    #         col = int_generator.random_integers(rng)
        
    #     for col in synthetic_data:
    #         self.assertIsInstance(col, int)
