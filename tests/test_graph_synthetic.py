from __future__ import print_function

import os
import unittest
import random

import numpy as np
import scipy.stats as st

from synthetic_data.graph_synthetic_data import GraphDataGenerator

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class TestSyntheticGraphGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)

        cls.expected_profile = dict(
            num_nodes=200,
            num_edges=20,
            categorical_attributes=["pop"],
            continuous_attributes=["edge_weight"],
            avg_node_degree=0.2,
            global_max_component_size=5,
            continuous_distribution={
                "pop": None,
                "edge_weight": {
                    "name": "norm",
                    "properties": {
                        "best_fit_properties": [2, 0.5, 0.5],
                        "mean": [1., 4.2, 1],
                        "variance": [8.5, 305.7, 0],
                        "skew": [4.9, 82.4, 0.5],
                        "kurtosis": [7.2, 117436.2, 0.6],
                    }
                },
            },
            categorical_distribution={
                "pop": {
                    "bin_counts": [15, 30, 100, 200, 200, 8, 89, 473],
                    "bin_edges": [1.0, 1.75, 2.5, 3.25, 4.25, 5.25, 6.25, 7.25, 8],
                },
                "edge_weight": None,
            },
        )

        cls.synthetic_graph = GraphDataGenerator(cls.expected_profile)

    def test_synthesize(self):
        np.random.seed(1)
        random.seed(1)

        synthetic_graph = self.synthetic_graph.synthesize()
        self.assertEqual(synthetic_graph.number_of_nodes(), 200)
        self.assertEqual(synthetic_graph.number_of_edges(), 23)

    def test_sample_continuous(self):
        np.random.seed(5)
        attribute = self.synthetic_graph._continuous_attributes[0]
        sample = self.synthetic_graph.sample_continuous(attribute)[0]
        self.assertAlmostEqual(2.22061374344252, sample)
        
    def test_sample_categorical(self):
        np.random.seed(1)
        attribute = self.synthetic_graph._categorical_attributes[0]
        self.assertEqual(4, self.synthetic_graph.sample_categorical(attribute))

    def test_categorical_histogram(self):
        np.random.seed(2)
        attribute = self.synthetic_graph._categorical_attributes[0]
        data = []
        for n in range(0, 2000):
            data.append(self.synthetic_graph.sample_categorical(attribute))

        hist, edges = np.histogram(data, bins=[1.0, 1.75, 2.5, 3.25, 4.25, 5.25, 6.25, 7.25, 8], density=False)
        self.assertEqual(list(hist), [24, 45, 374, 379, 201, 81, 72, 824])

    def test_continuous_properties(self):
        np.random.seed(5)
        attribute = self.synthetic_graph._continuous_attributes[0]
        data = self.synthetic_graph.sample_continuous(attribute, 2000)
        properties = self.expected_profile["continuous_distribution"][attribute]["properties"]
        self.assertEqual(properties, self.expected_profile["continuous_distribution"][attribute]["properties"])

if __name__ == "__main__":
    unittest.main()
