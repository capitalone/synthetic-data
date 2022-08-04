from __future__ import print_function
from cmath import exp

import os
import unittest
import matplotlib.pyplot as plt

import scipy.stats as st
import networkx as nx
import numpy as np

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
            avg_node_degree=0.5,
            global_max_component_size=5,
            continuous_distribution={
                "pop": None,
                "edge_weight": {
                    "name": "lognorm",
                    "distribution": st.norm()
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
        synthetic_graph = self.synthetic_graph._synthesize()
        graph_df = nx.to_pandas_edgelist(synthetic_graph)
        print(graph_df)

        #plot
        """   
        pos = nx.spring_layout(synthetic_graph)

        nx.draw(synthetic_graph, pos)
        
        edge_labels = nx.get_edge_attributes(synthetic_graph,'edge_weight')
        
        print(edge_labels)
        for node in edge_labels:
            edge_labels[node] = round(edge_labels[node], 3)
        nx.draw_networkx_edge_labels(synthetic_graph, pos, edge_labels)
        plt.show()
        """

    def test_sample_continuous(self):
        np.random.seed(5)
        attribute = self.synthetic_graph._continuous_attributes[0]
        self.assertAlmostEqual(0.441227486885, self.synthetic_graph.sample_continuous(attribute))
        
    def test_sample_categorical(self):
        np.random.seed(1)
        attribute = self.synthetic_graph._categorical_attributes[0]
        self.assertEqual(4, self.synthetic_graph.sample_categorical(attribute))
    
    def test_sample_categorical_plot(self):
        np.random.seed(2)
        attribute = self.synthetic_graph._categorical_attributes[0]
        data = []
        for n in range(0, 2000):
            data.append(self.synthetic_graph.sample_categorical(attribute))
        
        hist, edges = np.histogram(data, bins=[1.0, 1.75, 2.5, 3.25, 4.25, 5.25, 6.25, 7.25, 8], density=False)
        self.assertEqual(list(hist), [24, 45, 374, 379, 201, 81, 72, 824])

        # plots
        expected_hist = self.expected_profile["categorical_distribution"]["pop"]["bin_counts"]
        hist = hist/np.max(hist)
        expected_hist = expected_hist/np.max(expected_hist)

        num_bin = 8
        bin_lims = np.linspace(0,1,num_bin+1)
        bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
        bin_widths = bin_lims[1:]-bin_lims[:-1]

        fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2)
        ax1.bar(bin_centers, hist, width = bin_widths, align = 'center')
        ax2.bar(bin_centers, expected_hist, width = bin_widths, align = 'center', alpha = 0.5)
        ax1.set_title('amplitude-normalized expected distribution')
        ax2.set_title('amplitude-normalized computed distribution')
        plt.show()

    def test_sample_continuous_plot(self):
        np.random.seed(5)
        attribute = self.synthetic_graph._continuous_attributes[0]
        data = self.synthetic_graph.sample_continuous(attribute, 2000)

        # plot
        fig, ax1 = plt.subplots()
        ax1.hist(list(data), bins=100)
        pts = np.linspace(-3, 3)
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 0.65)
        ax2.plot(pts, self.synthetic_graph._continuous_distributions[attribute]["distribution"].pdf(pts), color='red')
        plt.show()
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()