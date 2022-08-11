
import math

import networkx as nx
import scipy.stats as st
import numpy as np

class GraphDataGenerator():

    def __init__(self, graph_profile, options=None):

        if not isinstance(graph_profile, dict):
            raise ValueError("Profile must be a dict.")

        self._num_nodes = graph_profile.get("num_nodes")
        self._num_edges = graph_profile.get("num_edges")
        self._avg_node_degree = graph_profile.get("avg_node_degree")
        self._categorical_attributes = graph_profile.get("categorical_attributes")
        self._continuous_attributes = graph_profile.get("continuous_attributes")
        self._global_max_component_size = graph_profile.get("global_max_component_size")
        self._continuous_distributions = graph_profile.get("continuous_distribution")
        self._categorical_distributions = graph_profile.get("categorical_distribution")

    def _synthesize(self):
        """ Synthesize static graph data with edge attributes. """
        probability = self._avg_node_degree/self._num_nodes

        graph = nx.erdos_renyi_graph(n=self._num_nodes, p=probability)

        for u,v in graph.edges:
            edge_attributes = dict()

            for continuous_attribute in self._continuous_attributes:
                if continuous_attribute is not None:
                    sample = self.sample_continuous(continuous_attribute)
                    edge_attributes[continuous_attribute] = sample
                    
            for categorical_attribute in self._categorical_attributes:
                if categorical_attribute is not None:
                    sample = self.sample_categorical(categorical_attribute)
                    edge_attributes[categorical_attribute] = sample

            graph.add_edge(u,v,**edge_attributes)
        return graph
    
    def sample_continuous(self, attribute, num_sample=1):
        """ Sample continuous distributions. """
        sample = self._continuous_distributions[attribute]["distribution"].rvs(size=num_sample)

        if num_sample == 1:
            return sample[0]
        return sample

    def sample_categorical(self, attribute, nsamples=1):
        """ Sample categorial distributions (histograms). """
        bin_counts = self._categorical_distributions[attribute]["bin_counts"]
        bin_edges = self._categorical_distributions[attribute]["bin_edges"]
        bin_number = self.random_bin_sample_categorial(bin_counts)

        sample = 0
        if bin_number == 0:
            sample = np.random.randint(low=0, high=bin_edges[bin_number])
        else:
            if bin_edges[bin_number]-bin_edges[bin_number-1] < 1:
                sample = math.ceil(bin_edges[bin_number-1])
            else:
                sample = np.random.randint(low=bin_edges[bin_number-1], high=math.ceil(bin_edges[bin_number]))
        return sample

    def random_bin_sample_categorial(self, bin_counts):
        """ Sample random bin from a categorical distribution histogram """
        cumulative_distribution = self.cumulative_histogram_distribution(bin_counts)
        random_var = np.random.uniform(0,1)
        bin_number = 0
        for cumulative_percent in cumulative_distribution:
            if random_var <= cumulative_percent:
                return bin_number+1
            bin_number+=1
        return None

    def cumulative_histogram_distribution(self, bin_counts):
        """ Calculate cumulative distribution for weighted bin sizes """
        total = sum(bin_counts)
        total_percent = []

        # get percent of bin counts
        for bin in range(0,len(bin_counts)):
            total_percent.append(bin_counts[bin]/total)
                
        # cumulate percent of bin counts
        for bin in range(1, len(total_percent)):
            total_percent[bin] = total_percent[bin]+total_percent[bin-1]

        return total_percent
