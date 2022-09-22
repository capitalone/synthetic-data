'''Generate a synthetic graph using a profile'''
from cmath import nan
import math

import dataprofiler as dp
import networkx as nx
import scipy.stats as st
import numpy as np

class GraphDataGenerator(object):
    '''
    Synthesize graph data from a graph profile object
    params:
    graph_profile

    params type:
    graph_profile: GraphProfiler object/graph profile (dict)/NetworkX Graph

    return:
    None
    '''

    def __init__(self, graph):

        if not isinstance(graph, dp.GraphProfiler) and not isinstance(graph, dict) and not isinstance(graph, nx.Graph):
            raise NotImplementedError("Graph Profile must be a GraphProfiler object, a NetworkX Graph, or a dict")

        if isinstance(graph, nx.Graph):
            data = dp.Data(graph)
            profiler = dp.GraphProfiler(data)
            self.profile = profiler.report()
        elif isinstance(graph, dp.GraphProfiler):
            self.profile = graph.report()
        else:
            self.profile = graph

        self._num_nodes = self.profile.get('num_nodes')
        self._num_edges = self.profile.get('num_edges')
        self._avg_node_degree = self.profile.get("avg_node_degree")
        self._categorical_attributes = self.profile.get("categorical_attributes")
        self._continuous_attributes = self.profile.get("continuous_attributes")
        self._global_max_component_size = self.profile.get("global_max_component_size")
        self._continuous_distributions = self.profile.get("continuous_distribution")
        self._categorical_distributions = self.profile.get("categorical_distribution")

    def synthesize(self):
        """ Synthesize static graph data with edge attributes. """
        probability = self._avg_node_degree/self._num_nodes

        graph = nx.erdos_renyi_graph(n=self._num_nodes, p=probability)

        for u,v in graph.edges:
            edge_attributes = dict()

            for iter_attribute, attribute in enumerate([self._continuous_attributes, self._categorical_attributes]):
                for value in attribute: 
                    if value is not None:
                        if iter_attribute == 0:
                            sample = self.sample_continuous(value)
                        if iter_attribute == 1:
                            sample = self.sample_categorical(value)
                        edge_attributes[value] = sample

            graph.add_edge(u,v,**edge_attributes)
        return graph
    
    def sample_continuous(self, attribute, num_sample=1):
        """ Sample continuous distributions. """
        name = self._continuous_distributions[attribute]["name"]
        best_fit_properties = self._continuous_distributions[attribute]["properties"]["best_fit_properties"]
        distribution = None
        sample = 0

        if name == "norm":
            distribution = st.norm(loc=best_fit_properties[0], scale=best_fit_properties[1])
        if name == "logistic":
            distribution = st.logistic(loc=best_fit_properties[0], scale=best_fit_properties[1])
        if name == "lognorm":
            distribution = st.lognorm(a=best_fit_properties[0], loc=best_fit_properties[1], scale=best_fit_properties[2])
        if name == "expon":
            distribution = st.expon(loc=best_fit_properties[0], scale=best_fit_properties[1])
        if name == "uniform":
            distribution = st.uniform(loc=best_fit_properties[0], scale=best_fit_properties[1])
        if name == "gamma":
            distribution = st.gamma(a=best_fit_properties[0], loc=best_fit_properties[1], scale=best_fit_properties[2])
            
        sample = distribution.rvs(size=num_sample)
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

        for bin_number, cumulative_percent in enumerate(cumulative_distribution):
            if random_var <= cumulative_percent:
                return bin_number+1            
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
