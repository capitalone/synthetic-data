"""Contains generators for tabular, graph, and unstructured data profiles."""

from redesign.base_generator import BaseGenerator
from synthetic_data.graph_synthetic_data import GraphDataGenerator
from synthetic_data.synthetic_data import make_data_from_report


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, seed=None, noise_level: float = 0.0):
        """Initialize tabular generator object."""
        super().__init__(profile, seed)
        self.noise_level = noise_level

    def synthesize(self, num_samples: int, seed=None, noise_level: float = None):
        """
        Generate synthetic tabular data.
        """
        if seed is None:
            seed = self.seed

        if noise_level is None:
            noise_level = self.noise_level

        return make_data_from_report(
            report=self.profile.report(),
            n_samples=num_samples,
            noise_level=noise_level,
            seed=seed,
        )


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, seed):
        """Initialize unstructured generator object."""
        super().__init__(profile, seed)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        raise NotImplementedError()


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, profile, seed=None):
        """Initialize graph generator object."""
        super().__init__(profile)

        self.generator = GraphDataGenerator(profile)

    def synthesize(self):
        """Generate synthetic graph data."""
        return self.generator.synthesize()
