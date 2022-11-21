"""Contains generators for tabular, graph, and unstructured data profiles."""

from synthetic_data.synthetic_data import make_data_from_report

from .base_generator import BaseGenerator


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, seed, *args, **kwargs):
        """Initialize tabular generator object."""
        super().__init__(seed, *args, **kwargs)

    def synthesize(self, num_samples: int, noise_level: float = 0.0):
        """
        Generate synthetic tabular data.
        """
        return make_data_from_report(
            report=self.profile.report(),
            n_samples=num_samples,
            noise_level=noise_level,
            seed=self.seed,
        )


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, seed, *args, **kwargs):
        """Initialize graph generator object."""
        super().__init__(seed, *args, **kwargs)

    def synthesize(self):
        """Generate synthetic graph data."""
        raise NotImplementedError()


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, seed, *args, **kwargs):
        """Initialize unstructured generator object."""
        super().__init__(seed, *args, **kwargs)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        raise NotImplementedError()
