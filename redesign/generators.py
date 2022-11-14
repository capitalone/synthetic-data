"""Contains generators for tabular, graph, and unstructured data profiles."""

from synthetic_data.synthetic_data import make_data_from_report

from .base_generator import BaseGenerator


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, options):
        """Initialize tabular generator object."""
        super().__init__(profile, options)

    def run(self):
        """
        Generate synthetic tabular data.
        """
        return make_data_from_report(
            self.profile,
            n_samples=self.options.get("samples"),
            noise_level=self.options.get("noise_level", 0.0),
            seed=self.options.get("seed"),
        )


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, profile, options):
        """Initialize graph generator object."""
        super().__init__(profile, options)

    def run(self):
        """Generate synthetic graph data."""
        if not self.options:
            return "Synthesized graph data!"


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, options):
        """Initialize unstructured generator object."""
        super().__init__(profile, options)

    def run(self):
        """Generate synthetic unstructured data."""
        if not self.options:
            return "Synthesized unstructured data!"
