"""Contains generators for tabular, graph, and unstructured data profiles."""

from synthetic_data.synthetic_data import make_data_from_report

from .base_generator import BaseGenerator


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, report, options):
        """Initialize tabular generator object."""
        super().__init__(report, options)

    def synthesize(self, num_samples, options = None):
        """
        Generate synthetic tabular data.
        """
        if not options:
            options = {
                "noise_level": 0,
                "seed": None,
            }

        return make_data_from_report(
            self.report,
            n_samples=num_samples,
            noise_level=options.get("noise_level", 0.0),
            seed=options.get("seed"),
        )


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, report, options):
        """Initialize graph generator object."""
        super().__init__(report, options)

    def synthesize(self):
        """Generate synthetic graph data."""
        if not self.options:
            return "Synthesized graph data!"


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, report, options):
        """Initialize unstructured generator object."""
        super().__init__(report, options)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        if not self.options:
            return "Synthesized unstructured data!"
