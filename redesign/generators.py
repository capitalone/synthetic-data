"""Contains generators for tabular, graph, and unstructured data profiles."""

from typing import Optional, Union

from dataprofiler import StructuredProfiler, UnstructuredProfiler

from synthetic_data.synthetic_data import make_data_from_report

from .base_generator import BaseGenerator


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile: Union[StructuredProfiler, UnstructuredProfiler]):
        """Initialize tabular generator object."""
        super().__init__(profile)

    def synthesize(self, num_samples: int, noise_level: Optional[float], seed=None):
        """
        Generate synthetic tabular data.
        """
        return make_data_from_report(
            self.report,
            n_samples=num_samples,
            noise_level=noise_level,
            seed=seed,
        )


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, profile: Union[StructuredProfiler, UnstructuredProfiler]):
        """Initialize graph generator object."""
        super().__init__(profile)

    def synthesize(self):
        """Generate synthetic graph data."""
        if not self.options:
            return "Synthesized graph data!"


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile: Union[StructuredProfiler, UnstructuredProfiler]):
        """Initialize unstructured generator object."""
        super().__init__(profile)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        if not self.options:
            return "Synthesized unstructured data!"
