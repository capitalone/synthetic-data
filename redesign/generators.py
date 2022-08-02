"""Contains generators for tabular, graph, and unstructured data profiles."""

from .base_generator import BaseGenerator


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile):
        """Initialize tabular generator object."""
        super().__init__(profile)

    def synthesize(self):
        """Generate synthetic tabular data."""
        pass


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, profile):
        """Initialize graph generator object."""
        super().__init__(profile)

    def synthesize(self):
        """Generate synthetic graph data."""
        pass


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile):
        """Initialize unstructured generator object."""
        super().__init__(profile)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        pass
