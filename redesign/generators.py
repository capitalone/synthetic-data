"""Contains generators for tabular, graph, and unstructured data profiles."""

from base_generator import BaseGenerator


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, options):
        """Initialize tabular generator object."""
        super().__init__(profile, options)

    def synthesize(self):
        """
        Generate synthetic tabular data.
        """
        if not self.options:
            return "Synthesized tabular data!"


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, profile, options):
        """Initialize graph generator object."""
        super().__init__(profile, options)

    def synthesize(self):
        """Generate synthetic graph data."""
        if not self.options:
            return "Synthesized graph data!"


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, options):
        """Initialize unstructured generator object."""
        super().__init__(profile, options)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        if not self.options:
            return "Synthesized unstructured data!"
