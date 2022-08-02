"""Contains generator factory class that picks generator based on its profile type."""

from .generators import GraphGenerator, TabularGenerator, UnstructuredGenerator
from dataprofiler import StructuredProfiler, UnstructuredProfiler, GraphProfiler


class Generator:
    """Generator-factory class."""
    @classmethod
    def pick(profile):
        """Pick a generator."""

        # Map string representations of generator classes to generator classes.
        types_dict = {
            "TabularGenerator": TabularGenerator,
            "GraphGenerator": GraphGenerator,
            "UnstructuredGenerator": UnstructuredGenerator,
        }

        # Extract string representation from profile __str__() output.
        pick_type = str(profile).split(".")[3].split()[0]

        # Return appropriate generator.
        return types_dict[pick_type]

    def __init__(self, profile):
        """Initialize wrapper object."""
        self.profile = profile
        self.generator = self.pick(profile)

    def data(self):
        """Make synthetic data."""
        return self.generator(self.profile).synthesize()
