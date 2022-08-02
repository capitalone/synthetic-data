"""Contains generator wrapper class that picks generator based on its profile type."""

from .generators import GraphGenerator, TabularGenerator, UnstructuredGenerator


class Generator:
    """Generator-wrapper class."""

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
        self.generator = pick(profile)
        self.data = self.generator(profile).synthesize()

    @property
    def data(self):
        """Get synthetic data."""
        return self.data
