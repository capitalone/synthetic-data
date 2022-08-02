"""Contains generator wrapper class that picks generator based on its profile type."""

from .generators import TabularGenerator, GraphGenerator, UnstructuredGenerator

class Generator():
    """Generator-wrapper class."""
    
    def pick(profile):
        """Pick a generator."""
        types_dict = {
            "TabularGenerator": TabularGenerator,
            "GraphGenerator": GraphGenerator,
            "UnstructuredGenerator": UnstructuredGenerator
        }
        pick_type = str(profile).split(".")[3].split()[0]
        return types_dict[pick_type]

    def __init__(self, profile):
        """Initialize wrapper object."""
        self.generator = pick(profile)
        self.data = self.generator(profile).synthesize()

    @property
    def data(self):
        """Get synthetic data."""
        return self.data

