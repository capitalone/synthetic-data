"""Contains abstract class from which generators will inherit."""
class BaseGenerator():
    """Abstract generator class."""

    def __init__(self, profile):
        """Initialize generator object."""
        self.profile = profile.report()

    def synthesize(self):
        """Make synthetic data."""
        raise NotImplementedError("Not Implemented")