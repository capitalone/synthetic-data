"""Contains abstract class from which generators will inherit."""

import abc


class BaseGenerator(metaclass=abc.ABCMeta):
    """Abstract generator class."""

    def __init__(self, profile, options):
        """Initialize generator object."""
        self.profile = profile.report()
        self.options = options

    @abc.abstractmethod
    def synthesize(self):
        """Make synthetic data."""
        raise NotImplementedError()
