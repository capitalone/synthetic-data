"""Contains abstract class from which generators will inherit."""

import abc


class BaseGenerator(metaclass=abc.ABCMeta):
    """Abstract generator class."""

    def __init__(self, profile, seed=None, *args, **kwargs):
        """Initialize generator object."""
        self.seed = seed
        self.profile = profile

    @abc.abstractmethod
    def synthesize(self, *args, **kwargs):
        """Make synthetic data."""
        raise NotImplementedError()
