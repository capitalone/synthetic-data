"""Contains abstract class from which generators will inherit."""

import abc


class BaseGenerator(metaclass=abc.ABCMeta):
    """Abstract generator class."""

    def __init__(self, profile, options):
        """Initialize generator object."""
        self.profile = profile.report()

        if isinstance(options, dict):
            self.options = options
        else:
            self.options = {}

    @abc.abstractmethod
    def run(self):
        """Make synthetic data."""
        raise NotImplementedError()
