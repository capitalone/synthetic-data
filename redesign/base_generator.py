"""Contains abstract class from which generators will inherit."""

import abc
from typing import Dict


class BaseGenerator(metaclass=abc.ABCMeta):
    """Abstract generator class."""

    def __init__(self, profile, options):
        """Initialize generator object."""
        self.report = profile.report()
        self.options = options

    @abc.abstractmethod
    def synthesize(self, options: Dict=None):
        """Make synthetic data."""
        raise NotImplementedError()
