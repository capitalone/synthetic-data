"""Contains abstract class from which generators will inherit."""

import abc
from typing import Dict


class BaseGenerator(metaclass=abc.ABCMeta):
    """Abstract generator class."""

    def __init__(self, seed, *args, **kwargs):
        """Initialize generator object."""
        self.seed = seed
        self.profile: Dict = kwargs.get("profile")

    @abc.abstractmethod
    def synthesize(self, *args, **kwargs):
        """Make synthetic data."""
        raise NotImplementedError()
