"""Contains abstract class from which generators will inherit."""

import abc
from typing import Dict, Optional, Union

from dataprofiler import StructuredProfiler, UnstructuredProfiler


class BaseGenerator(metaclass=abc.ABCMeta):
    """Abstract generator class."""

    def __init__(self, profile: Union[StructuredProfiler, UnstructuredProfiler]):
        """Initialize generator object."""
        if not isinstance(profile, (StructuredProfiler, UnstructuredProfiler)):
            raise ValueError(
                "Profile must be an object of class "
                "StructuredProfiler, UnstructuredProfiler, or GraphProfiler."
            )
        self.report: Dict = profile.report()

    @abc.abstractmethod
    def synthesize(self, num_samples: int, noise_level: Optional[float], seed=None):
        """Make synthetic data."""
        raise NotImplementedError()
