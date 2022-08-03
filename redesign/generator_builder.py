"""Contains generator factory class that picks generator based on its profile type."""

from generators import GraphGenerator, TabularGenerator, UnstructuredGenerator
from dataprofiler import StructuredProfiler, UnstructuredProfiler #, GraphProfiler

class Generator:
    """Generator class."""

    valid_data_types = {
        StructuredProfiler: TabularGenerator,
        #GraphProfiler: GraphGenerator, 
        UnstructuredProfiler: UnstructuredGenerator
    }

    @classmethod
    def is_valid_data(cls, profile):
        return profile.__class__ in cls.valid_data_types

    def __new__(cls, profile, config=None, options=None):
        """Instantiate proper generator."""
        if config:
            try:
                return config(profile, options)
            except Exception:
                print("Warning: profile doesn't match user setting. \
                        Proceeding with automatic generator selection...")
                        
        if cls.is_valid_data(profile):
            generator = cls.valid_data_types[profile.__class__]
            return generator(profile, options)