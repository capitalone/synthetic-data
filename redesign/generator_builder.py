"""Contains generator factory class that picks generator based on its profile type."""

from .generators import GraphGenerator, TabularGenerator, UnstructuredGenerator
from dataprofiler import StructuredProfiler, UnstructuredProfiler

class Generator:
    """Generator class."""

    valid_data_types = {
        StructuredProfiler: TabularGenerator, 
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
            except Exception as e:
                print("Warning: profile doesn't match user setting. \
                        Proceeding with automatic generator selection...")
        
        try:
            if not cls.is_valid_data(profile):
                raise TypeError("Only Structured and Unstructured are \
                                acceptable profile types.")
        except TypeError as e:
            print(e)
            
                        
        if cls.is_valid_data(profile):
            generator = cls.valid_data_types[profile.__class__]
            return generator(profile, options)
