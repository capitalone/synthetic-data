"""Contains generator factory class that picks generator based on its profile type."""

from dataprofiler import StructuredProfiler, UnstructuredProfiler

from dataprofiler.profilers.graph_profiler import GraphProfiler

from redesign.generators import GraphGenerator, TabularGenerator, UnstructuredGenerator


class Generator:
    """Generator class."""

    valid_data_types = {
        StructuredProfiler: TabularGenerator,
        UnstructuredProfiler: UnstructuredGenerator,
        GraphProfiler: GraphGenerator,
    }

    @classmethod
    def is_valid_data(cls, profile):
        return profile.__class__ in cls.valid_data_types

    def __new__(cls, seed=None, config=None, *args, **kwargs):
        """Instantiate proper generator."""
        if config:
            try:
                return config(*args, **kwargs)
            except Exception as e:
                raise ValueError("Warning: profile doesn't match user setting.")

        profile = kwargs.pop("profile", None)
        if not profile:
            raise ValueError(
                "No profile object was passed in kwargs. "
                "If you want to generate synthetic data from a "
                "profile, pass in a profile object through the "
                'key "profile" in kwargs.'
            )

        if not cls.is_valid_data(profile):
            raise ValueError(
                "Profile object is invalid. The supported profile "
                f"types are: {list(cls.valid_data_types.keys())}."
            )

        generator = cls.valid_data_types[profile.__class__]
        return generator(profile, seed, *args, **kwargs)
