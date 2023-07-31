"""Contains generator factory class that picks generator based on its profile type."""

import dataprofiler as dp
from dataprofiler import StructuredProfiler, UnstructuredProfiler
from dataprofiler.profilers.graph_profiler import GraphProfiler

from synthetic_data.generators import (
    GraphGenerator,
    TabularGenerator,
    UnstructuredGenerator,
)


class Generator:
    """Generator class."""

    valid_data_types = {
        StructuredProfiler: TabularGenerator,
        UnstructuredProfiler: UnstructuredGenerator,
        GraphProfiler: GraphGenerator,
    }

    @classmethod
    def is_valid_data(cls, profile):
        """Check if the profile's data is valid."""
        return profile.__class__ in cls.valid_data_types

    def __new__(cls, seed=None, config=None, *args, **kwargs):
        """Instantiate proper generator."""
        if config:
            try:
                return config(*args, **kwargs)
            except Exception:
                raise ValueError("Warning: profile doesn't match user setting.")

        profile = kwargs.pop("profile", None)
        data = kwargs.pop("data", None)
        if not profile and data is None:
            raise ValueError(
                "No profile object or dataset was passed in kwargs. "
                "If you want to generate synthetic data from a "
                "profile, pass in a profile object through the "
                'key "profile" or data through the key "data" in kwargs.'
            )
        if data is not None:
            profile_options = dp.ProfilerOptions()
            profile_options.set(
                {
                    "data_labeler.is_enabled": False,
                    "correlation.is_enabled": True,
                }
            )
            try:
                profile = dp.Profiler(
                    data, options=profile_options, samples_per_update=len(data)
                )
            except Exception as e:
                raise ValueError(
                    "data is not in an acceptable format for profiling."
                ) from e

        if not cls.is_valid_data(profile):
            raise ValueError(
                "Profile object is invalid. The supported profile "
                f"types are: {list(cls.valid_data_types.keys())}."
            )

        generator = cls.valid_data_types[profile.__class__]
        if data is not None:
            profile = generator.post_profile_processing_w_data(data, profile)
        return generator(profile, seed, *args, **kwargs)
