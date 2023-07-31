"""Contains generators for tabular, graph, and unstructured data profiles."""

import dataprofiler as dp
import numpy as np
from sklearn import preprocessing

from synthetic_data.base_generator import BaseGenerator
from synthetic_data.dataset_generator import generate_dataset
from synthetic_data.graph_synthetic_data import GraphDataGenerator
from synthetic_data.synthetic_data import make_data_from_report


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(
        self, profile, seed=None, noise_level: float = 0.0, method: str = "correlated"
    ):
        """Initialize tabular generator object."""
        super().__init__(profile, seed)
        self.noise_level = noise_level
        self.method = method

    @classmethod
    def post_profile_processing_w_data(cls, data, profile):
        """Create a profile from a dataset."""
        encoder_class = preprocessing.LabelEncoder
        profile_options = dp.ProfilerOptions()
        profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
            }
        )
        had_categorical_data = False
        text_cat_name_list = []
        for col_stat in profile.report()["data_stats"]:
            if col_stat["categorical"] and col_stat["data_type"] not in [
                "int",
                "float",
            ]:
                col_name = col_stat["column_name"]
                text_cat_name_list.append(col_name)
                encoder = encoder_class()
                encoder.fit(data.data[col_name])
                data.data[col_name] = encoder.transform(data.data[col_name])
                had_categorical_data = True

        if had_categorical_data:
            profile = dp.Profiler(
                data, options=profile_options, samples_per_update=len(data)
            )
        return profile

    def synthesize(
        self, num_samples: int, seed=None, noise_level: float = None, **kwargs
    ):
        """Generate synthetic tabular data."""
        if seed is None:
            seed = self.seed

        if noise_level is None:
            noise_level = self.noise_level

        if self.method == "correlated":
            return make_data_from_report(
                report=self.profile.report(),
                n_samples=num_samples,
                noise_level=noise_level,
                seed=seed,
            )
        else:

            random_seed = 0
            rng = np.random.default_rng(seed=random_seed)
            columns = self.profile.report()["data_stats"]
            col_data = []

            for i, col in enumerate(columns):
                data_type = col[i].get("data_type", None)
                ordered = col[i].get("order", None)

                if data_type == "datetime" and "date_format_list" in kwargs:
                    col_data.append(
                        {
                            "data_type": data_type,
                            "ordered": ordered,
                            "date_format_list": kwargs["date_format_list"],
                        }
                    )
                else:
                    col_data.append({"data_type": data_type, "ordered": ordered})

            return generate_dataset(
                rng=rng,
                columns_to_generate=col_data,
            )


class UnstructuredGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(self, profile, seed):
        """Initialize unstructured generator object."""
        super().__init__(profile, seed)

    def synthesize(self):
        """Generate synthetic unstructured data."""
        raise NotImplementedError()


class GraphGenerator(BaseGenerator):
    """Class for generating synthetic graph data."""

    def __init__(self, profile, seed=None):
        """Initialize graph generator object."""
        super().__init__(profile)

        self.generator = GraphDataGenerator(profile)

    def synthesize(self):
        """Generate synthetic graph data."""
        return self.generator.synthesize()
