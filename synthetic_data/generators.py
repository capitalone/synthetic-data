"""Contains generators for tabular, graph, and unstructured data profiles."""

import dataprofiler as dp
import numpy as np
import pandas as pd
from sklearn import preprocessing

from synthetic_data.base_generator import BaseGenerator
from synthetic_data.dataset_generator import generate_dataset
from synthetic_data.graph_synthetic_data import GraphDataGenerator
from synthetic_data.synthetic_data import make_data_from_report


class TabularGenerator(BaseGenerator):
    """Class for generating synthetic tabular data."""

    def __init__(
        self, profile, seed=None, noise_level: float = 0.0, is_correlated: bool = True
    ):
        """Initialize tabular generator object."""
        super().__init__(profile, seed)
        self.noise_level = noise_level
        self.is_correlated = is_correlated
        if not seed:
            self.tabular_generator_seed = self.seed
        else:
            self.tabular_generator_seed = seed
        self.rng = np.random.default_rng(seed=self.tabular_generator_seed)

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
        self,
        num_samples: int,
        noise_level: float = None,
    ):
        """Generate synthetic tabular data."""
        if noise_level is None:
            noise_level = self.noise_level

        if self.is_correlated:
            return make_data_from_report(
                report=self.profile.report(),
                n_samples=num_samples,
                noise_level=noise_level,
                seed=self.tabular_generator_seed,
            )
        else:
            col_data = self.generate_uncorrelated_column_data()

            return generate_dataset(
                rng=self.rng,
                columns_to_generate=col_data,
                dataset_length=num_samples,
            )

    def generate_uncorrelated_column_data(self):
        """Generate column data."""
        columns = self.profile.report()["data_stats"]
        col_data = []

        for col in columns:
            generator = col.get("data_type", None)
            order = col.get("order", None)
            col_stats = col["statistics"]
            min_value = col_stats.get("min", None)
            max_value = col_stats.get("max", None)

            if generator == "datetime":
                date_format = col_stats["format"]
                start_date = pd.to_datetime(
                    col_stats.get("min", None), format=date_format[0]
                )
                end_date = pd.to_datetime(
                    col_stats.get("max", None), format=date_format[0]
                )
                col_data.append(
                    {
                        "generator": generator,
                        "name": "dat",
                        "date_format_list": [date_format[0]],
                        "start_date": start_date,
                        "end_date": end_date,
                        "order": order,
                    }
                )
            elif generator == "int":
                col_data.append(
                    {
                        "generator": "integer",
                        "name": generator,
                        "min_value": min_value,
                        "max_value": max_value,
                        "order": order,
                    }
                )

            elif generator == "float":
                col_data.append(
                    {
                        "generator": generator,
                        "name": "flo",
                        "min_value": min_value,
                        "max_value": max_value,
                        "sig_figs": int(
                            col_stats.get("precision", None).get("max", None)
                        ),
                        "order": order,
                    }
                )

            elif generator == "string":
                if col.get("categorical", False):
                    total = 0
                    for count in col_stats["categorical_count"].values():
                        total += count

                    probabilities = []
                    for count in col_stats["categorical_count"].values():
                        probabilities.append(count / total)

                    col_data.append(
                        {
                            "generator": "categorical",
                            "name": "cat",
                            "categories": col_stats.get("categories", None),
                            "probabilities": probabilities,
                            "order": order,
                        }
                    )
                else:
                    col_data.append(
                        {
                            "generator": "text",
                            "name": "txt",
                            "chars": col_stats.get("vocab", None),
                            "str_len_min": min_value,
                            "str_len_max": max_value,
                            "order": order,
                        },
                    )
        return col_data


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
