"""Contains generators for tabular, graph, and unstructured data profiles."""

import copy
import inspect
import logging
from typing import List, Optional

import dataprofiler as dp
import numpy as np
import pandas as pd
from sklearn import preprocessing

from synthetic_data.base_generator import BaseGenerator
from synthetic_data.distinct_generators.categorical_generator import random_categorical
from synthetic_data.distinct_generators.datetime_generator import random_datetimes
from synthetic_data.distinct_generators.float_generator import random_floats
from synthetic_data.distinct_generators.int_generator import random_integers
from synthetic_data.distinct_generators.null_generator import null_generation
from synthetic_data.distinct_generators.text_generator import random_text
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
        self.gen_funcs = {
            "int": random_integers,
            "float": random_floats,
            "categorical": random_categorical,
            "datetime": random_datetimes,
            "string": random_text,
            "text": random_text,
            "null_generator": null_generation,
        }

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
            return self._generate_uncorrelated_column_data(num_samples)

    def _generate_uncorrelated_column_data(self, num_samples):
        """Generate column data."""
        columns = self.profile.report()["data_stats"]
        dataset = []
        column_names = []
        sorting_types = ["ascending", "descending"]

        for col in columns:
            col_ = copy.deepcopy(col)

            generator_name = col_.get("data_type", None)
            column_header = col_.get("column_name", None)

            col_["rng"] = self.rng
            col_["num_rows"] = num_samples
            if generator_name:
                if generator_name in ["string", "text"]:
                    if col_.get("categorical", False):
                        generator_name = "categorical"
                        total = 0
                        for count in col["statistics"]["categorical_count"].values():
                            total += count

                        probabilities = []
                        for count in col["statistics"]["categorical_count"].values():
                            probabilities.append(count / total)

                        col_["probabilities"] = probabilities
                        col_["categories"] = col_["statistics"].get("categories", None)

                    col_["vocab"] = col_["statistics"].get("vocab", None)

                col_["min"] = col_["statistics"].get("min", None)
                col_["max"] = col_["statistics"].get("max", None)

                # edge cases for extracting data from profiler report.
                if generator_name == "datetime":
                    col_["format"] = col_["statistics"].get("format", None)
                    col_["min"] = pd.to_datetime(
                        col_["statistics"].get("min", None), format=col_["format"][0]
                    )
                    col_["max"] = pd.to_datetime(
                        col_["statistics"].get("max", None), format=col_["format"][0]
                    )

                if generator_name == "float":
                    col_["precision"] = int(
                        col_["statistics"].get("precision", None).get("max", None)
                    )
            elif not generator_name:
                generator_name = "null_generator"

            generator_func = self.gen_funcs.get(generator_name, None)
            params_gen_funcs = inspect.signature(generator_func)

            param_build = {}
            for param in params_gen_funcs.parameters.items():
                param_build[param[0]] = col_[param[0]]

            generated_data = generator_func(**param_build)
            if (not generator_name == "null_generator") and col_[
                "order"
            ] in sorting_types:
                dataset.append(
                    self.get_ordered_column(
                        generated_data,
                        generator_name,
                        col_["order"],
                    )
                )
            else:
                if (not generator_name == "null_generator") and col_[
                    "order"
                ] is not None:
                    logging.warning(
                        f"""{generator_name} is passed with sorting type of {col_["order"]}.
                        Ascending and descending are the only supported options.
                        No sorting action will be taken."""
                    )
                if generator_name == "datetime":
                    date = generated_data[:, 0]
                    dataset.append(date)
                else:
                    dataset.append(generated_data)

            column_names.append(column_header)

        return self.convert_data_to_df(dataset, column_names=column_names)

    def convert_data_to_df(
        self,
        np_data: np.array,
        index: bool = False,
        column_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert np array to a pandas dataframe.

        :param np_data: np array to be converted
        :type np_data: numpy array
        :param index: whether to include index in output to csv
        :type index: bool, optional
        :param column_names: The names of the columns of a dataset
        :type column_names: List, None, optional
        :return: a pandas dataframe
        """
        # convert array into dataframe
        if not column_names:
            column_names = [x for x in range(len(np_data))]
        dataframe = pd.DataFrame.from_dict(dict(zip(column_names, np_data)))
        return dataframe

    def get_ordered_column(
        self,
        data: np.array,
        data_type: str,
        order: str = "ascending",
    ) -> np.array:
        """Sort a numpy array based on data type.

        :param data: numpy array to be sorted
        :type data: np.array
        :param data_type: type of data to be sorted
        :type data_type: str
        :param order: order of sort. Options: ascending or descending
        :type order: str

        :return: sorted numpy array
        """
        if data_type == "datetime":
            sorted_data = np.array(sorted(data, key=lambda x: x[1]))
            sorted_data = sorted_data[:, 0]

        else:
            sorted_data = np.sort(data)

        if order == "descending":
            return sorted_data[::-1]
        return sorted_data


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
