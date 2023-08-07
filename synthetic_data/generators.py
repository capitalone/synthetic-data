"""Contains generators for tabular, graph, and unstructured data profiles."""

import dataprofiler as dp
import numpy as np
import pandas as pd
from sklearn import preprocessing
import inspect
import logging
from typing import List, Optional
import copy

from synthetic_data.base_generator import BaseGenerator
from synthetic_data.dataset_generator import generate_dataset
from synthetic_data.graph_synthetic_data import GraphDataGenerator
from synthetic_data.synthetic_data import make_data_from_report
from synthetic_data.distinct_generators.categorical_generator import random_categorical
from synthetic_data.distinct_generators.datetime_generator import random_datetimes
from synthetic_data.distinct_generators.float_generator import random_floats
from synthetic_data.distinct_generators.int_generator import random_integers
from synthetic_data.distinct_generators.text_generator import random_text


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
                "integer": random_integers,
                "float": random_floats,
                "categorical": random_categorical,
                "datetime": random_datetimes,
                "text": random_text,
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
            return self.generate_uncorrelated_column_data(num_samples)
        

    def generate_uncorrelated_column_data(self, num_samples):
        """Generate column data."""
        columns = self.profile.report()["data_stats"]
        dataset = []
        column_names = []
        sorting_types = ["ascending", "descending"]

        for col in columns:
            col_ = copy.deepcopy(col)
            generator_name = col_.get("data_type", None)
            generator_func = self.gen_funcs.get(generator_name, None)
            
            params_gen_func_list = inspect.signature(generator_func)

            # edge cases for extracting data from profiler report.
            if generator_name == "datetime":
                col_["min"] = col_["statistics"].get("min", None)
                col_["max"] = col_["statistics"].get("max", None)
                col_["format"] = col_["statistics"].get("format", None)

            elif generator_name == "float":
                col_["precision"] = col_["statistics"].get("precision", None).get("max", None)

            elif generator_name == "string":
                if col_.get("categorical", False):
                    total = 0
                    for count in col["statistics"]["categorical_count"].values():
                        total += count

                    probabilities = []
                    for count in col["statistics"]["categorical_count"].values():
                        probabilities.append(count / total)
                    
                    col_["probabilities"] = probabilities
                    col_["categories"] = col_["statistics"].get("categories", None),
                
                else:
                    col_["vocab"] = col_["statistics"].get("vocab", None)   


            param_build = {}
            for param in params_gen_func_list:
                param_build[param] = col_[param]

           
            generated_data = generator_func(
                **param_build, num_rows=num_samples, rng=self.rng
            )
            if param_build["order"] in sorting_types:
                dataset.append(
                    self.get_ordered_column(
                        generated_data,
                        generator_func,
                        param_build["order"],
                    )
                )
            else:
                if param_build["order"] is not None:
                    logging.warning(
                        f"""{generator_name} is passed with sorting type of {param_build["order"]}.
                        Ascending and descending are the only supported options.
                        No sorting action will be taken."""
                    )
                if generator_name == "datetime":
                    date = generated_data[:, 0]
                    dataset.append(date)
                else:
                    dataset.append(generated_data)
            column_names.append(generator_name)
                 
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


"""gen_funcs = {"data_type": function_name}

data = [list of np.arrays]
for col in columns: 
	generator = col.get("data_type", None) #datetime 


	# get params as list of strings
	param_gen_func_list = generator.__params__
	# e.g. if generator == "datetime" --> ['rng', 'date_format', 'start_date', 'end_date']
	# e.g. if generator == "float" --> ['rng', 'min_value', 'max_value', 'sig_figs', 'num_rows']
	
	param_build = {}
	for col_data in col: #where col is report[column_index_of_report_data_stats]

	data.append(generator(**param_build))

	df = build_df(data)

>>> generator.__params__

def generate_datetime(
    rng: Generator,
    date_format: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> str:

def random_floats(
    rng: Generator,
    min_value: int = -1e6,
    max_value: int = 1e6,
    sig_figs: int = 3,
    num_rows: int = 1,
) -> np.array:
	pass """