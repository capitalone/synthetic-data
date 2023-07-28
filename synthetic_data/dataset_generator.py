"""Contains generator that returns collective df of requested distinct generators."""

import copy
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator

from synthetic_data.distinct_generators.categorical_generator import random_categorical
from synthetic_data.distinct_generators.datetime_generator import random_datetimes
from synthetic_data.distinct_generators.float_generator import random_floats
from synthetic_data.distinct_generators.int_generator import random_integers
from synthetic_data.distinct_generators.str_generator import random_string


def convert_data_to_df(
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


def generate_dataset(
    rng: Generator,
    columns_to_generate: List[dict],
    dataset_length: int = 100000,
) -> pd.DataFrame:
    """
    Randomizes a dataset with a mixture of different data classes.

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param columns_to_generate: Classes of data to be included in the dataset
    :type columns_to_generate: List[dict], None, optional
    :param dataset_length: length of the dataset generated, default 100,000
    :type dataset_length: int, optional

    :return: pandas DataFrame
    """
    gen_funcs = {
        "integer": random_integers,
        "float": random_floats,
        "categorical": random_categorical,
        "datetime": random_datetimes,
        "string": random_string,
    }

    if not columns_to_generate:
        logging.warning(
            "columns_to_generate is empty, empty dataframe will be returned."
        )
        return pd.DataFrame()

    dataset = []
    column_names = []
    for col in columns_to_generate:
        col_ = copy.deepcopy(col)
        col_generator = col_.pop("generator")
        if col_generator not in gen_funcs:
            raise ValueError(f"generator: {col_generator} is not a valid generator.")
        if "name" in col_:
            name = col_.pop("name")
        else:
            name = col_generator
        col_generator_function = gen_funcs.get(col_generator)
        dataset.append(col_generator_function(**col_, rng=rng, num_rows=dataset_length))
        column_names.append(name)
    return convert_data_to_df(dataset, column_names=column_names)
