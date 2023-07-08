import copy
from int_generator import random_integers
from text_generator import random_string, random_text
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator

def generate_dataset_by_class(
    rng: Generator,
    columns_to_generate: Optional[List[dict]] = None,
    dataset_length: int = 100000,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Randomly a dataset with a mixture of different data classes

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param columns_to_generate: Classes of data to be included in the dataset
    :type columns_to_generate: List[dict], None, optional
    :param dataset_length: length of the dataset generated
    :type dataset_length: int, optional
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional

    :return: pandas DataFrame
    """
    gen_funcs = {
        "integer": random_integers,
        "float": random_floats,
        "categorical": random_categorical,
        "ordered": get_ordered_column,
        "text": random_text,
        "datetime": random_datetimes,
        "string": random_string,
    }

    if columns_to_generate is None:
        columns_to_generate = [
            dict(generator="datetime"),
            dict(generator="integer"),
            dict(generator="float"),
            dict(generator="categorical"),
            dict(generator="ordered"),
            dict(generator="text"),
            dict(generator="string"),
        ]

    dataset = []
    for col in columns_to_generate:
        col_ = copy.deepcopy(col)
        col_generator = col_.pop("generator")
        if col_generator not in gen_funcs:
            raise ValueError(f"generator: {col_generator} is not a valid generator.")

        col_generator_function = gen_funcs.get(col_generator)
        dataset.append(col_generator_function(**col_, num_rows=dataset_length, rng=rng))
    return convert_data_to_df(dataset, path)


if __name__ == "__main__":
    # Params
    random_seed = 0
    GENERATED_DATASET_SIZE = 100000
    rng = np.random.default_rng(seed=random_seed)
    CLASSES_TO_GENERATE = [
        dict(
            generator="datetime", date_format_list=None, start_date=None, end_date=None
        ),
        dict(generator="integer", min_value=-1e6, max_value=1e6),
        dict(generator="float", min_value=-1e6, max_value=1e6, sig_figs=3),
        dict(generator="categorical", categories=None),
        dict(generator="ordered", start=0),
        dict(generator="text", chars=None, str_len_min=256, str_len_max=1000),
        dict(generator="string", chars=None, str_len_min=1, str_len_max=256),
    ]
    output_path = (
        f"data/seed_{random_seed}_"
        f"{'all' if CLASSES_TO_GENERATE is None else 'subset'}_"
        f"size_{GENERATED_DATASET_SIZE}.csv"
    )

    # Generate dataset
    data = generate_dataset_by_class(
        rng,
        columns_to_generate=CLASSES_TO_GENERATE,
        dataset_length=GENERATED_DATASET_SIZE,
        path=output_path,
    )
