import os
import unittest
from collections import OrderedDict

import dataprofiler as dp
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

import synthetic_data.dataset_generator as dg
from synthetic_data.generators import TabularGenerator


class TestDatetimeFunctions(unittest.TestCase):
    def setUp(self):
        profile_options = dp.ProfilerOptions()
        profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )
        test_dir = os.path.abspath("tests")
        # create dataset and profile for tabular
        data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        self.columns = dp.Profiler(
            data, profiler_type="structured", options=profile_options
        ).report()["data_stats"]

    def test_get_ordered_column_descending(self):
        data = OrderedDict(
            {
                "int": np.array([1, 2, 3, 4, 5]),
                "float": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                "string": np.array(["abc", "bca", "cab"]),
                "categorical": np.array(["A", "B", "C", "D", "E"]),
            }
        )
        ordered_data = [
            np.array([5, 4, 3, 2, 1]),
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            np.array(["cab", "bca", "abc"]),
            np.array(["E", "D", "C", "B", "A"]),
        ]
        output_data = []
        for data_type in data.keys():
            output_data.append(
                dg.get_ordered_column(data[data_type], data_type, order="descending")
            )

        for i in range(len(output_data)):
            self.assertTrue(np.array_equal(output_data[i], ordered_data[i]))

    def test_generate_dataset_columns_format_list(self):
        random_seed = 0
        rng = np.random.default_rng(seed=random_seed)
        col_data = []

        for i, col in enumerate(self.columns):
            data_type = col[i].get("data_type", None)
            ordered = col[i].get("order", None)

            if data_type == "datetime":
                col_data.append(
                    {
                        "data_type": data_type,
                        "ordered": ordered,
                        "date_format_list": "%m/%d/%Y, %H:%M:%S",
                    }
                )
            else:
                col_data.append({"data_type": data_type, "ordered": ordered})

        return dg.generate_dataset_by_class(
            rng=rng,
            columns_to_generate=col_data,
        )


class TabGenDatasetGen(unittest.TestCase):
    def test_setup(self):
        profile_options = dp.ProfilerOptions()
        profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )

        # create dataset and profile for tabular
        data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        self.columns = dp.Profiler(
            data, profiler_type="structured", options=profile_options
        ).report()["data_stats"]
        data = OrderedDict(
            {
                "int": np.array([5, 4, 3, 2, 1]),
                "float": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                "string": np.array(["abcde", "bcdea", "cdeab", "deabc", "eabcd"]),
                "categorical": np.array(["E", "D", "C", "B", "A"]),
            }
        )
        ordered_data = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array(["abcde", "bcdea", "cdeab", "deabc", "eabcd"]),
            np.array(["A", "B", "C", "D", "E"]),
        ]

        # create a tabular generator and see whether the output of synthesize when method == simple is a pd.frame that matches the thing that generate dataset_by_class returns.

        ordered_data = np.asarray(ordered_data)
        output_data = []
        for data_type in data.keys():
            output_data.append(dg.get_ordered_column(data[data_type], data_type))
        ordered_data = np.asarray(ordered_data)

        self.assertTrue(np.array_equal(output_data, ordered_data))

    def test_generate_dataset_columns_format_list(self):
        random_seed = 0
        rng = np.random.default_rng(seed=random_seed)
        col_data = []

        for i, col in enumerate(self.columns):
            data_type = col[i].get("data_type", None)
            ordered = col[i].get("order", None)

            if data_type == "datetime":
                col_data.append(
                    {
                        "data_type": data_type,
                        "ordered": ordered,
                        "date_format_list": "%m/%d/%Y, %H:%M:%S",
                    }
                )
            else:
                col_data.append({"data_type": data_type, "ordered": ordered})

        return dg.generate_dataset_by_class(
            rng=rng,
            columns_to_generate=col_data,
        )
