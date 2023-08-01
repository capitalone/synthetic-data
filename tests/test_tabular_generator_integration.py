import unittest
from unittest import mock
import os
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator
from collections import OrderedDict
from synthetic_data.generators import TabularGenerator
import synthetic_data.dataset_generator as dg
import dataprofiler as dp

class TestDatetimeFunctions(unittest.TestCase):
    def setUp(self):
        self.profile_options = dp.ProfilerOptions()
        self.profile_options.set(
            {
                "data_labeler.is_enabled": False,
                "correlation.is_enabled": True,
                "multiprocess.is_enabled": False,
            }
        )
        test_dir = os.path.abspath('tests')
        # create dataset and profile for tabular
        self.data = dp.Data(os.path.join(test_dir, "data/iris.csv"))
        self.columns = dp.Profiler(
            data=self.data, profiler_type="structured", options=self.profile_options
        ).report()["data_stats"]

    @mock.patch('synthetic_data.generators.make_data_from_report')
    def test_synthesize_correlated_method(self, mock_make_data):
        profile = dp.Profiler(
            data=self.data, options=self.profile_options, samples_per_update=len(self.data)
        )
        instance = TabularGenerator(profile)
        instance.method = "correlated"
        instance.synthesize(num_samples=10)
        mock_make_data.assert_called_once()

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

    # def test_generate_dataset_columns_format_list(self):
    #     random_seed = 0
    #     rng = np.random.default_rng(seed=random_seed)
    #     col_data = []

    #     for i, col in enumerate(self.columns):
    #         data_type = col[i].get("data_type", None)
    #         ordered = col[i].get("order", None)

    #         if data_type == "datetime":
    #             col_data.append({"data_type":data_type, "ordered":ordered, "date_format_list": "%m/%d/%Y, %H:%M:%S"})
    #         else:
    #             col_data.append({"data_type":data_type, "ordered":ordered})

    #     return dg.generate_dataset_by_class(
    #         rng=rng,
    #         columns_to_generate=col_data,
    #     )
