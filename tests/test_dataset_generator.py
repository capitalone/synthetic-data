"""Contains tests for dataset_generator"""

import unittest
import numpy as np
import pandas as pd
from collections import OrderedDict

from synthetic_data import dataset_generator as dg

class TestDatasetGenerator(unittest.TestCase):
    def test_setup(self):
        pass
        # self.report = 
        # self.data = 

    def test_get_ordered_column(self):

        data = OrderedDict({"int": np.array([5,4,3,2,1]),
                "float": np.array([5.0,4.0,3.0,2.0,1.0]),
                "string": np.array(["cab","bca","abc"]),
                "categorical": np.array(["E","D","C","B","A"]),
                "datetime": np.array(["February 21 2019 12:00:00", 
                                      "February 21 2019 13:00:00", 
                                      "March 21 2019 13:00:00", 
                                      "February 20 2019 13:00:00", 
                                      "February 20 2020 13:00:00"])
                }
                )
        ordered_data = [np.array([1,2,3,4,5]),
                        np.array([1.0,2.0,3.0,4.0,5.0]),
                        np.array(["abc","bca","cab"]),
                        np.array(["A","B","C","D","E"]),
                        np.array(["February 20 2019 13:00:00",
                                  "February 21 2019 12:00:00",
                                  "February 21 2019 13:00:00",
                                  "March 21 2019 13:00:00",
                                  "February 20 2020 13:00:00"
                                  ])
                        ]
        output_data = []
        for data_type in data.keys():
            output_data.append(dg.get_ordered_column(data[data_type], data_type))
        
        for i in range(len(output_data)):
            self.assertTrue(np.array_equal(output_data[i], ordered_data[i]))

    def test_generate_dataset_by_class(self):
        pass