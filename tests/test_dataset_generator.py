"""Contains tests for dataset_generator"""

import unittest
from collections import OrderedDict
from datetime import datetime

import dataprofiler as dp
import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from synthetic_data import dataset_generator as dg
from synthetic_data.distinct_generators import datetime_generator as dategen


class TestDatasetGenerator(unittest.TestCase):
    # one test for descending default one test for ascending default
    # other 2 for custom
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.original_format = "%B %d %Y %H:%M:%S"

    # def test_get_ordered_column_datetime_ascending(self):
    #     # generate datetime array (random_datetimes returns string, so need to convert each into a datetime)
    #     data = dategen.random_datetimes(
    #         self.rng, self.original_format, self.start_date, self.end_date, 5
    #     )

    #     # print(data)
    #     # data = np.array(
    #     #         [
    #     #             "February 21 2019 12:00:00",
    #     #             "February 21 2019 13:00:00",
    #     #             "March 21 2019 13:00:00",
    #     #             "February 20 2019 13:00:00",
    #     #             "February 20 2020 13:00:00",
    #     #         ]
    #     #     )
    #     # new_data = np.array([datetime.strptime(dt, "%B %d %Y %H:%M:%S") for dt in data])

    #     # make ordered datetime array
    #     ordered_data = np.array(
    #         [
    #             datetime.strptime("October 02 2006 22:34:32", self.original_format),
    #             datetime.strptime("March 11 2016 15:15:39", self.original_format),
    #             datetime.strptime("September 27 2018 18:24:03", self.original_format),
    #             datetime.strptime("March 13 2010 17:18:44", self.original_format),
    #             datetime.strptime("August 19 2008 16:53:49", self.original_format),
    #         ]
    #     )

    #     ordered_data = np.sort(ordered_data)
    #     for i, time in enumerate(ordered_data):
    #         ordered_data[i] = time.strftime(self.original_format)

    #     # this was sorting as if each item was a string and not datetime. But now i made datetime generator return a datetime object.
    #     output_data = dg.get_ordered_column(data)
    #     for i, time in enumerate(output_data):
    #         output_data[i] = time.strftime(self.original_format)
    #     print(output_data, "GG", ordered_data)
    #     self.assertTrue(np.array_equal(output_data, ordered_data))
    #     # ordered_data = np.array([datetime.strptime(dt, self.original_format) for dt in ordered_data])
    #     # ordered_data = np.array(
    #     #         [
    #     #             "February 20 2019 13:00:00",
    #     #             "February 21 2019 12:00:00",
    #     #             "February 21 2019 13:00:00",
    #     #             "March 21 2019 13:00:00",
    #     #             "February 20 2020 13:00:00",
    #     #         ]
    #     #     )
    #     # for i, time in enumerate(ordered_data):
    #     #     ordered_data[i] = datetime.strptime(time, "%B %d %Y %H:%M:%S")

    #     # output_data = []
    #     # output_data.append(dg.get_ordered_column(data))

    #     # output_data = np.asarray(output_data)

    #     # sort data

    # # def test_get_ordered_column_datetime(self):
    # #     date_format = ["%m/%d/%Y, %H:%M:%S"]
    # #     data = [
    # #         np.array(
    # #             [
    # #                 "February 21 2019 12:00:00",
    # #                 "February 21 2019 13:00:00",
    # #                 "March 21 2019 13:00:00",
    # #                 "February 20 2019 13:00:00",
    # #                 "February 20 2020 13:00:00",
    # #             ]
    # #         ),
    # #         np.array(
    # #             [
    # #                 "12/26/2018, 04:34:52",
    # #                 "11/26/2018, 04:34:52",
    # #                 "12/27/2018, 04:34:52",
    # #                 "12/26/2017, 04:34:52",
    # #                 "12/26/2018, 04:34:56",
    # #             ]
    # #         ),
    # #     ]
    # #     for date_array in data:
    # #         date_array = date_object = np.array([datetime.strptime(dt, ) for dt in data])
    # #     ordered_data = [
    # #         np.array(
    # #             [
    # #                 "February 20 2019 13:00:00",
    # #                 "February 21 2019 12:00:00",
    # #                 "February 21 2019 13:00:00",
    # #                 "March 21 2019 13:00:00",
    # #                 "February 20 2020 13:00:00",
    # #             ]
    # #         ),
    # #         np.array(
    # #             [
    # #                 "12/26/2017, 04:34:52",
    # #                 "11/26/2018, 04:34:52",
    # #                 "12/26/2018, 04:34:52",
    # #                 "12/26/2018, 04:34:56",
    # #                 "12/27/2018, 04:34:52",
    # #             ]
    # #         ),
    # #     ]

    # #     output_data = []
    # #     output_data.append(dg.get_ordered_column(data[0], "datetime"))
    # #     output_data.append(dg.get_ordered_column(data[1], "datetime", date_format))
    # #     print(output_data, "GG", ordered_data)
    # #     self.assertTrue(np.array_equal(output_data, ordered_data))

    # # def test_get_ordered_column_datetime_descending(self):
    # #     date_format = "%m/%d/%Y, %H:%M:%S"
    # #     data = [
    # #         np.array(
    # #             [
    # #                 "February 20 2019 13:00:00",
    # #                 "February 21 2019 12:00:00",
    # #                 "February 21 2019 13:00:00",
    # #                 "March 21 2019 13:00:00",
    # #                 "February 20 2020 13:00:00",
    # #             ]
    # #         ),
    # #         np.array(
    # #             [
    # #                 "12/26/2017, 04:34:52",
    # #                 "11/26/2018, 04:34:52",
    # #                 "12/26/2018, 04:34:52",
    # #                 "12/26/2018, 04:34:56",
    # #                 "12/27/2018, 04:34:52",
    # #             ]
    # #         ),
    # #     ]
    # #     ordered_data = [
    # #         np.array(
    # #             [
    # #                 "February 20 2020 13:00:00",
    # #                 "March 21 2019 13:00:00",
    # #                 "February 21 2019 13:00:00",
    # #                 "February 21 2019 12:00:00",
    # #                 "February 20 2019 13:00:00",
    # #             ]
    # #         ),
    # #         np.array(
    # #             [
    # #                 "12/27/2018, 04:34:52",
    # #                 "12/26/2018, 04:34:56",
    # #                 "12/26/2018, 04:34:52",
    # #                 "11/26/2018, 04:34:52",
    # #                 "12/26/2017, 04:34:52",
    # #             ]
    # #         ),
    # #     ]
    # #     output_data = []
    # #     output_data.append(
    # #         dg.get_ordered_column(data[0], "datetime", order="descending")
    # #     )
    # #     output_data.append(
    # #         dg.get_ordered_column(data[1], "datetime", date_format, order="descending")
    # #     )
    # #     print(output_data, "GG", ordered_data)
    # #     self.assertTrue(np.array_equal(output_data, ordered_data))

    def test_get_ordered_column(self):

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
        ordered_data = np.asarray(ordered_data)

        output_data = []
        for data_type in data.keys():
            output_data.append(dg.get_ordered_column(data[data_type], data_type))
        output_data = np.asarray(output_data)

        self.assertTrue(np.array_equal(output_data, ordered_data))
