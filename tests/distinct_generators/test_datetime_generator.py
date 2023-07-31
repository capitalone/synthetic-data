import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

import synthetic_data.distinct_generators.datetime_generator as date_generator


class TestGenerateDatetime(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%Y %m %d"]
        self.generate_datetime_output = date_generator.generate_datetime(
            self.rng, self.date_format_list[0], self.start_date, self.end_date
        )
        self.random_datetimes_output = date_generator.random_datetimes(
            self.rng, self.date_format_list, self.start_date, self.end_date, 10
        )

    def test_start_end_date_when_none(self):
        date_str = date_generator.generate_datetime(
            self.rng, self.date_format_list[0], start_date=None, end_date=None
        )
        try:
            pd.to_datetime(date_str, format=self.date_format_list[0])
        except:
            self.fail(
                "pd.to_datetime() raised ValueError for start_date, end_date = None"
            )

    def test_generate_datetime_return_type(self):
        self.assertIsInstance(self.generate_datetime_output, list)
        self.assertIsInstance(self.generate_datetime_output[0], str)
        self.assertIsInstance(self.generate_datetime_output[1], datetime)
        self.assertTrue(self.generate_datetime_output[0] == "2006 10 02")
        self.assertTrue(
            self.generate_datetime_output[1]
            == datetime.strptime(
                self.generate_datetime_output[0], self.date_format_list[0]
            )
        )

    def test_generate_datetime_format(self):
        try:
            pd.to_datetime(
                self.generate_datetime_output[1],
                format=self.generate_datetime_output[0],
            )
        except ValueError:
            self.fail("pd.to_datetime() raised ValueError for custom formatting")

    def test_generate_datetime_range(self):
        date_obj = pd.to_datetime(
            self.generate_datetime_output[1], format=self.generate_datetime_output[0]
        )
        self.assertTrue(self.start_date <= date_obj)
        self.assertTrue(date_obj <= self.end_date)

    def test_random_datetimes_return_type_and_size(self):
        self.assertIsInstance(self.random_datetimes_output, np.ndarray)
        self.assertEqual(self.random_datetimes_output.shape[0], 10)

    def test_random_datetimes_default_format_usage(self):
        dates = date_generator.random_datetimes(
            self.rng, start_date=self.start_date, end_date=self.end_date
        )
        for date in dates:
            try:
                pd.to_datetime(date, format="%B %d %Y %H:%M:%S")
            except ValueError:
                self.fail("pd.to_datetime() raised ValueError for default formatting")

    def test_random_datetimes_format_usage(self):
        date_formats = ["%Y-%m-%d", "%B %d %Y %H:%M:%S"]
        format_success = [False] * len(date_formats)
        for date in self.random_datetimes_output:
            for i in range(len(date_formats)):
                try:
                    pd.to_datetime(date[1], format=date[0])
                    format_success[i] = True
                except ValueError:
                    pass
        self.assertGreater(sum(format_success), 1)

    def test_random_datetimes_output(self):
        outputs = [
            np.array(["2008 08 19", datetime(2008, 8, 19, 0, 0)]),
            np.array(["2018 09 27", datetime(2018, 9, 27, 0, 0)]),
            np.array(["2016 03 11", datetime(2016, 3, 11, 0, 0)]),
            np.array(["2010 03 13", datetime(2010, 3, 13, 0, 0)]),
            np.array(["2008 12 21", datetime(2008, 12, 21, 0, 0)]),
            np.array(["2014 07 22", datetime(2014, 7, 22, 0, 0)]),
            np.array(["2005 11 25", datetime(2005, 11, 25, 0, 0)]),
            np.array(["2016 02 13", datetime(2016, 2, 13, 0, 0)]),
            np.array(["2021 10 11", datetime(2021, 10, 11, 0, 0)]),
            np.array(["2007 03 12", datetime(2007, 3, 12, 0, 0)]),
        ]
        for i in range(len(self.random_datetimes_output)):
            np.testing.assert_array_equal(self.random_datetimes_output[i], outputs[i])
