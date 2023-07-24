import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

import synthetic_data.distinct_generators.datetime_generator as date_generator


class TestDatetimeFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%Y %m %d"]

    def test_generate_datetime_return_type(self):
        date = date_generator.generate_datetime(
            self.rng, self.date_format_list[0], self.start_date, self.end_date
        )
        self.assertIsInstance(date, list)
        self.assertIsInstance(date[0], str)
        self.assertIsInstance(date[1], datetime)

    def test_generate_datetime_format(self):
        date = date_generator.generate_datetime(
            self.rng, self.date_format_list[0], self.start_date, self.end_date
        )
        try:
            pd.to_datetime(date[1], format=date[0])
        except ValueError:
            self.fail("pd.to_datetime() raised ValueError unexpectedly")

    def test_generate_datetime_range(self):
        date = date_generator.generate_datetime(
            self.rng, self.date_format_list[0], self.start_date, self.end_date
        )
        date_obj = pd.to_datetime(date[1], format=date[0])
        self.assertTrue(self.start_date <= date_obj)
        self.assertTrue(date_obj <= self.end_date)

    def test_random_datetimes_return_type_and_size(self):
        date = date_generator.random_datetimes(
            self.rng, self.date_format_list, self.start_date, self.end_date, 5
        )
        self.assertIsInstance(date, np.ndarray)
        self.assertEqual(date.shape[0], 5)

    def test_random_datetimes_default_format_usage(self):
        dates = date_generator.random_datetimes(
            self.rng, None, self.start_date, self.end_date, 10
        )
        for date in dates:
            try:
                pd.to_datetime(date[1], format=date[0])
            except ValueError:
                self.fail("pd.to_datetime() raised ValueError unexpectedly")

    def test_random_datetimes_format_usage(self):
        date_formats = ["%Y-%m-%d", "%B %d %Y %H:%M:%S"]
        dates = date_generator.random_datetimes(
            self.rng, self.date_format_list, self.start_date, self.end_date, 10
        )
        format_success = [False] * len(date_formats)
        for date in dates:
            for i in range(len(date_formats)):
                try:
                    pd.to_datetime(date[1], format=date[0])
                    format_success[i] = True
                except ValueError:
                    pass
        self.assertGreater(sum(format_success), 1)
