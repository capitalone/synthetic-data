import unittest
from unittest import mock
from numpy.random import Generator, PCG64
import pandas as pd
import numpy as np
import synthetic_data.dataset_generators.datetime_generator as dtg


class TestDatetimeFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%Y-%m-%d", "%d-%m-%Y"]

    def test_generate_datetime_return_type(self):
        date_str = dtg.generate_datetime(self.rng, 
                                        self.date_format_list[0],
                                        self.start_date,
                                        self.end_date)
        self.assertIsInstance(date_str, str)

    def test_generate_datetime_format(self):
        date_str = dtg.generate_datetime(self.rng, 
                                        self.date_format_list[0],
                                        self.start_date,
                                        self.end_date)
        try:
            pd.to_datetime(date_str, format=self.date_format_list[0])
        except ValueError:
            self.fail("pd.to_datetime() raised ValueError unexpectedly")

    def test_generate_datetime_range(self):
        date_str = dtg.generate_datetime(self.rng, 
                                        self.date_format_list[0],
                                        self.start_date,
                                        self.end_date)
        date_obj = pd.to_datetime(date_str, format = self.date_format_list[0])
        self.assertTrue(self.start_date <= date_obj)
        self.assertTrue(date_obj <= self.end_date)

    def test_random_datetimes_return_type_and_size(self):
        result = dtg.random_datetimes(self.rng,
                                      self.date_format_list,
                                      self.start_date,
                                      self.end_date,
                                      5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 5)

    def test_random_datetimes_default_format_usage(self):
        result = dtg.random_datetimes(self.rng,
                                      None,
                                      self.start_date,
                                      self.end_date,
                                      10)     
        for date_str in result:
            try:
                pd.to_datetime(date_str, format = "%B %d %Y %H:%M:%S")
            except ValueError:
                self.fail("pd.to_datetime() raised ValueError unexpectedly")

    def test_random_datetimes_format_usage(self):
        result = dtg.random_datetimes(self.rng,
                                      self.date_format_list,
                                      self.start_date,
                                      self.end_date,
                                      10)
        format_success = [False] * len(self.date_format_list)
        for date_str in result:
            for i, date_format in enumerate(self.date_format_list):
                try:
                    pd.to_datetime(date_str, format = date_format)
                    format_success[i] = True
                except ValueError:
                    pass
        self.assertGreater(sum(format_success), 1)
        