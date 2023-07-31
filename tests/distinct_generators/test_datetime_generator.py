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

    def test_start_end_date_when_default(self):
        date_str = self.generate_datetime_output
        self.assertEqual("2006 10 02", date_str[0])
        self.assertEqual(
            datetime.strptime("2006 10 02", self.date_format_list[0]), date_str[1]
        )

    def test_generate_datetime(self):
        self.assertEqual(
            ["2006 10 02", datetime.strptime("2006 10 02", self.date_format_list[0])],
            self.generate_datetime_output,
        )
        

class TestRandomDatetimes(unittest.TestCase):
    def setUp(self):
        self.rng = Generator(PCG64(12345))
        self.start_date = pd.Timestamp(2001, 12, 22)
        self.end_date = pd.Timestamp(2023, 1, 1)
        self.date_format_list = ["%Y %m %d"]
        self.random_datetimes_output = date_generator.random_datetimes(
            self.rng, self.date_format_list, self.start_date, self.end_date, 10
        )

    def test_random_datetime_range(self):
        for datetime in self.random_datetimes_output:
            date_timestamp = pd.to_datetime(
                datetime[0], format=self.date_format_list[0]
            )
            self.assertLessEqual(date_timestamp, self.end_date)
            self.assertGreaterEqual(date_timestamp, self.start_date)

    def test_random_datetimes_default_format_usage(self):
        actual = date_generator.random_datetimes(
            self.rng, start_date=self.start_date, end_date=self.end_date, num_rows=5
        )
        expected = np.array(
            [
                ["March 12 2007 12:39:00", datetime(2007, 3, 12, 12, 39)],
                ["December 04 2021 09:46:26", datetime(2021, 12, 4, 9, 46, 26)],
                ["January 02 2016 09:12:26", datetime(2016, 1, 2, 9, 12, 26)],
                ["December 28 2003 11:54:26", datetime(2003, 12, 28, 11, 54, 26)],
                ["April 07 2011 07:53:14", datetime(2011, 4, 7, 7, 53, 14)],
            ]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_random_datetimes_format_usage(self):
        expected = np.array(
            [
                ["December 04 2021 09:46:26", datetime(2021, 12, 4, 9, 46, 26)],
                ["2016-01-02", datetime(2016, 1, 2, 0, 0)],
                ["2011-04-07", datetime(2011, 4, 7, 0, 0)],
                ["2020-08-12", datetime(2020, 8, 12, 0, 0)],
                ["2008-11-02", datetime(2008, 11, 2, 0, 0)],
            ]
        )
        date_formats = ["%Y-%m-%d", "%B %d %Y %H:%M:%S"]
        actual = date_generator.random_datetimes(
            self.rng,
            date_formats,
            start_date=self.start_date,
            end_date=self.end_date,
            num_rows=5,
        )
        np.testing.assert_array_equal(expected, actual)

    def test_random_datetimes_output(self):
        expected = np.array(
            [
                ["2006 10 02", datetime(2006, 10, 2, 0, 0)],
                ["2008 08 19", datetime(2008, 8, 19, 0, 0)],
                ["2018 09 27", datetime(2018, 9, 27, 0, 0)],
                ["2016 03 11", datetime(2016, 3, 11, 0, 0)],
                ["2010 03 13", datetime(2010, 3, 13, 0, 0)],
                ["2008 12 21", datetime(2008, 12, 21, 0, 0)],
                ["2014 07 22", datetime(2014, 7, 22, 0, 0)],
                ["2005 11 25", datetime(2005, 11, 25, 0, 0)],
                ["2016 02 13", datetime(2016, 2, 13, 0, 0)],
                ["2021 10 11", datetime(2021, 10, 11, 0, 0)],
            ]
        )

        self.assertIsInstance(self.random_datetimes_output, np.ndarray)
        self.assertEqual(self.random_datetimes_output.shape[0], 10)
        for i in range(len(self.random_datetimes_output)):
            np.testing.assert_array_equal(self.random_datetimes_output[i], expected[i])
