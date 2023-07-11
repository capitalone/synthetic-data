import numpy as np
from ..base_generator import BaseGenerator
from numpy.random import Generator

class IntGenerator(BaseGenerator):
    def __init__(
            self,
            profile, #dataprofile
            generator, #Random generator
            ) -> None:
        self.min = int(profile.report()["data_stats"][0]["statistics"].get("min", None))
        self.max = int(profile.report()["data_stats"][0]["statistics"].get("max", None))
        self.num_rows = profile.report()["global_stats"].get("column_count", None)
        self.rng = generator

    def synthesize(self):
        """
        Generates random ints within given min and max range
        of the profile's statistics as an np.array

        :rtype: numpy array
        :return: np array of integers
        """
        return self._random_integer_generator(self.rng, self.min, self.max, self.num_rows)

    def _random_integer_generator(self,
                                  rng: Generator,
                                  min_value: int = -1e6,
                                  max_value: int = 1e6,
                                  num_rows: int = 1
                                  ) -> np.array:
        """
        Randomly generates an array of integers between a min and max value

        :param rng: the np rng object used to generate random values
        :type rng: numpy Generator
        :param min_value: the minimum integer that can be returned
        :type min_value: int, optional
        :param max_value: the maximum integer that can be returned
        :type max_value: int, optional
        :param num_rows: the number of rows in np array generated
        :type num_rows: int, optional

        :return: np array of integers
        """
        return rng.integers(min_value, max_value, (num_rows,))