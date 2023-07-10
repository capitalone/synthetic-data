from numpy.random import Generator
import numpy as np
from base_generator import BaseGenerator

# this generator will create the synthetic data
# generator = Generator(profile)
# df = generator.synthesize()


# >> df.head()
# this will contain the report
# which is then used in the primitive type generators to generate synthetic data

# plan of attack -- Implement the class first. Then implement the test



class Generator():
    def __init__(self, profile) -> None:
        self.profile = profile

    def synthesize(self):
        return self.profile.dataframe

class DatasetGenerator:
    def __init__(self) -> None:
        pass

    def synthesize(self):
        build_dataframe = pd.DataFrame()
        for column in profile.report()['data_stats']:
            build_dataframe.append(column['data_type']).__init__(, column_data)

        return build_dataframe

class IntGenerator(BaseGenerator):
    def __init__(
            self,
            profile_column_dict, #DP["data_stats"][0]["statistics"] 
            ) -> None:
        self.profile_dict = profile_column_dict
        self.min = profile_column_dict.get("min", None)
        self.max = profile_column_dict.get("max", None)

    def synthesize(self):
        """
        Given the min and max of the profile's statistics, 
        uses _random_integer_generator to generate a random int np.array
        
        :rtype: numpy array 
        :return: np array of integers
        """
        return self._random_integer_generator(self.min, self.max)

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
