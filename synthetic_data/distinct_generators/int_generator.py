from numpy.random import Generator
import numpy as np

# this generator will create the synthetic data
# generator = Generator(profile)
# df = generator.synthesize()


# >> df.head()
# this will contain the report



class Generator():
    
    def synthesize(self):
        return dataframe

class DatasetGenerator:
    def __init__(self) -> None:
        pass

    def synthesize(self):
        build_dataframe = pd.DataFrame()
        for column in profile.report()['data_stats']:
            build_dataframe.append(column['data_type'].__init__(, column_data))

        return build_dataframe

    

class IntGenerator(BaseGenerator):
    def __init__(
            self,
            profile_column_dict,
            ) -> None:
        self.profile_dict = profile_column_dict
        self.min = profile_column_dict.get("min", None)
        self.max = profile_column_dict.get("max", None)

    def synthesize(self):
        return self._random_integer_generator(self.min, self.max)

    def _random_integer_generator(self, min, max)    
        return rng.integers(min_value, max_value, (num_rows,))
        

def random_integers(
    rng: Generator, min_value: int = -1e6, max_value: int = 1e6, num_rows: int = 1
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