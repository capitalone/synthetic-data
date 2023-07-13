from distinct_generators import int_generator, text_generator, categorical_generator, float_generator, datetime_generator
import numpy as np


# Given the data, see if we can sort the order of each column/type: 
# by [magnitude, size/alphabetically, priority of categories, magnitude, new/old] respectively

class orderGenerator:
    def __init__(self) -> None:
        self.types = {type("s"), type(1), type(1.0)}
        # self.generator = 
        # self.int_generator = int_generator.random_integers
        # self.text_generator = text_generator.random_text
        # self.cat_generator = categorical_generator.random_categorical
        # self.float_generator = float_generator.random_floats
        # self.datetime_generator = datetime_generator.generate_datetime
    
    def Sort(self, data): # data is resulting np array?

        for col in data.T:
            
