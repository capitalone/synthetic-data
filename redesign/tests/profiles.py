'''
Contains instantiated structured and unstructured profiles
that will be used as inputs in unit tests.
'''

import dataprofiler as dp
import pandas as pd

tab_data = dp.Data("./tabular.csv")
tab_profile = dp.Profiler(tab_data)

unstruct_data = pd.Series(['first string', 'second string'])
unstruct_profile = dp.Profiler(unstruct_data, profiler_type='unstructured')