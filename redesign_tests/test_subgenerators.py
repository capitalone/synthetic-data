import unittest

from redesign.generator_builder import Generator
import pandas as pd
import dataprofiler as dp

class TestSubgenerators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tab_data = dp.Data("redesign_tests/tabular.csv")
        cls.tab_profile = dp.Profiler(tab_data)

        unstruct_data = pd.Series(['first string', 'second string'])
        cls.unstruct_profile = dp.Profiler(unstruct_data, profiler_type='unstructured')   

    def test_synthesize_tab(self):
        result = Generator(self.tab_profile).synthesize(100)
        self.assertEqual(len(result), 100)
    
    def test_synthesize_unstruct(self):
        result = Generator(self.unstruct_profile).run()
        self.assertEqual(result, "Synthesized unstructured data!")
