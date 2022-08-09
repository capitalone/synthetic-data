import unittest

from profiles import tab_profile, unstruct_profile
from redesign.generator_builder import Generator
#from ..redesign import generator_builder as gb

class TestSubgenerators(unittest.TestCase):

    def test_synthesize_tab(self):
        result = Generator(tab_profile).synthesize()
        self.assertEqual(result, "Synthesized tabular data!")
    
    def test_synthesize_unstruct(self):
        result = Generator(unstruct_profile).synthesize()
        self.assertEqual(result, "Synthesized unstructured data!")