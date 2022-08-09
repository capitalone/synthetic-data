import unittest

from profiles import tab_profile, unstruct_profile
import generator_builder as gb

data = dp.Data("./test.csv")
profile = dp.Profiler(data)
generator = gb.Generator(profile)
synth = generator.synthesize()
print(synth)