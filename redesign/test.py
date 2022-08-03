import dataprofiler as dp
import generator_builder as gb

profile = dp.Profiler(dp.Data("./test.csv"))
data = gb.Generator(profile).synthesize()
print(data)