"""Test stuff."""
import dataprofiler as dp
import numpy as np
import pandas as pd
from sklearn import datasets

from redesign.generator_builder import Generator

iris = datasets.load_iris()
data = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
data.target = data.target.astype(int)

profile_options = dp.ProfilerOptions()
profile_options.set(
    {
        "data_labeler.is_enabled": False,
        "correlation.is_enabled": True,
        "multiprocess.is_enabled": False,
    }
)
profile = dp.Profiler(data, options=profile_options)

generator = Generator(profile)
synth_data = generator.synthesize(100)

print(synth_data)
print(len(synth_data))