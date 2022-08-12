A data profile is a statistical overview of a raw data file. Each profile describes a particular type of raw data (tabular, graph, or unstructured) and therefore must be treated differently in the synthetic data generation process. This API is designed to differentially synthesize data in object-oriented fashion (i.e., with classes) based on the type of data described by its input profile.

The API implementation is split into three components: 1) Base class (i.e., the blueprint for each generator), three generators (tabular, graph, and unstructured), and 3) a generator factory class (i.e., a class that picks the right generator for a given data profile).

To test the generator, run the following script:

```
import dataprofiler as dp
import generator_builder as gb

profile = dp.Profiler(dp.Data("./test.csv"))
data = gb.Generator(profile).synthesize()
print(data)
```

If the terminal prints "Synthesized tabular data!", the API is working properly (since test.csv is a tabular data file and each generator subclass currently prints the type of data it would generate based on the input profile).

Future roadblocks include adding appropriate docstrings, implementation for actual data synthesis, and security features like differential privacy. Use cases include training ML models and obtaining data without the security risks associated with reaching across departments/teams/LOBs.
