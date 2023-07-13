#!/usr/bin/env python
"""
Given a tabular dataset, fit a copula to it.
"""

import matplotlib.pyplot as plt
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import compare_3d

df = pd.read_csv("samples.csv")
cols = ["x1", "x2", "x3"]

copula = GaussianMultivariate()
copula.fit(df[cols])


# generate synthetic data from our fit
sd = copula.sample(df.shape[0])


compare_3d(df[cols], sd)
plt.show()
