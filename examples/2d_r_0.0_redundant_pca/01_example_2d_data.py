#!/usr/bin/env python

"""
Test mapping y = f(X)
"""

import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from synthetic_data.synthetic_data import make_tabular_data
from synthetic_data.utils import resolve_output_path

output_path = resolve_output_path(pathlib.Path(__file__).parent.absolute())


# import sys  # isort:skip


# sys.path.insert(0, "../../")  # isort:skip


# define expression
# expr = x1 + 2 * x2
# expr = x1 ** 2 + 1.5 * x2 ** 2
# expr = cos(x1 * pi / 180.0) - sin(x2 * pi / 180.0)
expr = "cos(x1 ** 2 * pi / 180.0) - sin(x2 * pi / 180.0) + x1 * x2"

# define mapping from symbols to column of X
col_map = {"x1": 0, "x2": 1}


# define correlations via covariance matrix
cov = np.array([[1.0, 0.0], [0.0, 1.0]])


n_nuisance = 0
n_redundant = 2
X, y_reg, y_prob, y_label = make_tabular_data(
    n_samples=1000,
    n_redundant=2,
    n_nuisance=n_nuisance,
    cov=cov,
    col_map=col_map,
    expr=expr,
    p_thresh=0.5,
)


#
# check X
#
print("Correlation coefficients:")
print(np.corrcoef(X, rowvar=False))

h = sns.jointplot(X[:, 0], X[:, 1], kind="hex", stat_func=None)
h.set_axis_labels("x1", "x2", fontsize=16)
h.savefig(f"{output_path}/joint_dist_plot.png")

#
# check Y
#
# paste together df to hold X, y_reg, y_prob, y_label
df = pd.DataFrame(data=X)
# df["y_reg"] = y_reg
# df["y_prob"] = y_prob
df["label"] = y_label

# summary plot
h = sns.set(style="ticks", color_codes=True)
h = sns.pairplot(
    df,
    vars=[0, 1],
    hue="label",
    markers=[".", "."],
    diag_kind="kde",
    diag_kws={"alpha": 0.5, "clip": (-1, 1)},
)
# plt.show()
h.savefig(f"{output_path}/pairplot_2D_example.png")


# save the data out to pickle files for modeling/explanations/magic!
with open("x.pkl", "wb") as f:
    pickle.dump(X, f)

with open("y_label.pkl", "wb") as f:
    pickle.dump(y_label, f)

# check contour of y_reg vs (x1, x2)
levels = np.arange(0, 2.2, 0.2)
fig, ax = plt.subplots(figsize=(8, 8))
tri1 = ax.tricontourf(X[:, 0], X[:, 1], y_reg, levels=levels)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_label, label=y_label, marker=".")
leg1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="class")
cbar1 = fig.colorbar(tri1, ax=ax)
ax.set_title("y_reg contours")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
cbar1.formatter.set_powerlimits((0, 0))
cbar1.update_ticks()
fig.savefig(f"{output_path}/y_reg_contours.png")

# check contour of y_prob vs (x1, x2)
levels = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots(figsize=(8, 8))
tri1 = ax.tricontourf(X[:, 0], X[:, 1], y_prob, levels=levels)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_label, label=y_label, marker=".")
leg1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="class")
cbar1 = fig.colorbar(tri1, ax=ax)
ax.set_title("y_prob contours")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
cbar1.formatter.set_powerlimits((0, 0))
cbar1.update_ticks()
fig.savefig(f"{output_path}/y_prob_contours.png")


plt.show()


# try PCA decomp before passing to model
pca = PCA(n_components=2)
pca.fit(X.T)
PCA(n_components=2)
X_pca = pca.components_.T


# save the data out to pickle files for modeling/explanations/magic!
with open("x_pca.pkl", "wb") as f:
    pickle.dump(X_pca, f)
