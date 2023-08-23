#!/usr/bin/env python

"""
Test mapping y = f(X)
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from synthetic_data.synthetic_data import make_tabular_data
from synthetic_data.utils import resolve_output_path

output_path = resolve_output_path(pathlib.Path(__file__).parent.absolute())

plt.close("all")


def my_tricontour(ax, x1, x2, y, z, labels=None, title=None, scatter=True, cmap=None):
    if cmap is not None:
        tri2 = ax.tricontourf(x1, x2, z, cmap=cmap)
    else:
        tri2 = ax.tricontourf(x1, x2, z)

    ax.tricontour(x1, x2, z, colors="k")

    if scatter == True:
        scatter = ax.scatter(x1, x2, c=y, label=y, marker=".")
        # leg1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="class")
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    # cbar2 = fig.colorbar(tri2, ax=ax, fraction=0.046, pad=0.04, format="%0.2f")
    # cbar2.formatter.set_powerlimits((0, 0))
    # cbar2.update_ticks()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    # ax.set_aspect("equal", "box")
    ax.set_aspect("equal")

    return


# define expression
expr = "(1 - x1) ** 2 + 10 * (x2 - x1 ** 2) ** 2"

# define mapping from symbols to column of X
col_map = {"x1": 0, "x2": 1}


# define correlations via covariance matrix
cov = np.array([[3.0, 0.0], [0.0, 3.0]])


dist = [{"column": 0, "dist": "norm"}, {"column": 1, "dist": "norm"}]
X, y_reg, y_prob, y_label = make_tabular_data(
    dist=dist,
    n_samples=1000,
    cov=cov,
    col_map=col_map,
    expr=expr,
    p_thresh=0.5,
    #    sig_x0=0.0,
    seed=155,
)

#
# check X
#
print("Correlation coefficients:")
print(np.corrcoef(X, rowvar=False))

h = sns.jointplot(X[:, 0], X[:, 1], kind="hex", stat_func=None)
h.set_axis_labels("x1", "x2", fontsize=16)

#
# check Y
#
# paste together df to hold X, y_reg, y_prob, y_label
df = pd.DataFrame(data=X)
df["y_reg"] = y_reg
df["y_prob"] = y_prob
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
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# h.savefig("pairplot_2D_example.png")


# save the data out to pickle files for modeling/explanations/magic!
# with open("x.pkl", "wb") as f:
#    pickle.dump(X, f)
#
# with open("y_label.pkl", "wb") as f:
#    pickle.dump(y_label, f)


# taking back control of plot styles from seaborn plots above...

SMALL = 14
MEDIUM = 30
LARGE = 40

plt.rc("text", usetex=True)
plt.rc("font", size=MEDIUM)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM)  # legend fontsize
plt.rc("figure", titlesize=LARGE)  # fontsize of the figure title


# check contour levels for attr for x1 and x2
# levels = np.arange(0, 2.2, 0.2)
#
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
# x1 = X[:, 0]
# x2 = X[:, 1]
# z = y_reg
# my_tricontour(ax, x1, x2, y_label, z, labels=["x1", "x2"], title="", cmap='Blues', scatter=False)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# fig.savefig(f"{output_path}/y_reg_contours.png")

# check contour of y_prob vs (x1, x2)
levels = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots(figsize=(8, 8))
X1 = X[:, 0]
X2 = X[:, 1]
z = y_prob
my_tricontour(ax, X1, X2, y_label, z, labels=["x1", "x2"], title="", cmap="Blues")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig(f"{output_path}/rosenbrock.png")
# plt.show()


# linear data set
expr = "x1"

# define mapping from symbols to column of X
col_map = {"x1": 0, "x2": 1}


# define correlations via covariance matrix
cov = np.array([[1.0, 0.0], [0.0, 1.0]])


dist = [{"column": 0, "dist": "norm"}, {"column": 1, "dist": "uniform"}]
X, y_reg, y_prob, y_label = make_tabular_data(
    dist=dist,
    n_samples=1000,
    cov=cov,
    col_map=col_map,
    expr=expr,
    p_thresh=0.5,
    #    sig_x0=0.0,
    seed=111,
)

# check contour of y_prob vs (x1, x2)
levels = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots(figsize=(8, 8))
X1 = X[:, 0]
X2 = X[:, 1]
z = y_prob
my_tricontour(ax, X1, X2, y_label, z, labels=["x1", "x2"], title="", cmap="Blues")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig(f"{output_path}/linear.png")
plt.show()


#
# 2D nonlinear boundary experiment
#

# define expression
# expr = x1 + 2 * x2
# expr = x1 ** 2 + 1.5 * x2 ** 2
# expr = cos(x1 * pi / 180.0) - sin(x2 * pi / 180.0)
expr = "cos(x1 ** 2 * pi / 180.0) - sin(x2 * pi / 180.0) + x1 * x2"

# define mapping from symbols to column of X
col_map = {"x1": 0, "x2": 1}


# define correlations via covariance matrix
cov = np.array([[1.0, 0.0], [0.0, 1.0]])
seed = 1234

X, y_reg, y_prob, y_label = make_tabular_data(
    n_samples=1000, cov=cov, col_map=col_map, expr=expr, p_thresh=0.5, seed=seed
)

# check contour of y_prob vs (x1, x2)
levels = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots(figsize=(8, 8))
X1 = X[:, 0]
X2 = X[:, 1]
z = y_prob
my_tricontour(ax, X1, X2, y_label, z, labels=["x1", "x2"], title="", cmap="Blues")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig(f"{output_path}/2d_nonlinear.png")
plt.show()


# rastrigin
expr = "2*20 + x1**2 + x2**2 - (20*(cos(2*pi*x1) + cos(2*pi*x2)))"
seed = 234


cov = np.array([[0.5, 0.0], [0, 0.5]])
X, y_reg, y_prob, y_label = make_tabular_data(
    n_samples=2000, cov=cov, col_map=col_map, expr=expr, p_thresh=0.7, seed=seed
)


# check contour of y_prob vs (x1, x2)
levels = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots(figsize=(8, 8))
X1 = X[:, 0]
X2 = X[:, 1]
z = y_prob
my_tricontour(ax, X1, X2, y_label, z, labels=["x1", "x2"], title="", cmap="Blues")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig(f"{output_path}/rastrigin.png")
plt.show()
