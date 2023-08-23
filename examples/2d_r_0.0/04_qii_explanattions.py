# /usr/bin/env python3
"""
Load a model ('easy_dense.h5')
Create local attributions using a method (deeplift)
Check that we get rational results
Output a CSV file of explanations for train & test
Pickle our explainer for later use
...we use the same 'explainer' in gradient ascent

NOTE: this had issues running in ipython, runs fine from CLI

"""

import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qii.predictor import QIIPredictor
from qii.qii import QII
from qii.qoi import QuantityOfInterest
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from synthetic_data.utils import resolve_output_path

output_path = resolve_output_path(pathlib.Path(__file__).parent.absolute())


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
np.random.seed(111)
np.set_printoptions(
    precision=3, suppress=True, linewidth=160, formatter={"float": "{: 0.3f}".format}
)


# load the data
with open("x.pkl", "rb") as f:
    X = pickle.load(f)

with open("y_label.pkl", "rb") as f:
    y = pickle.load(f)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = load_model("simple_2d.h5", compile=False)


# baseline = np.array([0.0, 0.0]).reshape(1,2)
nsamples = 25

# TODO - get a grid of fixed points in [x1, x2] - so it's common across r values
background = x_train[np.random.choice(x_train.shape[0], nsamples, replace=False)]

x1_tmp = np.arange(-1, 1, 0.1)
x2_tmp = np.arange(-1, 1, 0.1)
m_grid = np.meshgrid(x1_tmp, x2_tmp, indexing="xy")
background = np.vstack([m_grid[0].ravel(), m_grid[1].ravel()]).T


# explain
# exp = shap.DeepExplainer(model, background)
# local_attr_list = exp.shap_values(x_train)
# local_attr = np.array(local_attr_list).squeeze()
#
# features = ["x1", "x2"]
# attr_df = pd.DataFrame(data=np.array(local_attr), columns=features)
# attr_df.to_csv(f"attr_round1.csv", index=False)


class LRPredictor(QIIPredictor):
    def __init__(self, predictor):
        super().__init__(predictor)

    def predict(self, x):
        # predict the label for instance x
        return self._predictor.predict(x)


lr_predictor = LRPredictor(model)
quantity_of_interest = QuantityOfInterest()

n_features = 2
# qii = QII(background, n_features, quantity_of_interest)
qii = QII(x_train, n_features, quantity_of_interest)

# pick one sample
# x_0 = x_test[0, :].reshape(1, -1)
# shapley_vals = qii.compute(
#    x_0=x_0,
#    predictor=lr_predictor,
#    show_approx=True,
#    evaluated_features=None,
#    data_exhaustive=True,
#    feature_exhaustive=True,
#    method="shapley",
# )
# print("Shapley: \n{0}\n\n".format(shapley_vals))

# loop over all of x_train
shap_list = []
for i, x_0 in enumerate(x_train):
    x_0 = x_0.reshape(1, -1)
    shapley_vals = qii.compute(
        x_0=x_0,
        predictor=lr_predictor,
        show_approx=False,
        evaluated_features=None,
        data_exhaustive=True,
        feature_exhaustive=True,
        method="shapley",
    )
    #    print(f"i: {shapley_vals}")
    shap_list.append(shapley_vals)


features = ["x1", "x2"]
attr_df = pd.DataFrame(data=shap_list)
attr_df.to_csv(f"qii_attr.csv", index=False)
local_attr = attr_df.values

# sys.exit()

# comparison of input vs attributions
#
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
scatter = ax[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train, label=y_train)
leg1 = ax[0].legend(*scatter.legend_elements(), loc="lower right", title="class")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].grid()

scatter2 = ax[1].scatter(local_attr[:, 0], local_attr[:, 1], c=y_train, label=y_train)
# ax[1].plot(background, "k+")
leg2 = ax[1].legend(*scatter2.legend_elements(), loc="lower right", title="class")
ax[1].set_xlabel("x1 attr")
ax[1].set_ylabel("x2 attr")
ax[1].grid()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"{output_path}/qii_scatter_round1.png", bbox_inches="tight")

# check contor levels for correlation
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
tri1 = ax[0].tricontourf(x_train[:, 0], x_train[:, 1], local_attr[:, 0])
scatter = ax[0].scatter(
    x_train[:, 0], x_train[:, 1], c=y_train, label=y_train, marker="."
)
# ax[0].plot(background[:, 0], background[:, 1], "w+")
leg1 = ax[0].legend(*scatter.legend_elements(), loc="lower right", title="class")
cbar1 = fig.colorbar(tri1, ax=ax[0])
ax[0].set_title("x1 attributions")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
# cbar1.set_label('x1 attribution')
cbar1.formatter.set_powerlimits((0, 0))
cbar1.update_ticks()

tri2 = ax[1].tricontourf(x_train[:, 0], x_train[:, 1], local_attr[:, 1])
scatter = ax[1].scatter(
    x_train[:, 0], x_train[:, 1], c=y_train, label=y_train, marker="."
)
leg1 = ax[1].legend(*scatter.legend_elements(), loc="lower right", title="class")
ax[1].set_title("x2 attribution")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
cbar2 = fig.colorbar(tri2, ax=ax[1])
cbar2.formatter.set_powerlimits((0, 0))
cbar2.update_ticks()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"{output_path}/qii_trif_round1.png", bbox_inches="tight")
plt.show()
