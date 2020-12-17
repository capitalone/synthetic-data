#!/usr/bin/env python

import pathlib
import pickle

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from synthetic_data.utils import resolve_output_path

output_path = resolve_output_path(pathlib.Path(__file__).parent.absolute())


# load the data
with open("x.pkl", "rb") as f:
    X = pickle.load(f)

with open("y_label.pkl", "rb") as f:
    y = pickle.load(f)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# what seems like an easy choice is not...play with these pairs of activation/initialization
# act1 = "relu"
# init1 = "he_uniform"
#
# act2 = "sigmoid"
# init2 = "glorot_normal"
#
# input = Input(shape=(2,))
# x = Dense(16, activation=act1, kernel_initializer=init1)(input)
# x = Dense(12, activation=act1, kernel_initializer=init1)(input)
# x = Dense(8, activation=act1, kernel_initializer=init1)(x)
# x = Dense(6, activation=act1, kernel_initializer=init1)(x)
# x = Dense(2, activation=act1, kernel_initializer=init1)(x)
# output = Dense(1, activation=act2, kernel_initializer=init2)(x)
#
# clf = Model(input, output)
# clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#
# clf.fit(
#    x_train,
#    y_train,
#    epochs=80,
#    batch_size=100,
#    shuffle=True,
#    validation_data=(x_test, y_test),
# )
#
# clf.save("simple_2d.h5")

clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)


with open("clf.pkl", "wb") as f:
    pickle.dump(clf, f)

# check confusion matrix
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_train)
p_thresh = 0.5  # this should match our data generation parameter
cm_train = confusion_matrix(y_train, y_train_pred >= p_thresh)
print("Confusion matrix on training set - ")
print(cm_train)

# calculate ROC curve and AUC...
# fpr, tpr, _ = roc_curve(y_test, y_test_pred)  # small split - use train tmp
fpr, tpr, _ = roc_curve(y_train, y_train_pred)
roc_auc = auc(fpr, tpr)
print(f"Model AUC = {roc_auc:5.3f}")

plt.figure()
lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (area = {roc_auc:4.3f})"
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.savefig(f"{output_path}/roc_curve.png")
plt.show()
