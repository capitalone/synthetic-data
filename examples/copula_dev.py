#!/usr/bin/env python

"""
Using copulas:
    1) use multivariate gaussians to impose correlation structure
    2) 'uniformify' from gaussian
    3) transform from uniform to desired marginal distribution
"""

import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

x = stats.uniform(0, 1).rvs(10000)

# fig, ax = plt.subplots(figsize=(5, 5))
# sns.distplot(x, kde=False, norm_hist=True, ax=ax)
# plt.show()


# bits we need for the probability integral transform

# norm = stats.distributions.norm()
# x_trans = norm.ppf(x)

# fig, ax = plt.subplots(figsize=(5, 5))
# sns.distplot(x_trans, ax=ax)
# plt.show()

# plot together to get a sense of the inverse CDF
# fig, ax = plt.subplots(figsize=(5, 5))
# h = sns.jointplot(x, x_trans, stat_func=None)
# h.set_axis_labels("original", "transformed", fontsize=16)
# plt.show()


# do with arbitrary distribution - like beta
# fig, ax = plt.subplots(figsize=(5, 5))
# beta = stats.distributions.beta(a=10, b=3)
# x_trans = beta.ppf(x)
# h = sns.jointplot(x, x_trans, stat_func=None)
# h.set_axis_labels("original", "transformed", fontsize=16)
# h.ax_marg_y.set_title("uniform to beta")
# plt.show()


# do the inverse - i.e. the inverse of the inverse CDF - or the CDF...
# gumbel = stats.distributions.gumbel_l()
# x_trans_trans = gumbel.cdf(x_trans)
# h = sns.jointplot(x_trans, x_trans_trans, stat_func=None)
# h.set_axis_labels("original", "transformed", fontsize=16)


# create samples from a correlated multivariate normal
n_samples = 10000  # 100000

cov = [[1.0, 0.5, 0.1], [0.5, 1.0, 0.1], [0.1, 0.5, 1.0]]
mvnorm = stats.multivariate_normal(mean=[0, 0, 0], cov=cov)
x = mvnorm.rvs(n_samples)
print("Finished generating multivariate gaussian samples...")

# h = sns.jointplot(x[:, 0], x[:, 1], kind="kde", stat_func=None)
# h.set_axis_labels("x1", "x2", fontsize=16)


# now tranform marginals back to 'uniform'
norm = stats.norm()
x_unif = norm.cdf(x)

print("Finished uniformifying ...ready to transform into final distribution")
h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind="hex", stat_func=None)
h.set_axis_labels("y1", "y2", fontsize=16)
h.ax_marg_y.set_title("hex")
plt.show()
