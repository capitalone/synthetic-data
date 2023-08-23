# Roadmap
Laying out feature enhancements to add.

### Features:
Inputs:
- [x] scaled inputs using other preprocessing scaler
- [ ] redundant (correlated and dependent - say by a linear combo of informative features)

- [ ] separation between classes (can we filter +/- k% on either side of p_thresh to create separation?)
- [ ] overlap - since we have ground truth probabilities, we could sample from a binomial distribution with probability of (py|x) to determine labels - this would work in conjuction with sig_k which controls the steepness of the sigmoid
- [ ] noise level - apply *during* generation of regression values/labels
  - [ ] sample coefficients of symbolic expression from std normal distribution
- [ ] outlier generation
- [ ] create fake PII with [pydbgen](https://github.com/tirthajyoti/pydbgen)  (stretch, *new*)

Output:
- [ ] mapping from y_reg value to y_class
    - [ ] partition and label - e.g. `y_class = y_reg < np.median(y_reg)`
    - [ ] Gompertz curve (a parameterized sigmoid - would give control over uncertainty?
  -[ ] noise (e.g. `flip_y`)
  -[ ] map class to probability using random draw from binomial distribution

Capability/Scaling:
- [ ] test scaling complexity based on number of features
- [ ] research alternatives to multivariate normal copula
