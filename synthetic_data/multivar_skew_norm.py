import numpy as np
from scipy.stats import multivariate_normal as mvn, norm

class MultivariateSkewNorm():
    '''
    Class to sample data from a Multivariate Skew-Normal Distribution
    
    Based on,
    (1) "A. AZZALINI, A. DALLA VALLE, The multivariate skew-normal distribution, Biometrika, Volume 83, Issue 4, December 1996, Pages 715–726, https://doi.org/10.1093/biomet/83.4.715"
    (2) "Azzalini, A. and Capitanio, A. (1999), Statistical applications of the multivariate skew normal distribution. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61: 579-602. https://doi.org/10.1111/1467-9868.00194"

    Snippets of code adapted from,
    http://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/ 

    Args:
        cov (np.ndarray): 2D Covariance matrix 
        skew (np.ndarray): 1D Array of skew values of each column from the dataset/DataProfiler report

    '''
    def __init__(self, cov, skew):
        self.cov = np.asarray(cov)
        self.skew = np.asarray(skew)
        self.dim = len(self.skew)

        self.delta = self._calc_delta()
        # self.capital_delta represents a diagonal matrix. Storing only the diagonal values for space efficiency.
        # If used in computation convert to diagonal matrix first by, capital_delta = np.diag(self.capital_delta)
        self.capital_delta = self._calc_capital_delta()
        self.omega = self._calc_omega()

        # Calculate only when it is needed.
        self.alpha = None

    def _calc_omega(self):
        """
        Calculate omega matrix as described in equation 2.6 in (Azzalini & Dalla Valle, 1996)
        """
        capital_delta = np.diag(self.capital_delta)
        skew = np.atleast_2d(self.skew)
        return capital_delta @ (self.cov + skew.T @ skew) @ capital_delta

    def _calc_delta(self):
        """
        Calculate array of delta values as a function of skew values as described in equation 1.3 in (Azzalini & Dalla Valle, 1996)
        """
        skew_squared = np.square(self.skew)
        bottom = np.sqrt(1+skew_squared)
        return self.skew / bottom

    def _calc_capital_delta(self):
        """
        Calculate array of uppercase delta values as a function of delta values as described in equation 2.4 in (Azzalini & Dalla Valle, 1996).
        """
        delta_squared = np.square(self.delta)
        diag_vals = np.sqrt(1-delta_squared)
        return diag_vals

    def _calc_alpha(self):
        """
        Calculate array of alpha as described in equation 2.4 in (Azzalini & Dalla Valle, 1996)
        """
        capital_delta = np.diag(self.capital_delta)
        inv_cov = np.linalg.inv(self.cov)
        skew_inv_cov = self.skew @ inv_cov
        top = skew_inv_cov @ np.linalg.inv(capital_delta)
        bottom = np.sqrt(1+(skew_inv_cov @ self.skew))
        alpha = top/bottom
        return alpha.flatten()


    def rvs(self, n_samples: int=1):
        """
        Draw random samples from a Multivariate Skew-Normal Distribution

        Args:
            n_samples (int): Number of random samples to draw
        """
        assert type(n_samples) is int and n_samples > 0, "Number of samples to generate must be a positive integer"
        omega_star = np.block([[np.ones(1),     self.delta],
                            [self.delta[:, None], self.omega]])
        x        = mvn(np.zeros(self.dim+1), omega_star).rvs(n_samples)
        x0, x1   = x[:, 0], x[:, 1:]
        inds     = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1

    def pdf(self, x):
        """
        pdf of the distribution. Mainly for testing whether distribution samples match the pdf contour.
        """
        mean = np.zeros(self.dim)
        if self.alpha is None:
            self.alpha = self._calc_alpha()

        pdf = mvn(mean, self.omega).pdf(x)
        cdf = norm(0, 1).cdf(x @ self.alpha)
        return 2*np.multiply(pdf, cdf)