import numpy as np

class MultivariateSkewNorm():
    '''
    Class to sample data from a Multivariate Skew-Normal Distribution
    
    Based on,
    (1) "A. AZZALINI, A. DALLA VALLE, The multivariate skew-normal distribution, Biometrika, Volume 83, Issue 4, December 1996, Pages 715–726, https://doi.org/10.1093/biomet/83.4.715"
    (2) "Azzalini, A. and Capitanio, A. (1999), Statistical applications of the multivariate skew normal distribution. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61: 579-602. https://doi.org/10.1111/1467-9868.00194"

    Snippets of code adapted from,
    http://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/ 

    Args:
        corr (np.ndarray): 2D Correlation matrix
        cov (np.ndarray): 2D Covariance matrix 
        skew (np.ndarray): 1D Array of skew values of each column from the dataset/DataProfiler report

    TODO:
    - Convert skews to shape
    - Write method to sample from distribution
    - Error checking?
    '''
    def __init__(self, corr: np.ndarray, cov: np.ndarray, skew: np.ndarray):
        self.corr = corr
        self.cov = cov
        self.skew = skew
        self.shape = self._gen_shape()
    
    def _gen_shape():
        """
        Generate shape array (alpha) as described in (Azzalini & Dalla Valle, 1996)
        """
        pass
    
    def rvs(n_samples: int=1):
        """
        Draw random samples from a Multivariate Skew-Normal Distribution

        Args:
            n_samples (int): Number of random samples to draw
        """
        assert type(n_samples) is int and n_samples > 0, "Number of samples to generate must be a positive integer"
        pass