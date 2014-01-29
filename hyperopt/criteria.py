"""Criteria for Bayesian optimization
"""
import numpy as np
import scipy.stats


def EI_empirical(samples, thresh):
    """Expected Improvement over threshold from samples

    (See example usage in EI_gaussian_empirical)
    """
    improvement = np.maximum(samples - thresh, 0)
    return improvement.mean()


def EI_gaussian_empirical(mean, var, thresh, rng, N):
    """Expected Improvement of Gaussian over threshold

    (estimated empirically)
    """
    return EI_empirical(rng.randn(N) * np.sqrt(var) + mean, thresh)


def EI_gaussian(mean, var, thresh):
    """Expected Improvement of Gaussian over threshold

    (estimated analytically)
    """
    sigma = np.sqrt(var)
    score = (mean - thresh) / sigma
    n = scipy.stats.norm
    return sigma * (score * n.cdf(score) + n.pdf(score))


def logEI_gaussian(mean, var, thresh):
    """Return log(EI(mean, var, thresh))

    This formula avoids underflow in cdf for
        thresh >= mean + 37 * sqrt(var)

    """
    sigma = np.sqrt(var)
    score = (mean - thresh) / sigma
    n = scipy.stats.norm
    if score < 0:
        pdf = n.logpdf(score)
        r = np.exp(np.log(-score) + n.logcdf(score) - pdf)
        if r == 1.0:
            return -np.inf
        return np.log(sigma) + pdf + np.log1p(-r)
    else:
        return np.log(sigma) + np.log(score * n.cdf(score) + n.pdf(score))


def UCB(mean, var, zscore):
    """Upper Confidence Bound

    For a model which predicts a Gaussian-distributed outcome, the UCB is

        mean + zscore * sqrt(var)
    """
    return mean + np.sqrt(var) * zscore


# -- flake8
