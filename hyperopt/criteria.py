"""Criteria for Bayesian optimization
"""
import numpy as np
import scipy.stats


def EI_numeric(samples, thresh):
    """Expected improvement over threshold from samples
    """
    samples = samples - thresh
    return samples[samples > 0].sum() / float(len(samples))


def EI(mean, var, thresh):
    """Expected improvement of Gaussian over threshold
    """
    sigma = np.sqrt(var)
    score = (mean - thresh) / sigma
    n = scipy.stats.norm
    return sigma * (score * n.cdf(score) + n.pdf(score))


def logEI(mean, var, thresh):
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
        return np.log(sigma) + pdf + np.log1p(-r)
    else:
        return np.log(sigma) + np.log(score * n.cdf(score) + n.pdf(score))


def UCB(mean, var, zscore):
    return mean - np.sqrt(var) * zscore


# -- flake8
