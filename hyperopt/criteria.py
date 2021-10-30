"""Criteria for Bayesian optimization
"""
from past.utils import old_div
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
    return EI_empirical(rng.standard_normal(N) * np.sqrt(var) + mean, thresh)


def EI_gaussian(mean, var, thresh):
    """Expected Improvement of Gaussian over threshold

    (estimated analytically)
    """
    sigma = np.sqrt(var)
    score = old_div((mean - thresh), sigma)
    n = scipy.stats.norm
    return sigma * (score * n.cdf(score) + n.pdf(score))


def logEI_gaussian(mean, var, thresh):
    """Return log(EI(mean, var, thresh))

    This formula avoids underflow in cdf for
        thresh >= mean + 37 * sqrt(var)

    """
    assert np.asarray(var).min() >= 0
    sigma = np.sqrt(var)
    score = old_div((mean - thresh), sigma)
    n = scipy.stats.norm
    try:
        float(mean)
        is_scalar = True
    except TypeError:
        is_scalar = False

    if is_scalar:
        if score < 0:
            pdf = n.logpdf(score)
            r = np.exp(np.log(-score) + n.logcdf(score) - pdf)
            rval = np.log(sigma) + pdf + np.log1p(-r)
            if not np.isfinite(rval):
                return -np.inf
            else:
                return rval
        else:
            return np.log(sigma) + np.log(score * n.cdf(score) + n.pdf(score))
    else:
        score = np.asarray(score)
        rval = np.zeros_like(score)

        olderr = np.seterr(all="ignore")
        try:
            negs = score < 0
            nonnegs = np.logical_not(negs)
            negs_score = score[negs]
            negs_pdf = n.logpdf(negs_score)
            r = np.exp(np.log(-negs_score) + n.logcdf(negs_score) - negs_pdf)
            rval[negs] = np.log(sigma[negs]) + negs_pdf + np.log1p(-r)
            nonnegs_score = score[nonnegs]
            rval[nonnegs] = np.log(sigma[nonnegs]) + np.log(
                nonnegs_score * n.cdf(nonnegs_score) + n.pdf(nonnegs_score)
            )
            rval[np.logical_not(np.isfinite(rval))] = -np.inf
        finally:
            np.seterr(**olderr)
        return rval


def UCB(mean, var, zscore):
    """Upper Confidence Bound

    For a model which predicts a Gaussian-distributed outcome, the UCB is

        mean + zscore * sqrt(var)
    """
    return mean + np.sqrt(var) * zscore


# -- flake8
