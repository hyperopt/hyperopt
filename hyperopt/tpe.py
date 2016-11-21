"""
Graphical model (GM)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging
import time

import numpy as np
from scipy.special import erf
import pyll
from pyll import scope
from pyll.stochastic import implicit_stochastic

from .base import miscs_to_idxs_vals
from .base import miscs_update_idxs_vals
from .base import Trials
import rand

logger = logging.getLogger(__name__)

EPS = 1e-12

# -- default linear forgetting. don't try to change by writing this variable
# because it's captured in function default args when this file is read
DEFAULT_LF = 25


adaptive_parzen_samplers = {}


def adaptive_parzen_sampler(name):
    def wrapper(f):
        assert name not in adaptive_parzen_samplers
        adaptive_parzen_samplers[name] = f
        return f
    return wrapper


#
# These are some custom distributions
# that are used to represent posterior distributions.
#

# -- Categorical

@scope.define
def categorical_lpdf(sample, p, upper):
    """
    """
    if sample.size:
        return np.log(np.asarray(p)[sample])
    else:
        return np.asarray([])


# -- Bounded Gaussian Mixture Model (BGMM)

@implicit_stochastic
@scope.define
def GMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None,
        size=()):
    """Sample from truncated 1-D Gaussian Mixture Model"""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    assert len(weights) == len(mus) == len(sigmas)
    n_samples = np.prod(size)
    #n_components = len(weights)
    if low is None and high is None:
        # -- draw from a standard GMM
        active = np.argmax(rng.multinomial(1, weights, (n_samples,)), axis=1)
        samples = rng.normal(loc=mus[active], scale=sigmas[active])
    else:
        # -- draw from truncated components
        # TODO: one-sided-truncation
        low = float(low)
        high = float(high)
        if low >= high:
            raise ValueError('low >= high', (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(draw)
    samples = np.reshape(np.asarray(samples), size)
    #print 'SAMPLES', samples
    if q is None:
        return samples
    else:
        return np.round(samples / q) * q


@scope.define
def normal_cdf(x, mu, sigma):
    top = (x - mu)
    bottom = np.maximum(np.sqrt(2) * sigma, EPS)
    z = top / bottom
    return 0.5 * (1 + erf(z))


@scope.define
def GMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    verbose = 0
    samples, weights, mus, sigmas = map(np.asarray,
            (samples, weights, mus, sigmas))
    if samples.size == 0:
        return np.asarray([])
    if weights.ndim != 1:
        raise TypeError('need vector of weights', weights.shape)
    if mus.ndim != 1:
        raise TypeError('need vector of mus', mus.shape)
    if sigmas.ndim != 1:
        raise TypeError('need vector of sigmas', sigmas.shape)
    assert len(weights) == len(mus) == len(sigmas)
    _samples = samples
    samples = _samples.flatten()

    if verbose:
        print 'GMM1_lpdf:samples', set(samples)
        print 'GMM1_lpdf:weights', weights
        print 'GMM1_lpdf:mus', mus
        print 'GMM1_lpdf:sigmas', sigmas
        print 'GMM1_lpdf:low', low
        print 'GMM1_lpdf:high', high
        print 'GMM1_lpdf:q', q

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
                weights * (
                    normal_cdf(high, mus, sigmas)
                    - normal_cdf(low, mus, sigmas)))

    if q is None:
        dist = samples[:, None] - mus
        mahal = (dist / np.maximum(sigmas, EPS)) ** 2
        # mahal shape is (n_samples, n_components)
        Z = np.sqrt(2 * np.pi * sigmas ** 2)
        coef = weights / Z / p_accept
        rval = logsum_rows(- 0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros(samples.shape, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + q / 2.0
            else:
                ubound = np.minimum(samples + q / 2.0, high)
            if low is None:
                lbound = samples - q / 2.0
            else:
                lbound = np.maximum(samples - q / 2.0, low)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * normal_cdf(ubound, mu, sigma)
            inc_amt -= w * normal_cdf(lbound, mu, sigma)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)

    if verbose:
        print 'GMM1_lpdf:rval:', dict(zip(samples, rval))

    rval.shape = _samples.shape
    return rval


# -- Mixture of Log-Normals

@scope.define
def lognormal_cdf(x, mu, sigma):
    # wikipedia claims cdf is
    # .5 + .5 erf( log(x) - mu / sqrt(2 sigma^2))
    #
    # the maximum is used to move negative values and 0 up to a point
    # where they do not cause nan or inf, but also don't contribute much
    # to the cdf.
    if len(x) == 0:
        return np.asarray([])
    if x.min() < 0:
        raise ValueError('negative arg to lognormal_cdf', x)
    olderr = np.seterr(divide='ignore')
    try:
        top = np.log(np.maximum(x, EPS)) - mu
        bottom = np.maximum(np.sqrt(2) * sigma, EPS)
        z = top / bottom
        return .5 + .5 * erf(z)
    finally:
        np.seterr(**olderr)


@scope.define
def lognormal_lpdf(x, mu, sigma):
    # formula copied from wikipedia
    # http://en.wikipedia.org/wiki/Log-normal_distribution
    assert np.all(sigma >= 0)
    sigma = np.maximum(sigma, EPS)
    Z = sigma * x * np.sqrt(2 * np.pi)
    E = 0.5 * ((np.log(x) - mu) / sigma) ** 2
    rval = -E - np.log(Z)
    return rval


@scope.define
def qlognormal_lpdf(x, mu, sigma, q):
    # casting rounds up to nearest step multiple.
    # so lpdf is log of integral from x-step to x+1 of P(x)

    # XXX: subtracting two numbers potentially very close together.
    return np.log(
            lognormal_cdf(x, mu, sigma)
            - lognormal_cdf(x - q, mu, sigma))


@implicit_stochastic
@scope.define
def LGMM1(weights, mus, sigmas, low=None, high=None, q=None,
        rng=None, size=()):
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    n_samples = np.prod(size)
    #n_components = len(weights)
    if low is None and high is None:
        active = np.argmax(
                rng.multinomial(1, weights, (n_samples,)),
                axis=1)
        assert len(active) == n_samples
        samples = np.exp(
                rng.normal(
                    loc=mus[active],
                    scale=sigmas[active]))
    else:
        # -- draw from truncated components
        # TODO: one-sided-truncation
        low = float(low)
        high = float(high)
        if low >= high:
            raise ValueError('low >= high', (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(np.exp(draw))
        samples = np.asarray(samples)

    samples = np.reshape(np.asarray(samples), size)
    if q is not None:
        samples = np.round(samples / q) * q
    return samples


def logsum_rows(x):
    R, C = x.shape
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m


@scope.define
def LGMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = map(np.asarray,
            (samples, weights, mus, sigmas))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    _samples = samples
    if samples.ndim != 1:
        samples = samples.flatten()

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
                weights * (
                    normal_cdf(high, mus, sigmas)
                    - normal_cdf(low, mus, sigmas)))

    if q is None:
        # compute the lpdf of each sample under each component
        lpdfs = lognormal_lpdf(samples[:, None], mus, sigmas)
        rval = logsum_rows(lpdfs + np.log(weights))
    else:
        # compute the lpdf of each sample under each component
        prob = np.zeros(samples.shape, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + q / 2.0
            else:
                ubound = np.minimum(samples + q / 2.0, np.exp(high))
            if low is None:
                lbound = samples - q / 2.0
            else:
                lbound = np.maximum(samples - q / 2.0, np.exp(low))
            lbound = np.maximum(0, lbound)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * lognormal_cdf(ubound, mu, sigma)
            inc_amt -= w * lognormal_cdf(lbound, mu, sigma)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)
    rval.shape = _samples.shape
    return rval


#
# This is the weird heuristic ParzenWindow estimator used for continuous
# distributions in various ways.
#

@scope.define_info(o_len=3)
def adaptive_parzen_normal_orig(mus, prior_weight, prior_mu, prior_sigma):
    """
    A heuristic estimator for the mu and sigma values of a GMM
    TODO: try to find this heuristic in the literature, and cite it - Yoshua
    mentioned the term 'elastic' I think?

    mus - matrix (N, M) of M, N-dimensional component centers
    """
    mus_orig = np.array(mus)
    mus = np.array(mus)
    assert str(mus.dtype) != 'object'

    if mus.ndim != 1:
        raise TypeError('mus must be vector', mus)
    if len(mus) == 0:
        mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
    elif len(mus) == 1:
        mus = np.asarray([prior_mu] + [mus[0]])
        sigma = np.asarray([prior_sigma, prior_sigma * .5])
    elif len(mus) >= 2:
        order = np.argsort(mus)
        mus = mus[order]
        sigma = np.zeros_like(mus)
        sigma[1:-1] = np.maximum(
                mus[1:-1] - mus[0:-2],
                mus[2:] - mus[1:-1])
        if len(mus) > 2:
            lsigma = mus[2] - mus[0]
            usigma = mus[-1] - mus[-3]
        else:
            lsigma = mus[1] - mus[0]
            usigma = mus[-1] - mus[-2]

        sigma[0] = lsigma
        sigma[-1] = usigma

        # XXX: is sorting them necessary anymore?
        # un-sort the mus and sigma
        mus[order] = mus.copy()
        sigma[order] = sigma.copy()

        if not np.all(mus_orig == mus):
            print 'orig', mus_orig
            print 'mus', mus
        assert np.all(mus_orig == mus)

        # put the prior back in
        mus = np.asarray([prior_mu] + list(mus))
        sigma = np.asarray([prior_sigma] + list(sigma))

    maxsigma = prior_sigma
    # -- magic formula:
    minsigma = prior_sigma / np.sqrt(1 + len(mus))

    #print 'maxsigma, minsigma', maxsigma, minsigma
    sigma = np.clip(sigma, minsigma, maxsigma)

    weights = np.ones(len(mus), dtype=mus.dtype)
    weights[0] = prior_weight

    #print weights.dtype
    weights = weights / weights.sum()
    if 0:
        print 'WEIGHTS', weights
        print 'MUS', mus
        print 'SIGMA', sigma

    return weights, mus, sigma


@scope.define
def linear_forgetting_weights(N, LF):
    assert N >= 0
    assert LF > 0
    if N == 0:
        return np.asarray([])
    elif N < LF:
        return np.ones(N)
    else:
        ramp = np.linspace(1.0 / N, 1.0, num=N - LF)
        flat = np.ones(LF)
        weights = np.concatenate([ramp, flat], axis=0)
        assert weights.shape == (N,), (weights.shape, N)
        return weights

# XXX: make TPE do a post-inference pass over the pyll graph and insert
# non-default LF argument
@scope.define_info(o_len=3)
def adaptive_parzen_normal(mus, prior_weight, prior_mu, prior_sigma,
        LF=DEFAULT_LF):
    """
    mus - matrix (N, M) of M, N-dimensional component centers
    """
    #mus_orig = np.array(mus)
    mus = np.array(mus)
    assert str(mus.dtype) != 'object'

    if mus.ndim != 1:
        raise TypeError('mus must be vector', mus)
    if len(mus) == 0:
        srtd_mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
        prior_pos = 0
    elif len(mus) == 1:
        if prior_mu < mus[0]:
            prior_pos = 0
            srtd_mus = np.asarray([prior_mu, mus[0]])
            sigma = np.asarray([prior_sigma, prior_sigma * .5])
        else:
            prior_pos = 1
            srtd_mus = np.asarray([mus[0], prior_mu])
            sigma = np.asarray([prior_sigma * .5, prior_sigma])
    elif len(mus) >= 2:

        # create new_mus, which is sorted, and in which
        # the prior has been inserted
        order = np.argsort(mus)
        prior_pos = np.searchsorted(mus[order], prior_mu)
        srtd_mus = np.zeros(len(mus) + 1)
        srtd_mus[:prior_pos] = mus[order[:prior_pos]]
        srtd_mus[prior_pos] = prior_mu
        srtd_mus[prior_pos + 1:] = mus[order[prior_pos:]]
        sigma = np.zeros_like(srtd_mus)
        sigma[1:-1] = np.maximum(
                srtd_mus[1:-1] - srtd_mus[0:-2],
                srtd_mus[2:] - srtd_mus[1:-1])
        lsigma = srtd_mus[1] - srtd_mus[0]
        usigma = srtd_mus[-1] - srtd_mus[-2]
        sigma[0] = lsigma
        sigma[-1] = usigma

    if LF and LF < len(mus):
        unsrtd_weights = linear_forgetting_weights(len(mus), LF)
        srtd_weights = np.zeros_like(srtd_mus)
        assert len(unsrtd_weights) + 1 == len(srtd_mus)
        srtd_weights[:prior_pos] = unsrtd_weights[order[:prior_pos]]
        srtd_weights[prior_pos] = prior_weight
        srtd_weights[prior_pos + 1:] = unsrtd_weights[order[prior_pos:]]

    else:
        srtd_weights = np.ones(len(srtd_mus))
        srtd_weights[prior_pos] = prior_weight

    # -- magic formula:
    maxsigma = prior_sigma / 1.0
    minsigma = prior_sigma / min(100.0, (1.0 + len(srtd_mus)))

    #print 'maxsigma, minsigma', maxsigma, minsigma
    sigma = np.clip(sigma, minsigma, maxsigma)

    sigma[prior_pos] = prior_sigma
    assert prior_sigma > 0
    assert maxsigma > 0
    assert minsigma > 0
    assert np.all(sigma > 0), (sigma.min(), minsigma, maxsigma)


    #print weights.dtype
    srtd_weights /= srtd_weights.sum()
    if 0:
        print 'WEIGHTS', srtd_weights
        print 'MUS', srtd_mus
        print 'SIGMA', sigma

    return srtd_weights, srtd_mus, sigma

#
# Adaptive Parzen Samplers
# These produce conditional estimators for various prior distributions
#

# -- Uniform


@adaptive_parzen_sampler('uniform')
def ap_uniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_weight, prior_mu, prior_sigma)
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=None,
            size=size, rng=rng)


@adaptive_parzen_sampler('quniform')
def ap_quniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_weight, prior_mu, prior_sigma)
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=q,
            size=size, rng=rng)


@adaptive_parzen_sampler('loguniform')
def ap_loguniform_sampler(obs, prior_weight, low, high,
        size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_weight, prior_mu, prior_sigma)
    rval = scope.LGMM1(weights, mus, sigmas, low=low, high=high,
            size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qloguniform')
def ap_qloguniform_sampler(obs, prior_weight, low, high, q,
        size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(
                # -- map observations that were quantized to be below exp(low)
                #    (particularly 0) back up to exp(low) where they will
                #    interact in a reasonable way with the AdaptiveParzen
                #    thing.
                scope.maximum(
                    obs,
                    scope.maximum(  # -- protect against exp(low) underflow
                        EPS,
                        scope.exp(low)))),
            prior_weight, prior_mu, prior_sigma)
    return scope.LGMM1(weights, mus, sigmas, low, high, q=q,
            size=size, rng=rng)


# -- Normal

@adaptive_parzen_sampler('normal')
def ap_normal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, size=size, rng=rng)


@adaptive_parzen_sampler('qnormal')
def ap_qnormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, q=q, size=size, rng=rng)


@adaptive_parzen_sampler('lognormal')
def ap_loglognormal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_weight, mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qlognormal')
def ap_qlognormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    log_obs = scope.log(scope.maximum(obs, EPS))
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            log_obs, prior_weight, mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, q=q, size=size, rng=rng)
    return rval


# -- Categorical

@adaptive_parzen_sampler('randint')
def ap_categorical_sampler(obs, prior_weight, upper,
        size=(), rng=None, LF=DEFAULT_LF):
    weights = scope.linear_forgetting_weights(scope.len(obs), LF=LF)
    counts = scope.bincount(obs, minlength=upper, weights=weights)
    # -- add in some prior pseudocounts
    pseudocounts = counts + prior_weight
    return scope.categorical(pseudocounts / scope.sum(pseudocounts),
            upper=upper, size=size, rng=rng)


# @adaptive_parzen_sampler('categorical')
# def ap_categorical_sampler(obs, prior_weight, p, upper, size=(), rng=None,
#                            LF=DEFAULT_LF):
#     return scope.categorical(p, upper, size=size, rng
#                              =rng)

@scope.define
def tpe_cat_pseudocounts(counts, upper, prior_weight, p, size):
    #print counts
    if size == 0 or np.prod(size) == 0:
        return []
    if p.ndim == 2:
        assert np.all(p == p[0])
        p = p[0]
    pseudocounts = counts + upper * (prior_weight * p)
    return pseudocounts / np.sum(pseudocounts)

@adaptive_parzen_sampler('categorical')
def ap_categorical_sampler(obs, prior_weight, p, upper=None,
        size=(), rng=None, LF=DEFAULT_LF):
    weights = scope.linear_forgetting_weights(scope.len(obs), LF=LF)
    counts = scope.bincount(obs, minlength=upper, weights=weights)
    pseudocounts = scope.tpe_cat_pseudocounts(counts, upper, prior_weight, p, size)
    return scope.categorical(pseudocounts, upper=upper, size=size, rng=rng)

#
# Posterior clone performs symbolic inference on the pyll graph of priors.
#

@scope.define_info(o_len=2)
def ap_filter_trials(o_idxs, o_vals, l_idxs, l_vals, gamma,
        gamma_cap=DEFAULT_LF):
    """Return the elements of o_vals that correspond to trials whose losses
    were above gamma, or below gamma.
    """
    o_idxs, o_vals, l_idxs, l_vals = map(np.asarray, [o_idxs, o_vals, l_idxs,
        l_vals])

    # XXX if this is working, refactor this sort for efficiency

    # Splitting is done this way to cope with duplicate loss values.
    n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
    l_order = np.argsort(l_vals)


    keep_idxs = set(l_idxs[l_order[:n_below]])
    below = [v for i, v in zip(o_idxs, o_vals) if i in keep_idxs]

    if 0:
        print 'DEBUG: thresh', l_vals[l_order[:n_below]]

    keep_idxs = set(l_idxs[l_order[n_below:]])
    above = [v for i, v in zip(o_idxs, o_vals) if i in keep_idxs]

    #print 'AA0', below
    #print 'AA1', above

    return np.asarray(below), np.asarray(above)


def build_posterior(specs, prior_idxs, prior_vals, obs_idxs, obs_vals,
        oloss_idxs, oloss_vals, oloss_gamma, prior_weight):
    """
    This method clones a posterior inference graph by iterating forward in
    topological order, and replacing prior random-variables (prior_vals) with
    new posterior distributions that make use of observations (obs_vals).

    """
    assert all(isinstance(arg, pyll.Apply)
            for arg in [oloss_idxs, oloss_vals, oloss_gamma])

    expr = pyll.as_apply([specs, prior_idxs, prior_vals])
    nodes = pyll.dfs(expr)

    # build the joint posterior distribution as the values in this memo
    memo = {}
    # map prior RVs to observations
    obs_memo = {}

    for nid in prior_vals:
        # construct the leading args for each call to adaptive_parzen_sampler
        # which will permit the "adaptive parzen samplers" to adapt to the
        # correct samples.
        obs_below, obs_above = scope.ap_filter_trials(
                obs_idxs[nid], obs_vals[nid],
                oloss_idxs, oloss_vals, oloss_gamma)
        obs_memo[prior_vals[nid]] = [obs_below, obs_above]
    for node in nodes:
        if node not in memo:
            new_inputs = [memo[arg] for arg in node.inputs()]
            if node in obs_memo:
                # -- this case corresponds to an observed Random Var
                # node.name is a distribution like "normal", "randint", etc.
                obs_below, obs_above = obs_memo[node]
                aa = [memo[a] for a in node.pos_args]
                fn = adaptive_parzen_samplers[node.name]
                b_args = [obs_below, prior_weight] + aa
                named_args = [[kw, memo[arg]]
                        for (kw, arg) in node.named_args]
                b_post = fn(*b_args, **dict(named_args))
                a_args = [obs_above, prior_weight] + aa
                a_post = fn(*a_args, **dict(named_args))

                assert a_post.name == b_post.name
                fn_lpdf = getattr(scope, a_post.name + '_lpdf')
                #print fn_lpdf
                a_kwargs = dict([(n, a) for n, a in a_post.named_args
                            if n not in ('rng', 'size')])
                b_kwargs = dict([(n, a) for n, a in b_post.named_args
                            if n not in ('rng', 'size')])

                # calculate the llik of b_post under both distributions
                below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
                above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)

                #improvement = below_llik - above_llik
                #new_node = scope.broadcast_best(b_post, improvement)
                new_node = scope.broadcast_best(b_post, below_llik, above_llik)
            elif hasattr(node, 'obj'):
                # -- keep same literals in the graph
                new_node = node
            else:
                # -- this case is for all the other stuff in the graph
                new_node = node.clone_from_inputs(new_inputs)
            memo[node] = new_node
    post_specs = memo[specs]
    post_idxs = dict([(nid, memo[idxs])
        for nid, idxs in prior_idxs.items()])
    post_vals = dict([(nid, memo[vals])
        for nid, vals in prior_vals.items()])
    assert set(post_idxs.keys()) == set(post_vals.keys())
    assert set(post_idxs.keys()) == set(prior_idxs.keys())
    return post_specs, post_idxs, post_vals


@scope.define
def idxs_prod(full_idxs, idxs_by_label, llik_by_label):
    """Add all of the  log-likelihoods together by id.

    Example arguments:
    full_idxs = [0, 1, ... N-1]
    idxs_by_label = {'node_a': [1, 3], 'node_b': [3]}
    llik_by_label = {'node_a': [0.1, -3.3], node_b: [1.0]}

    This would return N elements: [0, 0.1, 0, -2.3, 0, 0, ... ]
    """
    #print 'FULL IDXS'
    #print full_idxs
    assert len(set(full_idxs)) == len(full_idxs)
    full_idxs = list(full_idxs)
    rval = np.zeros(len(full_idxs))
    pos_of_tid = dict(zip(full_idxs, range(len(full_idxs))))
    assert set(idxs_by_label.keys()) == set(llik_by_label.keys())
    for nid in idxs_by_label:
        idxs = idxs_by_label[nid]
        llik = llik_by_label[nid]
        assert np.all(np.asarray(idxs) > 1)
        assert len(set(idxs)) == len(idxs)
        assert len(idxs) == len(llik)
        for ii, ll in zip(idxs, llik):
            rval[pos_of_tid[ii]] += ll
            #rval[full_idxs.index(ii)] += ll
    return rval


@scope.define
def broadcast_best(samples, below_llik, above_llik):
    if len(samples):
        #print 'AA2', dict(zip(samples, below_llik - above_llik))
        score = below_llik - above_llik
        if len(samples) != len(score):
            raise ValueError()
        best = np.argmax(score)
        return [samples[best]] * len(samples)
    else:
        return []


_default_prior_weight = 1.0

# -- suggest best of this many draws on every iteration
_default_n_EI_candidates = 24

# -- gamma * sqrt(n_trials) is fraction of to use as good
_default_gamma = 0.25

_default_n_startup_jobs = 20

_default_linear_forgetting = DEFAULT_LF


def tpe_transform(domain, prior_weight, gamma):
    s_prior_weight = pyll.Literal(float(prior_weight))

    # -- these dummy values will be replaced in suggest1() and never used
    observed = dict(
            idxs=pyll.Literal(),
            vals=pyll.Literal())
    observed_loss = dict(
            idxs=pyll.Literal(),
            vals=pyll.Literal())

    specs, idxs, vals = build_posterior(
            # -- vectorized clone of bandit template
            domain.vh.v_expr,
            # -- this dict and next represent prior dists
            domain.vh.idxs_by_label(),
            domain.vh.vals_by_label(),
            observed['idxs'],
            observed['vals'],
            observed_loss['idxs'],
            observed_loss['vals'],
            pyll.Literal(gamma),
            s_prior_weight
            )

    return (s_prior_weight, observed, observed_loss,
            specs, idxs, vals)


def suggest(new_ids, domain, trials, seed,
        prior_weight=_default_prior_weight,
        n_startup_jobs=_default_n_startup_jobs,
        n_EI_candidates=_default_n_EI_candidates,
        gamma=_default_gamma,
        linear_forgetting=_default_linear_forgetting,
        ):

    new_id = new_ids[0]

    t0 = time.time()
    (s_prior_weight, observed, observed_loss, specs, opt_idxs, opt_vals) \
            = tpe_transform(domain, prior_weight, gamma)
    tt = time.time() - t0
    logger.info('tpe_transform took %f seconds' % tt)

    best_docs = dict()
    best_docs_loss = dict()
    for doc in trials.trials:
        # get either this docs own tid or the one that it's from
        tid = doc['misc'].get('from_tid', doc['tid'])
        loss = domain.loss(doc['result'], doc['spec'])
        if loss is None:
            # -- associate infinite loss to new/running/failed jobs
            loss = float('inf')
        else:
            loss = float(loss)
        best_docs_loss.setdefault(tid, loss)
        if loss <= best_docs_loss[tid]:
            best_docs_loss[tid] = loss
            best_docs[tid] = doc

    tid_docs = best_docs.items()
    # -- sort docs by order of suggestion
    #    so that linear_forgetting removes the oldest ones
    tid_docs.sort()
    losses = [best_docs_loss[k] for k, v in tid_docs]
    tids = [k for k, v in tid_docs]
    docs = [v for k, v in tid_docs]

    if docs:
        logger.info('TPE using %i/%i trials with best loss %f' % (
            len(docs), len(trials), min(best_docs_loss.values())))
    else:
        logger.info('TPE using 0 trials')

    if len(docs) < n_startup_jobs:
        # N.B. THIS SEEDS THE RNG BASED ON THE new_id
        return rand.suggest(new_ids, domain, trials, seed)

    #    Sample and compute log-probability.
    if tids:
        # -- the +2 co-ordinates with an assertion above
        #    to ensure that fake ids are used during sampling
        fake_id_0 = max(max(tids), new_id) + 2
    else:
        # -- weird - we're running the TPE algo from scratch
        assert n_startup_jobs <= 0
        fake_id_0 = new_id + 2

    fake_ids = range(fake_id_0, fake_id_0 + n_EI_candidates)

    # -- this dictionary will map pyll nodes to the values
    #    they should take during the evaluation of the pyll program
    memo = {
        domain.s_new_ids: fake_ids,
        domain.s_rng: np.random.RandomState(seed),
           }

    o_idxs_d, o_vals_d = miscs_to_idxs_vals(
        [d['misc'] for d in docs], keys=domain.params.keys())
    memo[observed['idxs']] = o_idxs_d
    memo[observed['vals']] = o_vals_d

    memo[observed_loss['idxs']] = tids
    memo[observed_loss['vals']] = losses

    idxs, vals = pyll.rec_eval([opt_idxs, opt_vals], memo=memo,
            print_node_on_error=False)

    # -- retrieve the best of the samples and form the return tuple
    # the build_posterior makes all specs the same

    rval_specs = [None]  # -- specs are deprecated
    rval_results = [domain.new_result()]
    rval_miscs = [dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)]

    miscs_update_idxs_vals(rval_miscs, idxs, vals,
            idxs_map={fake_ids[0]: new_id},
            assert_all_vals_used=False)
    rval_docs = trials.new_trial_docs([new_id],
            rval_specs, rval_results, rval_miscs)

    return rval_docs


