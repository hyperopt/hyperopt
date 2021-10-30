"""
Graphical model (GM)-based optimization algorithm using Theano
"""
from past.utils import old_div
import logging
import time

import numpy as np
from scipy.special import erf
from . import pyll
from .pyll import scope
from .pyll.stochastic import implicit_stochastic

from .base import miscs_to_idxs_vals
from .base import miscs_update_idxs_vals

# from .base import Trials
from . import rand

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"
logger = logging.getLogger(__name__)

EPS = 1e-12

# -- default linear forgetting. don't try to change by writing this variable
# because it's captured in function default args when this file is read
DEFAULT_LF = 25


adaptive_parzen_samplers = {}


# a decorator to register functions to the dict `adaptive_parzen_samplers`
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
def categorical_lpdf(sample, p):
    if sample.size:
        return np.log(np.asarray(p)[sample])
    return np.asarray([])


@scope.define
def randint_via_categorical_lpdf(sample, p):
    if sample.size:
        return np.log(np.asarray(p)[sample])
    return np.asarray([])


# -- Bounded Gaussian Mixture Model (BGMM)


@implicit_stochastic
@scope.define
def GMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
    """Sample from truncated 1-D Gaussian Mixture Model"""
    weights, mus, sigmas = list(map(np.asarray, (weights, mus, sigmas)))
    assert len(weights) == len(mus) == len(sigmas)
    n_samples = int(np.prod(size))
    # n_components = len(weights)
    if low is None and high is None:
        # -- draw from a standard GMM
        active = np.argmax(rng.multinomial(1, weights, (n_samples,)), axis=1)
        samples = rng.normal(loc=mus[active], scale=sigmas[active])
    else:
        # -- draw from truncated components, handling one-sided truncation
        low = float(low) if low is not None else -float("Inf")
        high = float(high) if high is not None else float("Inf")
        if low >= high:
            raise ValueError("low >= high", (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(draw)
    samples = np.reshape(np.asarray(samples), size)
    if q is None:
        return samples
    return np.round(old_div(samples, q)) * q


@scope.define
def normal_cdf(x, mu, sigma):
    top = x - mu
    bottom = np.maximum(np.sqrt(2) * sigma, EPS)
    z = old_div(top, bottom)
    return 0.5 * (1 + erf(z))


@scope.define
def GMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    def print_verbose(s, x):
        return print(f"GMM1_lpdf:{s}", x)

    verbose = 0
    samples, weights, mus, sigmas = list(
        map(np.asarray, (samples, weights, mus, sigmas))
    )
    if samples.size == 0:
        return np.asarray([])
    if weights.ndim != 1:
        raise TypeError("need vector of weights", weights.shape)
    if mus.ndim != 1:
        raise TypeError("need vector of mus", mus.shape)
    if sigmas.ndim != 1:
        raise TypeError("need vector of sigmas", sigmas.shape)
    assert len(weights) == len(mus) == len(sigmas)
    _samples = samples
    samples = _samples.flatten()

    if verbose:
        print_verbose("samples", set(samples))
        print_verbose("weights", weights)
        print_verbose("mus", mus)
        print_verbose("sigmas", sigmas)
        print_verbose("low", low)
        print_verbose("high", high)
        print_verbose("q", q)

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
            weights * (normal_cdf(high, mus, sigmas) - normal_cdf(low, mus, sigmas))
        )

    if q is None:
        dist = samples[:, None] - mus
        mahal = (old_div(dist, np.maximum(sigmas, EPS))) ** 2
        # mahal shape is (n_samples, n_components)
        Z = np.sqrt(2 * np.pi * sigmas ** 2)
        coef = weights / Z / p_accept
        rval = logsum_rows(-0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros(samples.shape, dtype="float64")
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + old_div(q, 2.0)
            else:
                ubound = np.minimum(samples + old_div(q, 2.0), high)
            if low is None:
                lbound = samples - old_div(q, 2.0)
            else:
                lbound = np.maximum(samples - old_div(q, 2.0), low)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * normal_cdf(ubound, mu, sigma)
            inc_amt -= w * normal_cdf(lbound, mu, sigma)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)

    if verbose:
        print_verbose("rval:", dict(list(zip(samples, rval))))

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
        raise ValueError("negative arg to lognormal_cdf", x)
    olderr = np.seterr(divide="ignore")
    try:
        top = np.log(np.maximum(x, EPS)) - mu
        bottom = np.maximum(np.sqrt(2) * sigma, EPS)
        z = old_div(top, bottom)
        return 0.5 + 0.5 * erf(z)
    finally:
        np.seterr(**olderr)


@scope.define
def lognormal_lpdf(x, mu, sigma):
    # formula copied from wikipedia
    # http://en.wikipedia.org/wiki/Log-normal_distribution
    assert np.all(sigma >= 0)
    sigma = np.maximum(sigma, EPS)
    Z = sigma * x * np.sqrt(2 * np.pi)
    E = 0.5 * (old_div((np.log(x) - mu), sigma)) ** 2
    rval = -E - np.log(Z)
    return rval


@scope.define
def qlognormal_lpdf(x, mu, sigma, q):
    # casting rounds up to nearest step multiple.
    # so lpdf is log of integral from x-step to x+1 of P(x)

    # XXX: subtracting two numbers potentially very close together.
    return np.log(lognormal_cdf(x, mu, sigma) - lognormal_cdf(x - q, mu, sigma))


@implicit_stochastic
@scope.define
def LGMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
    weights, mus, sigmas = list(map(np.asarray, (weights, mus, sigmas)))
    n_samples = np.prod(size)
    # n_components = len(weights)
    if low is None and high is None:
        active = np.argmax(rng.multinomial(1, weights, (n_samples,)), axis=1)
        assert len(active) == n_samples
        samples = np.exp(rng.normal(loc=mus[active], scale=sigmas[active]))
    else:
        # -- draw from truncated components
        # TODO: one-sided-truncation
        low = float(low)
        high = float(high)
        if low >= high:
            raise ValueError("low >= high", (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(np.exp(draw))
        samples = np.asarray(samples)

    samples = np.reshape(np.asarray(samples), size)
    if q is not None:
        samples = np.round(old_div(samples, q)) * q
    return samples


def logsum_rows(x):
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m


@scope.define
def LGMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = list(
        map(np.asarray, (samples, weights, mus, sigmas))
    )
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
            weights * (normal_cdf(high, mus, sigmas) - normal_cdf(low, mus, sigmas))
        )

    if q is None:
        # compute the lpdf of each sample under each component
        lpdfs = lognormal_lpdf(samples[:, None], mus, sigmas)
        rval = logsum_rows(lpdfs + np.log(weights))
    else:
        # compute the lpdf of each sample under each component
        prob = np.zeros(samples.shape, dtype="float64")
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + old_div(q, 2.0)
            else:
                ubound = np.minimum(samples + old_div(q, 2.0), np.exp(high))
            if low is None:
                lbound = samples - old_div(q, 2.0)
            else:
                lbound = np.maximum(samples - old_div(q, 2.0), np.exp(low))
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
    assert str(mus.dtype) != "object"

    if mus.ndim != 1:
        raise TypeError("mus must be vector", mus)
    if len(mus) == 0:
        mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
    elif len(mus) == 1:
        mus = np.asarray([prior_mu] + [mus[0]])
        sigma = np.asarray([prior_sigma, prior_sigma * 0.5])
    elif len(mus) >= 2:
        order = np.argsort(mus)
        mus = mus[order]
        sigma = np.zeros_like(mus)
        sigma[1:-1] = np.maximum(mus[1:-1] - mus[0:-2], mus[2:] - mus[1:-1])
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
            print("orig", mus_orig)
            print("mus", mus)
        assert np.all(mus_orig == mus)

        # put the prior back in
        mus = np.asarray([prior_mu] + list(mus))
        sigma = np.asarray([prior_sigma] + list(sigma))

    maxsigma = prior_sigma
    # -- magic formula:
    minsigma = old_div(prior_sigma, np.sqrt(1 + len(mus)))

    sigma = np.clip(sigma, minsigma, maxsigma)

    weights = np.ones(len(mus), dtype=mus.dtype)
    weights[0] = prior_weight

    weights = old_div(weights, weights.sum())

    return weights, mus, sigma


@scope.define
def linear_forgetting_weights(N, LF):
    assert N >= 0
    assert LF > 0
    if N == 0:
        return np.asarray([])
    if N < LF:
        return np.ones(N)
    ramp = np.linspace(old_div(1.0, N), 1.0, num=N - LF)
    flat = np.ones(LF)
    weights = np.concatenate([ramp, flat], axis=0)
    assert weights.shape == (N,), (weights.shape, N)
    return weights


# XXX: make TPE do a post-inference pass over the pyll graph and insert
# non-default LF argument


@scope.define_info(o_len=3)
def adaptive_parzen_normal(mus, prior_weight, prior_mu, prior_sigma, LF=DEFAULT_LF):
    """
    mus - matrix (N, M) of M, N-dimensional component centers
    """
    mus = np.array(mus)
    assert str(mus.dtype) != "object"

    if mus.ndim != 1:
        raise TypeError("mus must be vector", mus)
    if len(mus) == 0:
        srtd_mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
        prior_pos = 0
    elif len(mus) == 1:
        if prior_mu < mus[0]:
            prior_pos = 0
            srtd_mus = np.asarray([prior_mu, mus[0]])
            sigma = np.asarray([prior_sigma, prior_sigma * 0.5])
        else:
            prior_pos = 1
            srtd_mus = np.asarray([mus[0], prior_mu])
            sigma = np.asarray([prior_sigma * 0.5, prior_sigma])
    elif len(mus) >= 2:

        # create new_mus, which is sorted, and in which
        # the prior has been inserted
        order = np.argsort(mus)
        prior_pos = np.searchsorted(mus[order], prior_mu)
        srtd_mus = np.zeros(len(mus) + 1)
        srtd_mus[:prior_pos] = mus[order[:prior_pos]]
        srtd_mus[prior_pos] = prior_mu
        srtd_mus[prior_pos + 1 :] = mus[order[prior_pos:]]
        sigma = np.zeros_like(srtd_mus)
        sigma[1:-1] = np.maximum(
            srtd_mus[1:-1] - srtd_mus[0:-2], srtd_mus[2:] - srtd_mus[1:-1]
        )
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
        srtd_weights[prior_pos + 1 :] = unsrtd_weights[order[prior_pos:]]

    else:
        srtd_weights = np.ones(len(srtd_mus))
        srtd_weights[prior_pos] = prior_weight

    # -- magic formula:
    maxsigma = old_div(prior_sigma, 1.0)
    minsigma = old_div(prior_sigma, min(100.0, (1.0 + len(srtd_mus))))

    sigma = np.clip(sigma, minsigma, maxsigma)

    sigma[prior_pos] = prior_sigma
    assert prior_sigma > 0
    assert maxsigma > 0
    assert minsigma > 0
    assert np.all(sigma > 0), (sigma.min(), minsigma, maxsigma)

    srtd_weights /= srtd_weights.sum()

    return srtd_weights, srtd_mus, sigma


#
# Adaptive Parzen Samplers
# These produce conditional estimators for various prior distributions
#
# NOTE: These are actually used in a fairly complicated way.
# They are actually returning pyll.Apply AST (Abstract Syntax Tree) objects.
# This AST is then manipulated and the corresponding _lpdf function is called
# (e.g  GMM1_lpdf)
#
# Please see the build_posterior function for details

# -- Uniform


@adaptive_parzen_sampler("uniform")
def ap_uniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        obs, prior_weight, prior_mu, prior_sigma
    )
    return scope.GMM1(
        weights, mus, sigmas, low=low, high=high, q=None, size=size, rng=rng
    )


@adaptive_parzen_sampler("quniform")
def ap_quniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        obs, prior_weight, prior_mu, prior_sigma
    )
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=q, size=size, rng=rng)


@adaptive_parzen_sampler("loguniform")
def ap_loguniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.log(obs), prior_weight, prior_mu, prior_sigma
    )
    rval = scope.LGMM1(weights, mus, sigmas, low=low, high=high, size=size, rng=rng)
    return rval


@adaptive_parzen_sampler("qloguniform")
def ap_qloguniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
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
                    EPS, scope.exp(low)
                ),
            )
        ),
        prior_weight,
        prior_mu,
        prior_sigma,
    )
    return scope.LGMM1(weights, mus, sigmas, low, high, q=q, size=size, rng=rng)


# -- Normal


@adaptive_parzen_sampler("normal")
def ap_normal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, size=size, rng=rng)


@adaptive_parzen_sampler("qnormal")
def ap_qnormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, q=q, size=size, rng=rng)


@adaptive_parzen_sampler("lognormal")
def ap_loglognormal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.log(obs), prior_weight, mu, sigma
    )
    rval = scope.LGMM1(weights, mus, sigmas, size=size, rng=rng)
    return rval


@adaptive_parzen_sampler("qlognormal")
def ap_qlognormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    log_obs = scope.log(scope.maximum(obs, EPS))
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        log_obs, prior_weight, mu, sigma
    )
    rval = scope.LGMM1(weights, mus, sigmas, q=q, size=size, rng=rng)
    return rval


# -- Categorical


@adaptive_parzen_sampler("randint")
def ap_randint_sampler(
    obs, prior_weight, low, high=None, size=(), rng=None, LF=DEFAULT_LF
):
    # randint can be seen as a categorical with high - low categories
    weights = scope.linear_forgetting_weights(scope.len(obs), LF=LF)
    # if high is None, then low represents high and there is no offset
    domain_size = low if high is None else high - low
    offset = pyll.Literal(0) if high is None else low
    counts = scope.bincount(obs, offset=offset, minlength=domain_size, weights=weights)
    # -- add in some prior pseudocounts
    pseudocounts = counts + prior_weight
    random_variable = scope.randint_via_categorical(
        old_div(pseudocounts, scope.sum(pseudocounts)), size=size, rng=rng
    )
    return random_variable


@scope.define
def tpe_cat_pseudocounts(counts, prior_weight, p, size):
    if np.prod(size) == 0:
        return []
    if p.ndim == 2:
        assert np.all(p == p[0])
        p = p[0]
    pseudocounts = counts + p.size * (prior_weight * p)
    return old_div(pseudocounts, np.sum(pseudocounts))


@adaptive_parzen_sampler("categorical")
def ap_categorical_sampler(obs, prior_weight, p, size=(), rng=None, LF=DEFAULT_LF):
    weights = scope.linear_forgetting_weights(scope.len(obs), LF=LF)
    # in order to support pchoice here, we need to find the size of p,
    # but p can have p.ndim == 2, so we pass p to bincount and unpack it
    # (if required) there
    counts = scope.bincount(obs, p=p, weights=weights)
    pseudocounts = scope.tpe_cat_pseudocounts(counts, prior_weight, p, size)
    return scope.categorical(pseudocounts, size=size, rng=rng)


#
# Posterior clone performs symbolic inference on the pyll graph of priors.
#


@scope.define_info(o_len=2)
def ap_split_trials(o_idxs, o_vals, l_idxs, l_vals, gamma, gamma_cap=DEFAULT_LF):
    """Split the elements of `o_vals` (observations values) into two groups: those for
    trials whose losses (`l_vals`) were above gamma, and those below gamma. Note that
    only unique elements are returned, so the total number of returned elements might
    be lower than `len(o_vals)`
    """
    o_idxs, o_vals, l_idxs, l_vals = list(
        map(np.asarray, [o_idxs, o_vals, l_idxs, l_vals])
    )

    # XXX if this is working, refactor this sort for efficiency

    # Splitting is done this way to cope with duplicate loss values.
    n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
    l_order = np.argsort(l_vals)

    keep_idxs = set(l_idxs[l_order[:n_below]])
    below = [v for i, v in zip(o_idxs, o_vals) if i in keep_idxs]

    keep_idxs = set(l_idxs[l_order[n_below:]])
    above = [v for i, v in zip(o_idxs, o_vals) if i in keep_idxs]

    return np.asarray(below), np.asarray(above)


@scope.define
def broadcast_best(samples, below_llik, above_llik):
    if len(samples):
        score = below_llik - above_llik
        if len(samples) != len(score):
            raise ValueError()
        best = np.argmax(score)
        return [samples[best]] * len(samples)
    else:
        return []


def build_posterior(
    specs,
    prior_idxs,
    prior_vals,
    obs_idxs,
    obs_vals,
    obs_loss_idxs,
    obs_loss_vals,
    oloss_gamma,
    prior_weight,
):
    """
    This method clones a posterior inference graph by iterating forward in
    topological order, and replacing prior random-variables (prior_idxs, prior_vals)
    with new posterior distributions (post_specs, post_idxs, post_vals) that make use
    of observations (obs_idxs, obs_vals).

    """
    assert all(
        isinstance(arg, pyll.Apply)
        for arg in [obs_loss_idxs, obs_loss_vals, oloss_gamma]
    )
    assert set(prior_idxs.keys()) == set(prior_vals.keys())

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
        obs_below, obs_above = scope.ap_split_trials(
            obs_idxs[nid], obs_vals[nid], obs_loss_idxs, obs_loss_vals, oloss_gamma
        )
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
                named_args = {kw: memo[arg] for (kw, arg) in node.named_args}
                b_post = fn(*b_args, **named_args)
                a_args = [obs_above, prior_weight] + aa
                a_post = fn(*a_args, **named_args)

                # fn is a function e.g ap_uniform_sampler, ap_normal_sampler, etc
                # b_post and a_post are pyll.Apply objects that are
                # AST (Abstract Syntax Trees).  They create the distribution,
                # (e.g. using adaptive_parzen_normal), and then
                # call a function to sample randomly from that distribution
                # (e.g. using scope.GMM1) which return those samples.
                #
                # However we are only interested in using the samples from b_post.
                # This code looks at the AST and grabs the function name that we used
                # for sampling (e.g. scope.GMM1)   and modifies it, e.g. to
                # "scope.GMM1_lpdf". It then calls this function, passing in the
                # samples as the first parameter.a_args
                #
                # The result is that we are effectively calling, for example:
                # below_llik = GMM1_lpdf( b_post, *adaptive_parzen_normal(obs_below, ...))
                # above_llik = GMM1_lpdf( b_post, *adaptive_parzen_normal(obs_above, ...))

                assert a_post.name == b_post.name
                fn_lpdf = getattr(scope, a_post.name + "_lpdf")
                a_kwargs = {
                    n: a for n, a in a_post.named_args if n not in ("rng", "size")
                }
                b_kwargs = {
                    n: a for n, a in b_post.named_args if n not in ("rng", "size")
                }

                # calculate the log likelihood of b_post under both distributions
                below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
                above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)
                # compute new_node based on below & above log likelihood
                new_node = scope.broadcast_best(b_post, below_llik, above_llik)
            elif hasattr(node, "obj"):
                # -- keep same literals in the graph
                new_node = node
            else:
                # -- this case is for all the other stuff in the graph
                new_node = node.clone_from_inputs(new_inputs)
            memo[node] = new_node

    post_idxs = {nid: memo[idxs] for nid, idxs in prior_idxs.items()}
    post_vals = {nid: memo[vals] for nid, vals in prior_vals.items()}
    return post_idxs, post_vals


# TODO: is this used?
# @scope.define
# def idxs_prod(full_idxs, idxs_by_label, llik_by_label):
#     """Add all of the  log-likelihoods together by id.
#
#     Example arguments:
#     full_idxs = [0, 1, ... N-1]
#     idxs_by_label = {'node_a': [1, 3], 'node_b': [3]}
#     llik_by_label = {'node_a': [0.1, -3.3], node_b: [1.0]}
#
#     This would return N elements: [0, 0.1, 0, -2.3, 0, 0, ... ]
#     """
#     assert len(set(full_idxs)) == len(full_idxs)
#     full_idxs = list(full_idxs)
#     rval = np.zeros(len(full_idxs))
#     pos_of_tid = dict(list(zip(full_idxs, list(range(len(full_idxs))))))
#     assert set(idxs_by_label.keys()) == set(llik_by_label.keys())
#     for nid in idxs_by_label:
#         idxs = idxs_by_label[nid]
#         llik = llik_by_label[nid]
#         assert np.all(np.asarray(idxs) > 1)
#         assert len(set(idxs)) == len(idxs)
#         assert len(idxs) == len(llik)
#         for ii, ll in zip(idxs, llik):
#             rval[pos_of_tid[ii]] += ll
#     return rval


_default_prior_weight = 1.0

# -- suggest best of this many draws on every iteration
_default_n_EI_candidates = 24

# -- gamma * sqrt(n_trials) is fraction of to use as good
_default_gamma = 0.25

_default_n_startup_jobs = 20

_default_linear_forgetting = DEFAULT_LF


def build_posterior_wrapper(domain, prior_weight, gamma):
    """
    Calls build_posterior
    Args:
        domain (hyperopt.base.Domain): contains info about the obj function and the hp
            space passed to fmin
        prior_weight (float): smoothing factor for counts, to avoid having 0 prob
        # TODO: consider renaming or improving documentation for suggest
        gamma (float): the threshold to split between l(x) and g(x), see eq. 2 in
            https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf

    Returns:

    """

    # -- these dummy values will be replaced in build_posterior() and never used
    observed = {"idxs": pyll.Literal(), "vals": pyll.Literal()}
    observed_loss = {"idxs": pyll.Literal(), "vals": pyll.Literal()}

    posterior = build_posterior(
        # -- vectorized clone of bandit template
        domain.vh.v_expr,
        # -- this dict and next represent prior dists
        domain.vh.idxs_by_label(),
        domain.vh.vals_by_label(),
        observed["idxs"],
        observed["vals"],
        observed_loss["idxs"],
        observed_loss["vals"],
        pyll.Literal(gamma),
        pyll.Literal(float(prior_weight)),
    )

    return observed, observed_loss, posterior


def suggest(
    new_ids,
    domain,
    trials,
    seed,
    prior_weight=_default_prior_weight,
    n_startup_jobs=_default_n_startup_jobs,
    n_EI_candidates=_default_n_EI_candidates,
    gamma=_default_gamma,
    verbose=True,
):
    """
    Given previous trials and the domain, suggest the best expected hp point
    according to the TPE-EI algo


    Args:
        prior_weight(
        n_startup_jobs:
        n_EI_candidates:
        gamma:
        verbose:

    Returns:

    """

    t0 = time.time()
    # use build_posterior_wrapper to create the pyll nodes
    observed, observed_loss, posterior = build_posterior_wrapper(
        domain, prior_weight, gamma
    )
    tt = time.time() - t0
    if verbose:
        logger.info("build_posterior_wrapper took %f seconds" % tt)

    # Loop over previous trials to collect best_docs and best_docs_loss
    best_docs = dict()
    best_docs_loss = dict()
    for doc in trials.trials:

        # get either these docs own tid or the one that it's from
        tid = doc["misc"].get("from_tid", doc["tid"])

        # associate infinite loss to new/running/failed jobs
        loss = doc["result"].get("loss")
        loss = float("inf") if loss is None else float(loss)

        # if set, update loss for this tid if it's higher than current loss
        # otherwise, set it
        best_docs_loss.setdefault(tid, loss)
        if loss <= best_docs_loss[tid]:
            best_docs_loss[tid] = loss
            best_docs[tid] = doc

    # -- sort docs by order of suggestion
    #    so that linear_forgetting removes the oldest ones
    tid_docs = sorted(best_docs.items())
    losses = [best_docs_loss[tid] for tid, doc in tid_docs]
    tids, docs = list(zip(*tid_docs)) if tid_docs else ([], [])

    if verbose:
        if docs:
            s = "%i/%i trials with best loss %f" % (
                len(docs),
                len(trials),
                np.nanmin(losses),
            )
        else:
            s = "0 trials"
        logger.info("TPE using %s" % s)

    if len(docs) < n_startup_jobs:
        # N.B. THIS SEEDS THE RNG BASED ON THE new_id
        return rand.suggest(new_ids, domain, trials, seed)

    # Sample and compute log-probability.
    first_new_id = new_ids[0]
    if tids:
        # -- the +2 coordinates with an assertion above
        #    to ensure that fake ids are used during sampling
        #    TODO: not sure what assertion this refers to...
        fake_id_0 = max(max(tids), first_new_id) + 2
    else:
        # -- weird - we're running the TPE algo from scratch
        assert n_startup_jobs <= 0
        fake_id_0 = first_new_id + 2

    fake_ids = list(range(fake_id_0, fake_id_0 + n_EI_candidates))

    # -- this dictionary will map pyll nodes to the values
    #    they should take during the evaluation of the pyll program
    memo = {domain.s_new_ids: fake_ids, domain.s_rng: np.random.default_rng(seed)}

    memo[observed_loss["idxs"]] = tids
    memo[observed_loss["vals"]] = losses

    observed_idxs_dict, observed_vals_dict = miscs_to_idxs_vals(
        [doc["misc"] for doc in docs], keys=list(domain.params.keys())
    )
    memo[observed["idxs"]] = observed_idxs_dict
    memo[observed["vals"]] = observed_vals_dict

    # evaluate `n_EI_candidates` pyll nodes in `posterior` using `memo`
    # TODO: it seems to return idxs, vals, all the same. Is this correct?
    idxs, vals = pyll.rec_eval(posterior, memo=memo, print_node_on_error=False)

    # hack to add offset again for randint params
    for label, param in domain.params.items():
        if param.name == "randint" and len(param.pos_args) == 2:
            offset = param.pos_args[0].obj
            vals[label] = [val + offset for val in vals[label]]

    # -- retrieve the best of the samples and form the return tuple

    # specs are deprecated since build_posterior makes all the same
    rval_specs = [None]
    rval_results = [domain.new_result()]
    rval_miscs = [{"tid": first_new_id, "cmd": domain.cmd, "workdir": domain.workdir}]

    miscs_update_idxs_vals(
        rval_miscs,
        idxs,
        vals,
        idxs_map={fake_ids[0]: first_new_id},
        assert_all_vals_used=False,
    )
    # return the doc for the best new trial
    return trials.new_trial_docs([first_new_id], rval_specs, rval_results, rval_miscs)
