"""
Graphical model (GM)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.special import erf
import pyll
from pyll import scope
from pyll.stochastic import implicit_stochastic

from .base import BanditAlgo
from .base import STATUS_OK
from .base import miscs_to_idxs_vals
from .base import miscs_update_idxs_vals


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
def categorical_lpdf(sample, p, eps=0):
    """
    Return a random integer from 0 .. N-1 inclusive according to the
    probabilities p[0] .. P[N-1].

    This is formally equivalent to np.where(multinomial(n=1, p))
    """
    return np.log(np.asarray(p)[sample] + eps)


# -- Bounded Gaussian Mixture Model (BGMM)

@implicit_stochastic
@scope.define
def GMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None,
        size=()):
    """Sample from truncated 1-D Gaussian Mixture Model"""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    assert len(weights) == len(mus) == len(sigmas)
    n_samples = np.prod(size)
    n_components = len(weights)
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
            if low < draw < high:
                samples.append(draw)
    samples = np.reshape(np.asarray(samples), size)
    if q is None:
        return samples
    else:
        return np.floor(samples / q) * q

@scope.define
def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / np.sqrt(2 * sigma ** 2)))

@scope.define
def GMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = map(np.asarray,
            (samples, weights, mus, sigmas))
    assert weights.ndim == mus.ndim == sigmas.ndim == 1
    assert len(weights) == len(mus) == len(sigmas)
    _samples = samples
    samples = _samples.flatten()

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
                weights * (
                    normal_cdf(high, mus, sigmas)
                    - normal_cdf(low, mus, sigmas)))

    if q is None:
        dist = samples[:, None] - mus
        mahal = ((dist ** 2) / (sigmas ** 2))
        # mahal shape is (n_samples, n_components)
        Z = np.sqrt(2 * np.pi * sigmas**2)
        coef = weights / Z / p_accept
        rval = logsum_rows(- 0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros_like(samples, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            prob += normal_cdf(samples, mu, sigma)
            prob -= normal_cdf(samples - q, mu, sigma)
        rval = np.log(prob)

    rval.shape = _samples.shape
    return rval


# -- Mixture of Log-Normals

@scope.define
def lognormal_cdf(x, mu, sigma, eps=1e-12):
    # wikipedia claims cdf is
    # .5 + .5 erf( log(x) - mu / sqrt(2 sigma^2))
    #
    # the maximum is used to move negative values and 0 up to a point
    # where they do not cause nan or inf, but also don't contribute much
    # to the cdf.
    return .5 + .5 * erf(
            (np.log(np.maximum(x, eps)) - mu)
            / np.sqrt(2 * sigma**2))

@scope.define
def lognormal_lpdf(x, mu, sigma):
    # formula copied from wikipedia
    # http://en.wikipedia.org/wiki/Log-normal_distribution
    Z = sigma * x * np.sqrt(2 * np.pi)
    E = 0.5 * ((np.log(x) - mu) / sigma)**2
    return -E - np.log(Z)

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
def LGMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    n_samples = np.prod(size)
    n_components = len(weights)
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
            if low < draw < high:
                samples.append(np.exp(draw))
        samples = np.asarray(samples)

    if not np.all(np.isfinite(samples)):
        logger.warning('overflow in LognormalMixture')
        logger.warning('  mu = %s' % str(mus[active]))
        logger.warning('  sigma = %s' % str(sigmas[active]))
        logger.warning('  samples = %s' % str(samples))

    if q is not None:
        samples = np.floor(samples / q) * q
    samples = np.reshape(np.asarray(samples), size)
    return samples

def logsum_rows(x):
    R, C = x.shape
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m

@scope.define
def LGMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = map(np.asarray, (
        samples, weights, mus, sigmas))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    _samples = samples
    if samples.ndim != 1:
        samples = samples.flatten()

    if q is None:
        # compute the lpdf of each sample under each component
        lpdfs = lognormal_lpdf(samples[:, None], mus, sigmas)
        assert lpdfs.ndim == 2

        # XXX: Make sure this is done in a numerically good way
        rval = np.log(
                np.sum(
                    np.exp(lpdfs) * weights,
                    axis=1))
    else:
        # compute the lpdf of each sample under each component
        lpdfs = np.log(
                lognormal_cdf(samples[:, None], mus, sigmas)
                - lognormal_cdf( samples[:, None] - q, mus, sigmas)
                + 1.0e-7)
        assert lpdfs.ndim == 2
        rval = logsum_rows(lpdfs + np.log(weights))
    rval = samples.reshape(_samples.shape)
    return rval


#
# This is the weird heuristic ParzenWindow estimator used for continuous
# distributions in various ways.
#

@scope.define_info(o_len=3)
def adaptive_parzen_normal(mus, prior_mu, prior_sigma):
    """
    A heuristic estimator for the mu and sigma values of a GMM
    TODO: try to find this heuristic in the literature, and cite it - Yoshua
    mentioned the term 'elastic' I think?

    mus - matrix (N, M) of M, N-dimensional component centers
    """
    mus_orig = np.array(mus)
    mus = np.array(mus)
    assert str(mus.dtype) != 'object'
    # XXX: I think prior_mu arrives a list whose length matches the number of
    # new_ids that we're drawing for. VectorizeHelper is the cause, not sure
    # what's the solution.
    if hasattr(prior_mu, '__iter__'):
        prior_mu, = prior_mu
    if hasattr(prior_sigma, '__iter__'):
        prior_sigma, = prior_sigma

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
        if len(mus)>2:
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
    minsigma = prior_sigma / np.sqrt(len(mus))   # XXX: magic formula

    #print 'maxsigma, minsigma', maxsigma, minsigma

    sigma = np.clip(sigma, minsigma, maxsigma)

    weights = np.ones(len(mus), dtype=mus.dtype)
    weights[0] = np.sqrt(1 + len(mus))

    print weights.dtype
    weights = weights / weights.sum()
    return weights, mus, sigma


#
# Adaptive Parzen Samplers
# These produce conditional estimators for various prior distributions
#

# -- Uniform

@adaptive_parzen_sampler('uniform')
def ap_uniform_sampler(obs, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_mu, prior_sigma)
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=None,
            size=size, rng=rng)


@adaptive_parzen_sampler('quniform')
def ap_quniform_sampler(obs, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_mu, prior_sigma)
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=q,
            size=size, rng=rng)


@adaptive_parzen_sampler('loguniform')
def ap_loguniform_sampler(obs, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_mu, prior_sigma)
    rval = scope.LGMM1(weights, mus, sigmas, low=low, high=high,
            size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qloguniform')
def ap_qloguniform_sampler(obs, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_mu, prior_sigma)
    return scope.LGMM1(weights, mus, sigmas, low, high, q=q,
            size=size, rng=rng)


# -- Normal

@adaptive_parzen_sampler('normal')
def ap_normal_sampler(obs, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, size=size, rng=rng)


@adaptive_parzen_sampler('qnormal')
def ap_qnormal_sampler(obs, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, q=q, size=size, rng=rng)


@adaptive_parzen_sampler('lognormal')
def ap_loglognormal_sampler(obs, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qlognormal')
def ap_qlognormal_sampler(obs, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, q=q, size=size, rng=rng)
    return rval


# -- Categorical

@adaptive_parzen_sampler('randint')
def ap_categorical_sampler(obs, upper, size=(), rng=None):
    counts = scope.bincount(obs, minlength=upper)
    # -- add in some prior pseudocounts
    pseudocounts = counts + scope.sqrt(scope.len(obs))
    return scope.categorical(pseudocounts / scope.sum(pseudocounts),
            size=size, rng=rng)


#
# Posterior clone performs symbolic inference on the pyll graph of priors.
#

def posterior_clone(specs, prior_idxs, prior_vals, obs_idxs, obs_vals):
    """
    This method clones a posterior inference graph by iterating forward in
    topological order, and replacing prior random-variables (prior_vals) with
    new posterior distributions that make use of observations (obs_vals).

    Since the posterior is actually factorial, the observation idxs are not
    used.
    """
    expr = pyll.as_apply([specs, prior_idxs, prior_vals])
    nodes = pyll.dfs(expr)
    memo = {}
    obs_memo = dict([(prior_vals[nid], obs_vals[nid]) for nid in prior_vals])
    for node in nodes:
        if node not in memo:
            new_inputs = [memo[arg] for arg in node.inputs()]
            if node in obs_memo:
                fn = adaptive_parzen_samplers[node.name]
                args = [obs_memo[node]] + [memo[a] for a in node.pos_args]
                named_args = [[kw, memo[arg]]
                        for (kw, arg) in node.named_args]
                new_node = fn(*args, **dict(named_args))
            else:
                new_node = node.clone_from_inputs(new_inputs)
            memo[node] = new_node
    post_specs = memo[specs]
    post_idxs = dict([(nid, memo[idxs])
        for nid, idxs in prior_idxs.items()])
    post_vals = dict([(nid, memo[vals])
        for nid, vals in prior_vals.items()])
    return post_specs, post_idxs, post_vals


@scope.define
def idxs_prod(full_idxs, idxs_by_nid, llik_by_nid):
    """Add all of the  log-likelihoods together by id.

    Example arguments:
    full_idxs = [0, 1, ... N-1]
    idxs_by_nid = {'node_a': [1, 3], 'node_b': [3]}
    llik_by_nid = {'node_a': [0.1, -3.3], node_b: [1.0]}

    This would return N elements: [0, 0.1, 0, -2.3, 0, 0, ... ]
    """
    print 'FULL IDXS'
    print full_idxs
    assert len(set(full_idxs)) == len(full_idxs)
    full_idxs = list(full_idxs)
    rval = np.zeros(len(full_idxs))
    assert set(idxs_by_nid.keys()) == set(llik_by_nid.keys())
    for nid in idxs_by_nid:
        idxs = idxs_by_nid[nid]
        llik = llik_by_nid[nid]
        assert np.all(np.asarray(idxs) > 1)
        assert len(set(idxs)) == len(idxs)
        assert len(idxs) == len(llik)
        for ii, ll in zip(idxs, llik):
            rval[full_idxs.index(ii)] += ll
    return rval


class TreeParzenEstimator(BanditAlgo):
    """
    XXX
    """

    # -- suggest this many jobs from prior before attempting to optimize
    n_startup_jobs = 5

    # -- suggest best of this many draws on every iteration
    n_EI_candidates = 256

    # -- fraction of trials to consider as good
    gamma = 0.20

    def __init__(self, bandit):
        BanditAlgo.__init__(self, bandit)

        self.observed = dict(
                idxs=pyll.Literal({'n0': [1]}),
                vals=pyll.Literal({'n0': ['obs_n0_val1']}))
        self.sampled = dict(
                idxs=pyll.Literal({'n0': [99]}),
                vals=pyll.Literal({'n0': ['sample_n0_val99']}))

        if 0:
            print 'PRIOR IDXS_BY_NID'
            for k, v in self.idxs_by_nid.items():
                print k
                print v

            print 'PRIOR VALS_BY_NID'
            for k, v in self.vals_by_nid.items():
                print k
                print v

        post_specs, post_idxs, post_vals = posterior_clone(
                self.vtemplate,    # vectorized clone of bandit template
                self.idxs_by_nid,
                self.vals_by_nid,
                self.observed['idxs'],
                self.observed['vals'])
        self.post_specs = post_specs
        self.post_idxs = post_idxs
        self.post_vals = post_vals
        assert set(post_idxs.keys()) == set(self.post_vals.keys())
        assert set(post_idxs.keys()) == set(self.idxs_by_nid.keys())

        if 0:
            print 'POSTERIOR IDXS_BY_NID'
            for k, v in self.post_idxs.items():
                print k
                print v

            print 'POSTERIOR VALS_BY_NID'
            for k, v in self.post_vals.items():
                print k
                print v

        # -- add the log-likelihood computation graph
        self.post_llik_by_nid = {}
        self.sampled_llik_by_nid = {}
        for nid in post_idxs:
            pv = post_vals[nid]
            sv = self.sampled['vals'][nid]
            def llik_fn (vv):
                lpdf_fn = getattr(scope, pv.name + '_lpdf')
                return lpdf_fn(vv,
                    *pv.pos_args,
                    **dict([(n, a) for n, a in pv.named_args
                        if n not in ('rng', 'size')]))
            self.post_llik_by_nid[nid] = llik_fn(pv)
            self.sampled_llik_by_nid[nid] = llik_fn(sv)

        self.sampled['llik'] = scope.idxs_prod(self.s_new_ids,
                self.sampled['idxs'],
                self.sampled_llik_by_nid)
        self.post_llik = scope.idxs_prod(self.s_new_ids,
                self.post_idxs,
                self.post_llik_by_nid)

    @staticmethod
    def set_iv(iv, idxs, vals):
        iv['idxs'].obj.clear()
        iv['vals'].obj.clear()
        iv['idxs'].obj.update(idxs)
        iv['vals'].obj.update(vals)

    def filter_trials(self, specs, results, miscs, ok_ids):
        # -- purge non-ok trials
        # TODO: Assign pessimistic fantasy scores to running trials (requires
        # support from bandit, to update the result['status'] on start.)
        specs, results, miscs = zip(*[(s, r, m)
            for s, r, m in zip(specs, results, miscs)
            if m['tid'] in ok_ids])

        # -- determine the threshold between good and bad trials
        losses = sorted(map(self.bandit.loss, results, specs))
        loss_thresh_idx = int(self.gamma * len(losses))
        loss_thresh = np.mean(losses[loss_thresh_idx : loss_thresh_idx + 2])

        # -- good trials
        good_specs, good_results, good_miscs = zip(*[(s, r, m)
            for s, r, m in zip(specs, results, miscs)
            if self.bandit.loss(r, s) < loss_thresh])

        # -- bad trials
        bad_specs, bad_results, bad_miscs = zip(*[(s, r, m)
            for s, r, m in zip(specs, results, miscs)
            if self.bandit.loss(r, s) >= loss_thresh])

        msg = 'TreeParzenEstimator splitting %i results at %f (split %i / %i)'
        logger.info(msg % (len(ok_ids), loss_thresh,
            len(good_specs), len(bad_specs)))

        return ((good_specs, good_results, good_miscs),
                (bad_specs, bad_results, bad_miscs))

    def suggest(self, new_ids, specs, results, miscs):
        if len(new_ids) > 1:
            # write a loop to draw new points sequentially
            raise NotImplementedError()
        else:
            return self.suggest1(new_ids, specs, results, miscs)

    def suggest1(self, new_ids, specs, results, miscs):
        assert len(new_ids) == 1
        print self.post_llik

        ok_ids = set([m['tid'] for m, r in zip(miscs, results)
                if r['status'] == STATUS_OK])

        if len(ok_ids) < self.n_startup_jobs:
            logger.info('TreeParzenEstimator warming up %i/%i'
                    % (len(ok_ids), self.n_startup_jobs))
            return BanditAlgo.suggest(self, new_ids, specs, results, miscs)

        good, bad = self.filter_trials(specs, results, miscs, ok_ids)

        # -- Condition on good trials.
        #    Sample and compute log-probability.

        fake_ids = range(max(ok_ids), max(ok_ids) + self.n_EI_candidates)
        #self.new_ids[:] = fake_ids

        self.set_iv(self.observed, *miscs_to_idxs_vals(good[2]))
        # the c_ prefix here is for "candidate"
        c_specs, c_idxs, c_vals, c_good_llik = pyll.rec_eval([
                self.post_specs, self.post_idxs, self.post_vals,
                self.post_llik])

        self.set_iv(self.observed, *miscs_to_idxs_vals(bad[2]))
        self.set_iv(self.sampled, c_idxs, c_vals)
        c_bad_llik = pyll.rec_eval(self.sampled['llik'])

        # -- retrieve the best of the samples and form the return tuple
        winning_pos = np.argmax(c_good_llik - c_bad_llik)
        winning_id = winning_pos + fake_ids[0]

        assert len(new_ids) == 1
        rval_specs = [c_specs[winning_pos]]
        rval_results = [self.bandit.new_result()]
        rval_miscs = [dict(tid=new_ids[0])]
        miscs_update_idxs_vals(rval_miscs, c_idxs, c_vals,
                assert_all_vals_used=False)

        return rval_specs, rval_results, rval_miscs


class TPE_NIPS2011(TreeParzenEstimator):
    n_startup_jobs = 30

