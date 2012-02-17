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
import pyll
from pyll import scope
from pyll.stochastic import implicit_stochastic

from .base import BanditAlgo


adaptive_parzen_samplers = {}
def adaptive_parzen_sampler(name):
    def wrapper(f):
        assert name not in adaptive_parzen_samplers
        adaptive_parzen_samplers[name] = f
        return f
    return wrapper


adaptive_parzen_lpdfs = {}
def adaptive_parzen_lpdf(name):
    def wrapper(f):
        assert name not in adaptive_parzen_lpdfs
        adaptive_parzen_lpdfs[name] = f
        return f
    return wrapper

#
# These are some custom distributions
# that are used to represent posterior distributions.
#

# -- Categorical

def categorical_lpdf(node, sample, kw):
    """
    Return a random integer from 0 .. N-1 inclusive according to the
    probabilities p[0] .. P[N-1].

    This is formally equivalent to numpy.where(multinomial(n=1, p))
    """
    # WARNING: I think the p[-1] is not used, but assumed to be p[:-1].sum()
    s_rstate, p, draw_shape = node.inputs
    return p[sample]


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


@adaptive_parzen_lpdf('GMM1')
@scope.define
def GMM1_lpdf(sample, weights, mus, sigmas, low=None, high=None, q=None):
    sample, weights, mus, sigmas = map(np.asarray,
            (sample, weights, mus, sigmas))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    _sample = sample
    sample = _sample.flatten()

    if low is None and high is None:
        if q is None:
            dist = sample1[:, None] - mus
            mahal = ((dist ** 2) / (sigmas ** 2))
            # mahal shape is (n_samples, n_components)
            Z = np.sqrt(2 * np.pi * sigmas**2)
            coef = weights / Z
            T = -0.5 * mahal
            rmax = np.max(T, axis=1)
            rval = np.log(np.sum(np.exp(T - rmax[:, None]) * coef, axis=1) ) + rmax
            return rval.reshape(_sample.shape)
    raise NotImplementedError()


# -- Mixture of Log-Normals

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

    if not np.all(np.isfinite(samples)):
        logger.warning('overflow in LognormalMixture')
        logger.warning('  mu = %s' % str(mus[active]))
        logger.warning('  sigma = %s' % str(sigmas[active]))
        logger.warning('  samples = %s' % str(samples))

    if q is not None:
        samples = np.floor(samples / q) * q
    samples = np.reshape(np.asarray(samples), size)
    return samples

@adaptive_parzen_lpdf('lognormal_mixture')
@scope.define
def LGMM1_lpdf(node, sample, kw):
    r, weights, mus, sigmas, draw_shape = node.inputs
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    _sample = sample
    if sample.ndim != 1:
        sample = sample.flatten()

    # compute the lpdf of each sample under each component
    lpdfs = lognormal_lpdf_math(sample.dimshuffle(0, 'x'), mus, sigmas)
    assert lpdfs.ndim == 2

    # XXX: Make sure this is done in a numerically good way
    rval = tensor.log(
            tensor.sum(
                tensor.exp(lpdfs) * weights,
                axis=1))

    if not sample is _sample:
        rval = rval.reshape(_sample.shape)
        assert rval.ndim != 1
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
def ap_uniform_sampler(obs, low, high, size=()):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_mu, prior_sigma)
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=None,
            size=size)


@adaptive_parzen_sampler('quniform')
def ap_quniform_sampler(obs, low, high, q, size=()):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_mu, prior_sigma)
    return scope.GMM1(weights, mus, sigmas, low=low, high=high, q=q,
            size=size)


@adaptive_parzen_sampler('loguniform')
def ap_loguniform_sampler(obs, low, high, size=()):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_mu, prior_sigma)
    rval = scope.LGMM1(weights, mus, sigmas, low=low, high=high,
            size=size)
    return rval


@adaptive_parzen_sampler('qloguniform')
def ap_qloguniform_sampler(obs, low, high, q, size=()):
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_mu, prior_sigma)
    return scope.LGMM1(weights, mus, sigmas, low, high, q=q,
            size=size)


# -- Normal

@adaptive_parzen_sampler('normal')
def ap_normal_sampler(obs, mu, sigma, size=()):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, mu, sigma)
    return scope.GMM1(weights, mus, sigmas)


@adaptive_parzen_sampler('qnormal')
def ap_qnormal_sampler(obs, mu, sigma, q, size=()):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, q=q)


@adaptive_parzen_sampler('lognormal')
def ap_loglognormal_sampler(obs, mu, sigma, size=()):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, size=size)
    return rval


@adaptive_parzen_sampler('qlognormal')
def ap_qlognormal_sampler(obs, mu, sigma, q, size=()):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, q=q, size=size)
    return rval


# -- Categorical

@adaptive_parzen_sampler('randint')
def ap_categorical_sampler(obs, upper, size=()):
    counts = scope.bincount(obs, minlength=upper)
    # -- add in some prior pseudocounts
    pseudocounts = counts + scope.sqrt(scope.len(obs))
    return scope.categorical(pseudocounts / scope.sum(pseudocounts),
            size=size)



#
# Posterior clone performs symbolic inference on the pyll graph of priors.
#

def posterior_clone(prior_idxs, prior_vals, obs_idxs, obs_vals):
    """
    This method clones a posterior inference graph by iterating forward in
    topological order, and replacing prior random-variables (prior_vals) with
    new posterior distributions that make use of observations (obs_vals).

    Since the posterior is actually factorial, the observation idxs are not
    used.
    """
    expr = pyll.as_apply([prior_idxs, prior_vals])
    nodes = pyll.dfs(expr)
    memo = {}
    obs_memo = dict([(prior_vals[nid], prior_vals[nid]) for nid in prior_vals])
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
    post_idxs = dict([(nid, memo[idxs])
        for nid, idxs in prior_idxs.items()])
    post_vals = dict([(nid, memo[vals])
        for nid, vals in prior_vals.items()])
    return post_idxs, post_vals


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

        print 'IDXS_BY_NID'
        for k, v in self.idxs_by_nid.items():
            print k
            print v

        print 'VALS_BY_NID'
        for k, v in self.vals_by_nid.items():
            print k
            print v

        post_idxs, post_vals = posterior_clone(
                self.idxs_by_nid,
                self.vals_by_nid,
                self.observed['idxs'],
                self.observed['vals'])
        self.post_idxs = post_idxs
        self.post_vals = post_vals

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

        ok_ids = set([m['tid'] for m, r in zip(miscs, results)
                if r['status'] == STATUS_OK])

        if len(ok_ids) < self.n_startup_jobs:
            logger.info('TreeParzenEstimator warming up %i/%i'
                    % (len(ivls['losses']['ok'].idxs), self.n_startup_jobs))
            return BanditAlgo.suggest(self, new_ids, specs, results, miscs)

        good, bad = self.filter_trials(specs, results, miscs, ok_ids)

        msg = 'TreeParzenEstimator splitting %i results at %f (split %i / %i)'
        logger.info(msg % (len(losses), loss_thresh,
            len(good[0]), len(bad[0])))

        # -- Condition on good trials.
        #    Sample and compute log-probability.

        fake_ids = range(max(ok_ids), max(ok_ids) + self.n_EI_candidates)
        self.new_ids[:] = fake_ids

        set_iv(self.observed, *miscs_to_idxs_vals(good[2]))
        # the c_ prefix here is for "candidate"
        c_specs, c_idxs, c_vals, c_good_llik = pyll.rec_eval()

        set_iv(self.observed, *miscs_to_idxs_vals(bad[2]))
        set_iv(self.sampled, c_idxs, c_vals)
        c_bad_llik = pyll.rec_eval()

        # -- retrieve the best of the samples and form the return tuple
        winning_pos = np.argmax(c_good_llik - c_bad_llik)
        winning_id = winning_pos + fake_ids[0]

        rval_specs = [c_specs[winning_pos]]
        rval_results = [self.bandit.new_result()]
        rval_miscs = dict(tid=new_ids[0])
        miscs_update_idxs_vals(rval_miscs, c_idxs, c_vals)

        return rval_specs, rval_results, rval_miscs


class TPE_NIPS2011(TreeParzenEstimator):
    n_startup_jobs = 30

