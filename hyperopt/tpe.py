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



@pyll.stochastic.implicit_stochastic
@pyll.scope.define
def GMM1(weights, mus, sigmas, rng=None, size=()):
    """Return samples from a 1-D Gaussian Mixture Model"""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    assert len(weights) == len(mus) == len(sigmas)
    n_samples = np.prod(size)
    active = np.argmax(rng.multinomial(1, weights, (n_samples,)), axis=1)
    assert len(active) == n_samples
    samples = rng.normal(loc=mus[active], scale=sigmas[active])
    rval = np.asarray(np.reshape(samples, size))
    return rval


@adaptive_parzen_lpdf('GMM1')
@pyll.scope.define
def GMM1_lpdf(sample, weights, mus, sigmas):
    sample, weights, mus, sigmas = map(np.asarray,
            (sample, weights, mus, sigmas))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    if sample.ndim != 1:
        sample1 = sample.flatten()
    else:
        sample1 = sample

    dist = sample1[:, None] - mus
    mahal = ((dist ** 2) / (sigmas ** 2))
    # mahal shape is (n_samples, n_components)
    Z = np.sqrt(2 * np.pi * sigmas**2)
    coef = weights / Z
    T = -0.5 * mahal
    rmax = np.max(T, axis=1)
    rval = np.log(np.sum(np.exp(T - rmax[:, None]) * coef, axis=1) ) + rmax
    return rval.reshape(sample.shape)


@pyll.scope.define_info(o_len=3)
def adaptive_parzen_normal(mus, prior_mu, prior_sigma):
    """
    A heuristic estimator for the mu and sigma values of a GMM
    TODO: try to find this heuristic in the literature, and cite it - Yoshua
    mentioned the term 'elastic' I think?

    mus - matrix (N, M) of M, N-dimensional component centers
    """
    mus_orig = mus.copy()
    mus = mus.copy()
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

    weights /= weights.sum()
    return weights, mus, sigma


@adaptive_parzen_sampler('normal')
def ap_normal_sampler(obs, mu, sigma):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, mu, sigma)
    return scope.GMM1(weights, mus, sigmas)


class TreeParzenEstimator(BanditAlgo):

    # -- suggest this many jobs from prior before attempting to optimize
    n_startup_jobs = 5

    # -- suggest best of this many draws on every iteration
    n_EI_candidates = 256

    # -- fraction of trials to consider as good
    gamma = 0.20

    def __init__(self, bandit, good_estimator, bad_estimator):
        BanditAlgo.__init__(self, bandit)
        self.good_estimator = good_estimator
        self.bad_estimator = bad_estimator

        self.observed = dict(idxs=pyll.Literal({}), vals=pyll.literal({}))
        self.sampled = dict(idxs=pyll.Literal({}), vals=pyll.literal({}))


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
        def set_iv(iv, idxs, vals):
            iv['idxs'].clear()
            iv['vals'].clear()
            iv['idxs'].update(idxs)
            iv['vals'].update(vals)

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


class TPE(TreeParzenEstimator):
    def __init__(self, bandit):
        TreeParzenEstimator.__init__(self, bandit,
            good_estimator=IndependentAdaptiveParzenEstimator(),
            bad_estimator=IndependentAdaptiveParzenEstimator())

class TPE_NIPS2011(TPE):
    n_startup_jobs = 30




@adaptive_parzen_sampler('uniform')
def asdf():
    low, high = prior.vals.owner.inputs[2:4]
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)
    weights, mus, sigmas = AdaptiveParzen()(obs.vals,
            prior_mu, prior_sigma)
    post_rv = s_rng.BGMM1(weights, mus, sigmas, low, high,
            draw_shape=prior.vals.shape,
            ndim=prior.vals.ndim,
            dtype=prior.vals.dtype)
    return post_rv
@adaptive_parzen_sampler('exp_normal')
def asdf():
    prior_mu, prior_sigma = prior.vals.owner.inputs[2:4]
    weights, mus, sigmas = AdaptiveParzen()(
            tensor.log(tensor.maximum(obs.vals, 1.0e-8)),
            prior_mu, prior_sigma)
    post_rv = s_rng.lognormal_mixture(weights, mus, sigmas,
            draw_shape=prior.vals.shape,
            ndim=prior.vals.ndim,
            dtype=prior.vals.dtype)
    return post_rv
@adaptive_parzen_sampler('quantized_log_normal')
def asdf():
    prior_mu, prior_sigma, step = prior.vals.owner.inputs[2:5]
    weights, mus, sigmas = AdaptiveParzen()(
            tensor.log(obs.vals),
            prior_mu, prior_sigma)
    post_rv = s_rng.quantized_lognormal_mixture(
            weights, mus, sigmas, step,
            draw_shape=prior.vals.shape,
            ndim=prior.vals.ndim,
            dtype=prior.vals.dtype)
    return post_rv

@adaptive_parzen_sampler('categorical')
def asdf():
    prior_strength = self.categorical_prior_strength
    p = prior.vals.owner.inputs[1]
    if p.ndim != 1:
        raise TypeError()
    prior_counts = p * p.shape[0] * tensor.sqrt(obs.vals.shape[0])
    pseudocounts = tensor.inc_subtensor(
            (prior_strength * prior_counts)[obs.vals],
            1)
    post_rv = s_rng.categorical(
            p=pseudocounts / pseudocounts.sum(),
            draw_shape = prior.vals.shape)
    return post_rv


