"""
Graphical model (GM)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
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

EPS = 1e-12


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
def categorical_lpdf(sample, p):
    """
    Return a random integer from 0 .. N-1 inclusive according to the
    probabilities p[0] .. P[N-1].

    This is formally equivalent to np.where(multinomial(n=1, p))
    """
    return np.log(np.asarray(p)[sample])


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
            if low <= draw < high:
                samples.append(draw)
    samples = np.reshape(np.asarray(samples), size)
    #print 'SAMPLES', samples
    if q is None:
        return samples
    else:
        return np.ceil(samples / q) * q

@scope.define
def normal_cdf(x, mu, sigma):
    top = (x - mu)
    bottom = np.maximum(np.sqrt(2) * sigma, EPS)
    z = top / bottom
    return 0.5 * (1 + erf(z))

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
        mahal = (dist / np.maximum(sigmas, EPS) ) ** 2
        # mahal shape is (n_samples, n_components)
        Z = np.sqrt(2 * np.pi * sigmas**2)
        coef = weights / Z / p_accept
        rval = logsum_rows(- 0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros(samples.shape, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            prob += w * normal_cdf(samples, mu, sigma)
            prob -= w * normal_cdf(samples - q, mu, sigma)
        rval = np.log(prob) - np.log(p_accept)

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
    E = 0.5 * ((np.log(x) - mu) / sigma)**2
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
def LGMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
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
        samples = np.maximum(np.ceil(samples / q) * q, q)
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
            prob += w * lognormal_cdf(samples, mu, sigma)
            prob -= w * lognormal_cdf(samples - q, mu, sigma)
        rval = np.log(prob) - np.log(p_accept)
    rval.shape = _samples.shape
    return rval


#
# This is the weird heuristic ParzenWindow estimator used for continuous
# distributions in various ways.
#

@scope.define_info(o_len=3)
def adaptive_parzen_normal(mus, prior_weight, prior_mu, prior_sigma):
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
    # -- magic formula:
    minsigma = prior_sigma / np.sqrt(1 + len(mus))

    #print 'maxsigma, minsigma', maxsigma, minsigma
    sigma = np.clip(sigma, minsigma, maxsigma)

    weights = np.ones(len(mus), dtype=mus.dtype)
    weights[0] = prior_weight * np.sqrt(1 + len(mus))

    #print weights.dtype
    weights = weights / weights.sum()
    if 0:
        print 'WEIGHTS', weights
        print 'MUS', mus
        print 'SIGMA', sigma

    return weights, mus, sigma


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
def ap_loguniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_weight, prior_mu, prior_sigma)
    rval = scope.LGMM1(weights, mus, sigmas, low=low, high=high,
            size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qloguniform')
def ap_qloguniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs,
            prior_weight, prior_mu, prior_sigma)
    return scope.LGMM1(weights, mus, sigmas, low, high, q=q,
            size=size, rng=rng)


# -- Normal

@adaptive_parzen_sampler('normal')
def ap_normal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, size=size, rng=rng)


@adaptive_parzen_sampler('qnormal')
def ap_qnormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, q=q, size=size, rng=rng)


@adaptive_parzen_sampler('lognormal')
def ap_loglognormal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_weight, mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qlognormal')
def ap_qlognormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_weight, mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, q=q, size=size, rng=rng)
    return rval


# -- Categorical

@adaptive_parzen_sampler('randint')
def ap_categorical_sampler(obs, prior_weight, upper, size=(), rng=None):
    counts = scope.bincount(obs, minlength=upper)
    # -- add in some prior pseudocounts
    prior = scope.sqrt(1 + scope.len(obs)) / upper
    pseudocounts = counts + prior_weight * prior
    return scope.categorical(pseudocounts / scope.sum(pseudocounts),
            size=size, rng=rng)


#
# Posterior clone performs symbolic inference on the pyll graph of priors.
#

@scope.define
def ap_filter_trials(o_idxs, o_vals, l_idxs, l_vals, gamma, above_or_below):
    """Return the elements of o_vals that correspond to trials whose losses
    were above gamma, or below gamma.
    """
    o_idxs, o_vals = map(np.asarray, [o_idxs, o_vals])

    # o_loss: the loss for every observation in o_idxs 
    loss_dict = dict(zip(l_idxs, l_vals))
    o_loss = [loss_dict[ii] for ii in o_idxs]

    # Splitting is done this way to cope with duplicate loss values.
    l_sort = np.argsort(o_loss)
    loss_thresh_idx = int(np.ceil(gamma * len(l_sort)))

    # -- keep is the subset of o_idxs corresponding to examples
    #    either above or below the loss threshold.
    if above_or_below == 'above_gamma':
        keep = l_sort[loss_thresh_idx:]
    elif above_or_below == 'below_gamma':
        keep = l_sort[:loss_thresh_idx]
    else:
        raise ValueError(above_or_below)
    rval = o_vals[keep]
    #print 'KEEP', rval
    if rval.ndim != 1:
        raise TypeError('messed up', (rval.ndim, ovals, keep))
    return rval


def build_posterior(specs, prior_idxs, prior_vals, obs_idxs, obs_vals,
        oloss_idxs, oloss_vals, oloss_gamma, above_or_below, prior_weight):
    """
    This method clones a posterior inference graph by iterating forward in
    topological order, and replacing prior random-variables (prior_vals) with
    new posterior distributions that make use of observations (obs_vals).

    """
    assert all(isinstance(arg, pyll.Apply)
            for arg in [oloss_idxs, oloss_vals, oloss_gamma, above_or_below])

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
        obs = scope.ap_filter_trials(obs_idxs[nid], obs_vals[nid],
            oloss_idxs, oloss_vals, oloss_gamma, above_or_below)
        obs_memo[prior_vals[nid]] = [obs, prior_weight]
    for node in nodes:
        if node not in memo:
            new_inputs = [memo[arg] for arg in node.inputs()]
            if node in obs_memo:
                # -- this case corresponds to an observed Random Var
                # node.name is a distribution like "normal", "randint", etc.
                fn = adaptive_parzen_samplers[node.name]
                args = obs_memo[node] + [memo[a] for a in node.pos_args]
                named_args = [[kw, memo[arg]]
                        for (kw, arg) in node.named_args]
                new_node = fn(*args, **dict(named_args))
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
def idxs_prod(full_idxs, idxs_by_nid, llik_by_nid):
    """Add all of the  log-likelihoods together by id.

    Example arguments:
    full_idxs = [0, 1, ... N-1]
    idxs_by_nid = {'node_a': [1, 3], 'node_b': [3]}
    llik_by_nid = {'node_a': [0.1, -3.3], node_b: [1.0]}

    This would return N elements: [0, 0.1, 0, -2.3, 0, 0, ... ]
    """
    #print 'FULL IDXS'
    #print full_idxs
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

    # -- the prior takes a weight in the Parzen mixture
    #    that is the sqrt of the number of observations
    #    times this number.
    prior_weight = 1.0

    # -- suggest best of this many draws on every iteration
    n_EI_candidates = 256

    # -- fraction of trials to consider as good
    gamma = 0.22

    def __init__(self, bandit,
            gamma=gamma,
            prior_weight=prior_weight,
            n_EI_candidates=n_EI_candidates,
            **kwargs):
        BanditAlgo.__init__(self, bandit, **kwargs)
        self.gamma = gamma
        self.prior_weight = prior_weight
        self.n_EI_candidates = n_EI_candidates
        self.s_prior_weight = pyll.Literal(float(self.prior_weight))

        # -- these dummy values will be replaced in suggest1() and never used
        self.observed = dict(
                idxs=pyll.Literal(),
                vals=pyll.Literal())
        self.observed_loss = dict(
                idxs=pyll.Literal(),
                vals=pyll.Literal())

        self.post_above = self.init_posterior('above_gamma')
        self.post_below = self.init_posterior('below_gamma')

        # -- llik of RVs from the below dist under the above dist
        self.post_above['llik'] = self.llik(self.post_below, self.post_above)
        # -- llik of RVs from the below dist under the below dist
        self.post_below['llik'] = self.llik(self.post_below, self.post_below)

        if 0:
            print 'PRIOR IDXS_BY_NID'
            for k, v in self.idxs_by_nid.items():
                print k
                print v

            print 'PRIOR VALS_BY_NID'
            for k, v in self.vals_by_nid.items():
                print k
                print v

    def init_posterior(self, rel_to_gamma):
        specs, idxs, vals = build_posterior(
                self.vtemplate,    # vectorized clone of bandit template
                self.idxs_by_nid,  # this dict and next represent prior distributions
                self.vals_by_nid,  # 
                self.observed['idxs'],  # these dicts, represent observations
                self.observed['vals'],
                self.observed_loss['idxs'],
                self.observed_loss['vals'],
                pyll.Literal(self.gamma),
                pyll.Literal(rel_to_gamma),
                self.s_prior_weight
                )
        return dict(specs=specs, idxs=idxs, vals=vals)

    def llik(self, obs, density):
        """Add log-likelihood functions for the values"""
        llik = {}
        assert set(obs.keys()) == set(density.keys())
        for nid in obs['vals']:
            dvals = density['vals'][nid]
            lpdf_fn = getattr(scope, dvals.name + '_lpdf')
            args = [obs['vals'][nid]] + dvals.pos_args
            kwargs = dict([(n, a) for n, a in dvals.named_args
                        if n not in ('rng', 'size')])
            llik[nid] = lpdf_fn(*args, **kwargs)
        rval = scope.idxs_prod(self.s_new_ids, obs['idxs'], llik)
        return rval

    def suggest(self, new_ids, trials):
        if len(new_ids) > 1:
            # write a loop to draw new points sequentially
            raise NotImplementedError()
        else:
            return self.suggest1(new_ids, trials)

    def suggest1(self, new_ids, trials):
        """Suggest a single new document"""
        assert len(new_ids) == 1
        new_id, = new_ids
        #print self.post_llik

        bandit = self.bandit
        ok_docs = [d for d in trials.trials
                if bandit.status(d['result'], d['spec']) == STATUS_OK]
        orig_docs = [d for d in ok_docs if 'from_tid' not in d['misc']]
        inj_docs  = [d for d in ok_docs if 'from_tid' in d['misc']]
        orig_losses = [bandit.loss(d['result'], d['spec']) for d in orig_docs]
        min_loss = min(orig_losses)
        docs = orig_docs + [d for d in inj_docs
                            if bandit.loss(d['result'], d['spec']) < min_loss]
        logger.info('TPE using %i orig trials plus %i of %i injected trials' % (
            len(orig_docs), len(docs) - len(orig_docs), len(inj_docs)))

        tids = [d['tid'] for d in docs]

        #    Sample and compute log-probability.
        if tids:
            # -- the +2 co-ordinates with an assertion above
            #    to ensure that fake ids are used during sampling
            fake_id_0 = max(max(tids), new_id) + 2
        else:
            fake_id_0 = new_id + 2
        fake_ids = range(fake_id_0, fake_id_0 + self.n_EI_candidates)
        self.new_ids[:] = fake_ids

        # -- this dictionary will map pyll nodes to the values
        #    they should take during the evaluation of the pyll program
        memo = {}

        o_idxs_d, o_vals_d = miscs_to_idxs_vals(
            [d['misc'] for d in docs], keys=self.idxs_by_nid.keys())
        memo[self.observed['idxs']] = o_idxs_d
        memo[self.observed['vals']] = o_vals_d

        memo[self.observed_loss['idxs']] = tids
        memo[self.observed_loss['vals']] = \
                [bandit.loss(d['result'], d['spec']) for d in docs]

        R = pyll.rec_eval(
                dict(
                    specs=self.post_below['specs'],
                    above_llik=self.post_above['llik'],
                    below_llik=self.post_below['llik'],
                    idxs=self.post_below['idxs'],
                    vals=self.post_below['vals'],
                    ),
                memo=memo)

        for k in 'specs', 'above_llik', 'below_llik':
            assert len(R[k]) == self.n_EI_candidates

        # -- retrieve the best of the samples and form the return tuple
        llik_diff = R['below_llik'] - R['above_llik']
        winning_pos = np.argmax(llik_diff)
        winning_fake_id = winning_pos + fake_ids[0]

        rval_specs = [R['specs'][winning_pos]]
        rval_results = [bandit.new_result()]
        rval_miscs = [dict(tid=new_id, cmd=self.cmd, workdir=self.workdir)]

        miscs_update_idxs_vals(rval_miscs, R['idxs'], R['vals'],
                idxs_map={winning_fake_id: new_id},
                assert_all_vals_used=False)
        rval_docs = trials.new_trial_docs(new_ids,
                rval_specs, rval_results, rval_miscs)

        if 0:
            foo = np.argsort(llik_diff)
            for j in range(self.n_EI_candidates):
                i = foo[j]
                print '%i\tx=%f\tll=(%f)-(%f)=%f' % (
                        i, R['specs'][i]['x'],
                        R['below_llik'][i],
                        R['above_llik'][i],
                        llik_diff[i],
                        )

                print 'SUGGESTION', rval_docs[0]
        return rval_docs


