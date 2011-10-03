"""
XXX
"""
import numpy
import theano

import base
import ht_dist2

import montetheano as MT
from montetheano.for_theano import ancestors, as_variable

class Random(base.BanditAlgo):
    """Random search director
    """

    def __init__(self, *args, **kwargs):
        base.BanditAlgo.__init__(self, *args, **kwargs)
        self.rng = numpy.random.RandomState(self.seed)

    def suggest(self, X_list, Ys, Y_status, N):
        return [self.bandit.template.sample(self.rng)
                for n in range(N)]


class TheanoRandom(base.TheanoBanditAlgo):
    """Random search director, but testing the machinery that translates
    doctree configurations into sparse matrix configurations.
    """
    def set_bandit(self, bandit):
        base.TheanoBanditAlgo.set_bandit(self, bandit)
        self._sampler = theano.function(
                [self.s_N],
                self.s_idxs + self.s_vals)

    def theano_suggest(self, X_idxs, X_vals, Y, Y_status, N):
        """Ignore X and Y, draw from prior"""
        rvals = self._sampler(N)
        return rvals[:len(rvals)/2], rvals[len(rvals)/2:]


def idxs_vals_take(idxs, vals, elements):
    """Advanced sparse vector indexing by int-list `elements`
    """
    assert len(idxs) == len(vals)
    if idxs.ndim != 1 or 'int' not in str(idxs.dtype):
        raise TypeError('idxs must be int vector', idxs)
    elements = set(elements)
    tf = [(i in elements) for i in idxs]
    return idxs[tf], vals[tf]


def as_variables(*args):
    return [theano.tensor.as_tensor_variable(a) for a in args]


class AdaptiveParzen(theano.Op):
    """
    A heuristic estimator for the mu and sigma values of a GMM
    TODO: try to find this heuristic in the literature, and cite it - Yoshua
    mentioned the term 'elastic' I think?

    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, mu, low, high, minsigma):
        mu, low, high, minsigma = as_variables(mu, low, high, minsigma)
        if mu.ndim == 0:
            raise TypeError()
        if mu.ndim > 1:
            raise NotImplementedError()
        if low.ndim:
            raise TypeError(low)
        if high.ndim:
            raise TypeError(high)
        return theano.gof.Apply(self,
                [mu, low, high, minsigma],
                [mu.type(), mu.type()])

    def perform(self, node, inputs, outstorage):
        mu, low, high, minsigma = inputs
        mu_orig = mu.copy()
        mu = mu.copy()
        if len(mu) == 0:
            mu = numpy.asarray([0.5 * (low + high)])
            sigma = numpy.asarray([0.5 * (high - low)])
        elif len(mu) == 1:
            sigma = numpy.maximum(abs(mu-high), abs(mu-low))
        elif len(mu) >= 2:
            order = numpy.argsort(mu)
            mu = mu[order]
            sigma = numpy.zeros_like(mu)
            sigma[1:-1] = numpy.maximum(
                    mu[1:-1] - mu[0:-2],
                    mu[2:] - mu[1:-1])
            if len(mu)>2:
                lsigma = mu[2] - mu[0]
                usigma = mu[-1] - mu[-3]
            else:
                lsigma = mu[1] - mu[0]
                usigma = mu[-1] - mu[-2]

            sigma[0] = max(mu[0]-low, lsigma)
            sigma[-1] = max(high - mu[-1], usigma)

            # un-sort the mu and sigma
            mu[order] = mu.copy()
            sigma[order] = sigma.copy()

            print mu, sigma

            assert numpy.all(mu_orig == mu)

        outstorage[0][0] = mu
        outstorage[1][0] = numpy.maximum(sigma, minsigma)


class GM_BanditAlgo(base.TheanoBanditAlgo):
    """
    Graphical Model (GM) algo described in NIPS2011 paper.
    """
    n_startup_jobs = 10  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic

    gamma = .2           # fraction of trials to consider as good
                         # this is should in theory be bandit-dependent

    def set_bandit(self, bandit):
        base.TheanoBanditAlgo.set_bandit(self, bandit)

        self.s_obs_idxs = [idx.type() for idx in self.s_idxs]
        self.s_obs_vals = [val.type() for val in self.s_vals]

        self.s_post_idxs, self.s_post_vals = self.s_posterior(
                self.s_obs_idxs, self.s_obs_vals)

        self.s_test_idxs = [idx.type() for idx in self.s_post_idxs]
        self.s_test_vals = [val.type() for val in self.s_post_vals]

        post_ld = idxs_vals_full_log_likelihood(
                self.s_post_idxs, self.s_post_vals,
                self.s_test_idxs, self.s_test_vals)
        self.s_post_ld_idx, self.s_post_ld_val = post_ld

    def s_posterior_helper(self, rv, obs, s_rng):
        """
        Return a posterior RV for rv having observed obs
        """
        # XXX: factor this out of the GM_BanditAlgo so that the density
        # modeling strategy is not coupled to the optimization strategy.
        dist_name = MT.rstreams.rv_dist_name(rv)
        if dist_name == 'normal':
            # GMM
            raise NotImplementedError()
        elif dist_name == 'uniform':
            mus, sigmas = AdaptiveParzen()(obs, -5, 5, 1)
            post_rv = s_rng.gmm(mus, sigmas, draw_shape=rv.shape, ndim=rv.ndim)
            return post_rv
        elif dist_name == 'lognormal':
            # logGMM
            raise NotImplementedError()
        elif dist_name == 'categorical':
            # weighted categorical
            raise NotImplementedError()
        else:
            raise TypeError("unsupported distribution", dist_name)

    def s_posterior(self, s_observed_idxs, s_observed_vals, s_rng=None):
        """Return symbolic RVs representing the posterior sampling density
        """
        assert len(s_observed_idxs) == len(s_observed_vals)
        assert len(s_observed_idxs) == len(self.s_idxs)

        if s_rng is None:
            s_rng = self.seed + 12345

        if isinstance(s_rng, int):
            s_rng = MT.RandomStreams(s_rng)

        new_s_vals = []
        for s_idx, s_val, o_idxs, o_val  in zip(
                self.s_idxs, self.s_vals,
                s_observed_idxs, s_observed_vals):
            # s_observed_idxs can be valid or invalid.
            # They could be invalid, in the sense of describing samples that
            # could not have been drawn from the prior.
            # This code does not try to detect that situation.
            # As long as s_observed_idxs are valid, it doesn't matter what they
            # are for the purpose of density estimation.
            # So we ignore them for now.

            new_s_val = self.s_posterior_helper(s_val, o_val, s_rng)
            new_s_vals.append(new_s_val)

        # At this point, each new_s_val is connected to the original graph
        # formed of s_val nodes (i.e. of the prior).
        #
        # We want each new_s_val to be connected instead to the other new_s_val
        # nodes we just created, corresponding to the posterior values of other
        # random vars.
        #
        # The way to get all the new_s_val nodes hooked up to one another is to
        # use the clone_keep_replacements function.

        blockers = s_observed_idxs + s_observed_vals
        rvs_anc = ancestors(self.s_idxs + self.s_vals, blockers=blockers)
        frontier = [r for r in rvs_anc if r.owner is None or r in blockers]
        _frontier, cloned_post_ivs = MT.shallow_clone.clone_keep_replacements(
                i=frontier,
                o=self.s_idxs + self.s_vals,
                replacements=dict(zip(self.s_vals, new_s_vals)))
        T = len(self.s_idxs)
        assert len(cloned_post_ivs) == 2 * T
        return cloned_post_ivs[:T], cloned_post_ivs[T:]

    def log_density(self, data_X_idxs, data_X_vals, test_X_idxs, test_X_vals):
        """Return an (idx, log_density) such that idxs matches test_X_idxs.
        """
        if not hasattr(self._log_density_fn):
            inputs = (self.s_post_idxs
                    + self.s_post_vals
                    + self.s_test_idxs
                    + self.s_test_vals)
            outputs = [self.s_post_ld_idx, self.s_post_ld_val]
            self._log_density_fn = theano.function(
                    inputs=inputs,
                    outputs=outputs)
        return self._log_density_fn(
                data_X_idxs + data_X_vals + test_X_idxs + test_X_vals)

    def sample(self, data_X_idxs, data_X_vals, N):
        """Return an (idxs, vals) such that idxs accounts for samples 0 to N-1.
        """
        # TODO: factor this out of the GM_BanditAlgo so that the density
        # modeling strategy is not coupled to the optimization strategy.

        # define new symbolic posterior random variables
        # s_idxs_posterior, s_vals_posterior
        if not hasattr(self._sample_fn):
            self._sample_fn = theano.function(
                    inputs = [self.s_N] + self.s_obs_idxs + self.s_obs_vals,
                    outputs = self.s_post_idxs + self.s_post_vals
                    )
        flat_rval = self._sample_fn(*([N] + data_X_idxs + data_X_vals))
        L = len(self.s_post_idxs)
        # return idxs, vals
        r_idxs, r_vals = flat_rval[:L], flat_rval[L:]
        assert len(r_vals) == len(self.s_post_vals)
        return r_idxs, r_vals

    def theano_suggest(self, X_idxs, X_vals, Y, Y_status, N):
        ok_idxs = [i for i, s in enumerate(Y_status) if s == 'ok']

        ylist = list(Y[ok_idxs])
        ylist.sort()
        if len(ylist) < self.n_startup_jobs:
            raise ValueError('insufficient jobs to estimate EI')
        y_thresh_idx = int(self.gamma*.999 * len(ylist))
        y_thresh = .5 * ylist[y_thresh_idx] + .5 * ylist[y_thresh_idx+1]

        # y_thresh is the boundary between 'good' and 'bad' regions of the
        # search space.

        # define something like the l(x) and g(x) functions
        # from the paper draft
        l_idxs, l_vals = idxs_vals_take(X_idxs, X_vals, self.y < y_thresh)
        g_idxs, g_vals = idxs_vals_take(X_idxs, X_vals, self.y >= y_thresh)

        # To "optimize" EI we just draw a pile of samples from the density
        # of good points and then just take the best of those.
        cand_idxs, cand_vals = self.sample(l_idxs, l_vals,
                N=self.n_EI_candidates_to_draw)
        _, cand_ld_l = self.log_density(l_idxs, l_vals, cand_idxs, cand_vals)
        _, cand_ld_g = self.log_density(g_idxs, g_vals, cand_idxs, cand_vals)
        cand_EI = cand_ld_l - cand_ld_g

        best_N_candidates = numpy.argsort(cand_EI)[:N]

        r_idxs, r_vals = idxs_vals_take(cand_idxs, cand_vals, best_N_candidates)

        return r_idxs, r_vals


class GP_BanditAlgo(base.BanditAlgo):
    pass
