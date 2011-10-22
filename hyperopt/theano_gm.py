"""
Graphical model (GM)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import sys
import logging
logger = logging.getLogger(__name__)

import numpy
import theano
from theano import tensor

import base
import ht_dist2

import montetheano
from montetheano.for_theano import ancestors
from montetheano.for_theano import argsort
from montetheano.for_theano import as_variable
from montetheano.for_theano import clone_keep_replacements
from montetheano.for_theano import where

import idxs_vals_rnd
from idxs_vals_rnd import IdxsVals
from idxs_vals_rnd import IdxsValsList

from theano_bandit_algos import TheanoBanditAlgo


class GM_BanditAlgo(TheanoBanditAlgo):
    """
    Graphical Model (GM) algo described in NIPS2011 paper.
    """
    n_startup_jobs = 30  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic

    n_EI_candidates = 256

    gamma = 0.15         # fraction of trials to consider as good
                         # this is should in theory be bandit-dependent

    def __init__(self, bandit, good_estimator, bad_estimator):
        TheanoBanditAlgo.__init__(self, bandit)
        self.good_estimator = good_estimator
        self.bad_estimator = bad_estimator

    def __getstate__(self):
        rval = dict(self.__dict__)
        for name in '_helper', 'helper_locals', '_prior_sampler':
            if name in rval:
                del rval[name]
        return rval

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        # this allows loading of old pickles
        # from before the current implementation
        # of __getstate__
        for name in '_helper', 'helper_locals', '_prior_sampler':
            if hasattr(self, name):
                delattr(self, name)

    def build_helpers(self, do_compile=True, mode=None):
        s_prior = IdxsValsList.fromlists(self.s_idxs, self.s_vals)
        s_obs = s_prior.new_like_self()

        # y_thresh is the boundary between 'good' and 'bad' regions of the
        # search space.
        y_thresh = tensor.scalar()

        yvals = tensor.vector()
        n_to_draw = self.s_N
        n_to_keep = tensor.iscalar()

        s_rng = montetheano.RandomStreams(self.seed + 9)

        GE = self.good_estimator
        BE = self.bad_estimator

        Gobs = s_obs.symbolic_take(where(yvals < y_thresh))
        Bobs = s_obs.symbolic_take(where(yvals >= y_thresh))

        # To "optimize" EI we just draw a pile of samples from the density
        # of good points and then just take the best of those.
        Gsamples = GE.posterior(s_prior, Gobs, s_rng)
        Bsamples = BE.posterior(s_prior, Bobs, s_rng)

        G_ll = GE.log_likelihood(Gsamples, Gsamples,
                llik = tensor.zeros((n_to_draw,)))
        B_ll = BE.log_likelihood(Bsamples, Gsamples,
                llik = tensor.zeros((n_to_draw,)))

        # subtract B_ll from G_ll
        log_EI = G_ll - B_ll
        keep_idxs = argsort(log_EI)[-n_to_keep:]

        # store all these vars for the unittests
        self.helper_locals = locals()
        del self.helper_locals['self']

        if do_compile:
            self._helper = theano.function(
                [n_to_draw, n_to_keep, y_thresh, yvals] + s_obs.flatten(),
                (Gsamples.symbolic_take(keep_idxs).flatten()
                    + Gobs.flatten()
                    + Bobs.flatten()
                    ),
                allow_input_downcast=True,
                mode=mode,
                )

            self._prior_sampler = theano.function(
                    [n_to_draw],
                    s_prior.flatten(),
                    mode=mode)

    def suggest_from_prior(self, N):
        rvals = self._prior_sampler(N)
        return IdxsValsList.fromflattened(rvals)

    def suggest_from_model(self, ivls, N):
        ylist = numpy.asarray(sorted(ivls['losses']['ok'].vals), dtype='float')
        y_thresh_idx = int(self.gamma * len(ylist))
        y_thresh = ylist[y_thresh_idx : y_thresh_idx + 2].mean()

        logger.info('GM_BanditAlgo splitting results at y_thresh = %f'
                % y_thresh)
        logger.info('GM_BanditAlgo keeping %i results as good'
                % y_thresh_idx)
        logger.info('GM_BanditAlgo keeping %i results as bad'
                % (len(ylist) - y_thresh_idx))
        logger.info('GM_BanditAlgo good scores: %s'
                % str(ylist[:y_thresh_idx]))

        x_all = ivls['x_IVLs']['ok'].as_list()
        y_all_iv = ivls['losses']['ok'].as_list()

        for pseudo_bad_status in 'new', 'running':
            logger.info('GM_BanditAlgo assigning bad scores to %i new jobs'
                    % len(ivls['losses'][pseudo_bad_status].idxs))
            x_all.stack(ivls['x_IVLs'][pseudo_bad_status])
            y_all_iv.stack(IdxsVals(
                ivls['losses'][pseudo_bad_status].idxs,
                [y_thresh + 1] * len(ivls['losses'][pseudo_bad_status].idxs)))

        # renumber the configurations in x_all to be 0 .. (n_train - 1)
        idmap = y_all_iv.reindex()
        idmap = x_all.reindex(idmap)

        assert y_all_iv.idxset() == x_all.idxset()

        assert numpy.all(y_all_iv.idxs == numpy.arange(len(y_all_iv.idxs)))

        y_all = y_all_iv.as_numpy(vdtype=theano.config.floatX).vals
        x_all = x_all.as_numpy_floatX()

        logger.info('GM_BanditAlgo drawing %i candidates'
                % self.n_EI_candidates)

        helper_rval = self._helper(self.n_EI_candidates, N,
            y_thresh, y_all, *x_all.flatten())
        assert len(helper_rval) == 6 * len(x_all)

        keep_flat = helper_rval[:2 * len(x_all)]
        Gobs_flat = helper_rval[2 * len(x_all): 4 * len(x_all)]
        Bobs_flat = helper_rval[4 * len(x_all):]
        assert len(keep_flat) == len(Gobs_flat) == len(Bobs_flat)

        Gobs = IdxsValsList.fromflattened(Gobs_flat)
        Bobs = IdxsValsList.fromflattened(Bobs_flat)

        # guard against book-keeping error
        # ensure that all observations were counted as either good or bad
        gis = Gobs.idxset()
        bis = Bobs.idxset()
        xis = x_all.idxset()
        assert len(xis) == len(y_all)
        assert gis.union(bis) == xis
        assert gis.intersection(bis) == set()

        return IdxsValsList.fromflattened(keep_flat)

    def suggest(self, trials, results, N):
        if not hasattr(self, '_prior_sampler'):
            self.build_helpers()
            assert hasattr(self, '_prior_sampler')

        ivls = self.idxs_vals_by_status(trials, results)
        if len(ivls['losses']['ok'].idxs) < self.n_startup_jobs:
            logger.info('GM_BanditAlgo warming up %i/%i'
                    % (len(ivls['losses']['ok'].idxs), self.n_startup_jobs))
            return self.suggest_ivl(self.suggest_from_prior(N))
        else:
            return self.suggest_ivl(self.suggest_from_model(ivls, N))


def AdaptiveParzenGM():
    GE = idxs_vals_rnd.IndependentAdaptiveParzenEstimator()
    BE = idxs_vals_rnd.IndependentAdaptiveParzenEstimator()
    rval = GM_BanditAlgo(
            good_estimator=GE,
            bad_estimator=BE)
    return rval
