"""
XXX
"""
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


class GM_BanditAlgo(base.TheanoBanditAlgo):
    """
    Graphical Model (GM) algo described in NIPS2011 paper.
    """
    n_startup_jobs = 30  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic

    n_EI_candidates = 256

    gamma = 0.15         # fraction of trials to consider as good
                         # this is should in theory be bandit-dependent

    def __init__(self, good_estimator, bad_estimator):
        base.TheanoBanditAlgo.__init__(self)
        self.good_estimator = good_estimator
        self.bad_estimator = bad_estimator

    def set_bandit(self, bandit):
        base.TheanoBanditAlgo.set_bandit(self, bandit)

    def build_helpers(self, do_compile=True, mode=None):
        s_prior = IdxsValsList([IdxsVals(i,v)
            for i, v in zip(self.s_idxs, self.s_vals)])

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

        Gobs = s_obs.take(where(yvals < y_thresh))
        Bobs = s_obs.take(where(yvals >= y_thresh))

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
                (Gsamples.take(keep_idxs).flatten()
                    + [log_EI]
                    + Gsamples.flatten()),
                allow_input_downcast=True,
                mode=mode,
                )

            self._prior_sampler = theano.function(
                    [n_to_draw],
                    self.s_idxs + self.s_vals,
                    mode=mode)

    def theano_suggest(self, X_idxs, X_vals, Y, Y_status, N):
        if not hasattr(self, '_prior_sampler'):
            self.build_helpers()
            assert hasattr(self, '_prior_sampler')

        ok_idxs = [i for i, s in enumerate(Y_status) if s == 'ok']

        assert len(X_idxs) == len(X_vals)

        ylist = [float(yi) for yi in numpy.asarray(Y)[ok_idxs]]

        # if there are not enough completed jobs to estimate EI
        # then we return draws from the bandit's prior
        if len(ylist) < self.n_startup_jobs:
            logger.info('GM_BanditAlgo warming up %i/%i'
                    % (len(ylist), self.n_startup_jobs))
            rvals = self._prior_sampler(N)
            # rvals here are idx0, idx1, ... val0, val1, ...
            return rvals[:len(rvals)/2], rvals[len(rvals)/2:]

        ylist.sort()
        y_thresh_idx = int(self.gamma*.999 * len(ylist))
        y_thresh = .5 * ylist[y_thresh_idx] + .5 * ylist[y_thresh_idx+1]

        logger.info('GM_BanditAlgo splitting results at y_thresh = %f'
                % y_thresh)
        logger.info('GM_BanditAlgo keeping %i results as good'
                % y_thresh_idx)
        logger.info('GM_BanditAlgo keeping %i results as bad'
                % (len(ylist) - y_thresh_idx))
        logger.info('GM_BanditAlgo good scores: %s'
                % str(ylist[:y_thresh_idx]))

        X_iv_zip = []
        for i, v in zip(X_idxs, X_vals):
            X_iv_zip.extend([i, v])

        logger.info('GM_BanditAlgo drawing %i candidates'
                % self.n_EI_candidates)
        logger.info('GM_BanditAlgo lens %i %i'
                % (len(Y), len(ylist)))
        print >> sys.stderr, zip(Y, Y_status)


        helper_rval = self._helper(self.n_EI_candidates, N,
            y_thresh, Y, *X_iv_zip)

        # rvals here are idx0, val0, idx1, val1, ...
        return (helper_rval[:2 * len(X_idxs):2],
                helper_rval[1:2 * len(X_idxs):2])


def AdaptiveParzenGM():
    GE = idxs_vals_rnd.IndependentAdaptiveParzenEstimator()
    BE = idxs_vals_rnd.IndependentAdaptiveParzenEstimator()
    rval = GM_BanditAlgo(
            good_estimator=GE,
            bad_estimator=BE)
    return rval

class GP_BanditAlgo(base.BanditAlgo):
    pass
