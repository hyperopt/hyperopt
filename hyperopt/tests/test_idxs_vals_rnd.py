import unittest
import sys

import numpy

import theano
import montetheano
from montetheano.for_theano import where

import hyperopt
import hyperopt.bandits
from hyperopt.bandit_algos import GM_BanditAlgo, TheanoRandom
from hyperopt.experiments import SerialExperiment
import idxs_vals_rnd
from idxs_vals_rnd import IdxsValsList
from idxs_vals_rnd import IndependentAdaptiveParzenEstimator

from hyperopt.ht_dist2 import one_of, rSON2, uniform

def ops(fn, OpCls):
    if isinstance(fn, list):
        return [v.owner for v in montetheano.for_theano.ancestors(fn)
                if v.owner and isinstance(v.owner.op, OpCls)]
    else:
        return [ap for ap in fn.maker.env.toposort()
            if isinstance(ap.op, OpCls)]


def categoricals(fn):
    return ops(fn, montetheano.distributions.Categorical)


class NestedUniform(hyperopt.bandits.Base):
    """
    Problem in which a uniform applies only to some of the examples.
    """

    loss_target = -2  # best score is +2

    def __init__(self):
        hyperopt.bandits.Base.__init__(self,
                one_of(
                    rSON2(
                        'kind', 'raw'),
                    rSON2(
                        'kind', 'negcos',
                        'amp', uniform(0, 1))))

    def score(self, pt):
        r = numpy.random.uniform()
        if pt['kind'] == 'negcos':
            r += pt['amp']
        return r


class IndependentNullEstimator(idxs_vals_rnd.IndependentNodeTreeEstimator):
    """Dummy node estimator that keeps graphs small.
    Useful for debugging IndependentNodeTreeEstimator
    """

    def s_posterior_helper(self, prior, obs, s_rng):
        try:
            dist_name = montetheano.rstreams.rv_dist_name(prior.vals)
        except:
            print >> sys.stderr, 'problem with', prior.vals
            raise

        if dist_name == 'normal':
            if obs.vals.ndim == 1:
                return s_rng.normal(
                        mu=obs.vals.mean(),
                        sigma=0.75,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
            else:
                raise NotImplementedError()
        elif dist_name == 'uniform':
            if obs.vals.ndim == 1:
                low, high = prior.vals.owner.inputs[2:4]
                return s_rng.uniform(
                        low=obs.vals.min(),
                        high=high,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
            else:
                raise NotImplementedError()
        elif dist_name == 'lognormal':
            raise NotImplementedError()
        elif dist_name == 'categorical':
            if obs.vals.ndim == 1:
                pseudocounts = prior.vals.owner.inputs[1] + obs.vals.sum()
                post_rv = s_rng.categorical(
                        p=pseudocounts / pseudocounts.sum(),
                        draw_shape = prior.vals.shape)
                return post_rv
            else:
                raise NotImplementedError()
        else:
            raise TypeError("unsupported distribution", dist_name)


class TestIndependentNodeTreeEstimator(unittest.TestCase):
    def setUp(self):
        self.TE = IndependentNullEstimator()
        self.bandit = NestedUniform()
        self.experiment = SerialExperiment(
            bandit=self.bandit,
            bandit_algo=GM_BanditAlgo(
                    good_estimator=IndependentNullEstimator(),
                    bad_estimator=IndependentNullEstimator()))
        self.experiment.set_bandit()

        self.s_rng = montetheano.RandomStreams(123)
        prior_idxs, prior_vals, s_N = self.bandit.template.theano_sampler(self.s_rng)
        #print prior_idxs
        #print prior_vals
        self.prior = IdxsValsList.fromlists(
                [i for i in prior_idxs if i is not None],
                [v for v in prior_vals if v is not None])
        self.s_N = s_N
        self.observations = self.prior.new_like_self()
        for i, o in enumerate(self.observations):
            o.idxs.name = 'Obs_idxs{%i}' % i
            o.vals.name = 'Obs_vals{%i}' % i

    def test_posterior_op_count(self):
        posterior = self.TE.posterior(self.prior, self.observations, self.s_rng)
        if 0:
            for i, p in enumerate(posterior):
                print ''
                print 'POSTERIOR', i
                print '============'
                theano.printing.debugprint([p.idxs, p.vals])

        assert len(categoricals(posterior.flatten())) == 2
        # one for the shape
        # one for the posterior of the choice variable

    def test_posterior_runs(self):
        posterior = self.TE.posterior(self.prior, self.observations, self.s_rng)
        f = theano.function([self.s_N] + self.observations.flatten(),
                posterior.flatten(),
                allow_input_downcast=True)
        assert len(categoricals(posterior.flatten())) == 2

        if 0:
            # we should be able to optimize out the first categorical
            # because it is used only for shape.  This doesn't currently work.
            theano.printing.debugprint(f)
            assert len(categoricals(f)) == 1

        obs_vals = [0,1,2,3], [0,0,0,1], [3], [.95]
        idxs0, vals0, idxs1, vals1  = f(100, *obs_vals)

        assert numpy.all(idxs0 == range(100))
        assert (vals0 == 0).sum() > 10
        assert (vals0 == 1).sum() > 10
        assert (vals0 == 1).sum()  + (vals0 == 0).sum() == 100

        assert len(idxs1) == (vals0 == 1).sum()
        assert vals1.min() >= .95
