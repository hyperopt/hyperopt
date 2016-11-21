from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
from functools import partial
import os
import unittest

import nose

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from hyperopt import pyll
from hyperopt.pyll import scope

from hyperopt import Trials

from hyperopt.base import miscs_to_idxs_vals, STATUS_OK

from hyperopt import hp

from hyperopt.tpe import adaptive_parzen_normal_orig
from hyperopt.tpe import GMM1
from hyperopt.tpe import GMM1_lpdf
from hyperopt.tpe import LGMM1
from hyperopt.tpe import LGMM1_lpdf

import hyperopt.rand as rand
import hyperopt.tpe as tpe
from hyperopt import fmin

from .test_domains import (
    domain_constructor,
    CasePerDomain)

DO_SHOW = int(os.getenv('HYPEROPT_SHOW', '0'))


def passthrough(x):
    return x


def test_adaptive_parzen_normal_orig():
    rng = np.random.RandomState(123)

    prior_mu = 7
    prior_sigma = 2
    mus = rng.randn(10) + 5

    weights2, mus2, sigmas2 = adaptive_parzen_normal_orig(
        mus, 3.3, prior_mu, prior_sigma)

    print(weights2)
    print(mus2)
    print(sigmas2)

    assert len(weights2) == len(mus2) == len(sigmas2) == 11
    assert np.all(weights2[0] > weights2[1:])
    assert mus2[0] == 7
    assert np.all(mus2[1:] == mus)
    assert sigmas2[0] == 2


class TestGMM1(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(234)

    def test_mu_is_used_correctly(self):
        assert np.allclose(10,
                           GMM1([1], [10.0], [0.0000001], rng=self.rng))

    def test_sigma_is_used_correctly(self):
        samples = GMM1([1], [0.0], [10.0], size=[1000], rng=self.rng)
        assert 9 < np.std(samples) < 11

    def test_mus_make_variance(self):
        samples = GMM1([.5, .5], [0.0, 1.0], [0.000001, 0.000001],
                       rng=self.rng, size=[1000])
        print(samples.shape)
        # import matplotlib.pyplot as plt
        # plt.hist(samples)
        # plt.show()
        assert .45 < np.mean(samples) < .55, np.mean(samples)
        assert .2 < np.var(samples) < .3, np.var(samples)

    def test_weights(self):
        samples = GMM1([.9999, .0001], [0.0, 1.0], [0.000001, 0.000001],
                       rng=self.rng,
                       size=[1000])
        assert samples.shape == (1000,)
        # import matplotlib.pyplot as plt
        # plt.hist(samples)
        # plt.show()
        assert -.001 < np.mean(samples) < .001, np.mean(samples)
        assert np.var(samples) < .0001, np.var(samples)

    def test_mat_output(self):
        samples = GMM1([.9999, .0001], [0.0, 1.0], [0.000001, 0.000001],
                       rng=self.rng,
                       size=[40, 20])
        assert samples.shape == (40, 20)
        assert -.001 < np.mean(samples) < .001, np.mean(samples)
        assert np.var(samples) < .0001, np.var(samples)

    def test_lpdf_scalar_one_component(self):
        llval = GMM1_lpdf(1.0,  # x
                          [1.],           # weights
                          [1.0],          # mu
                          [2.0],          # sigma
                          )
        assert llval.shape == ()
        assert np.allclose(llval,
                           np.log(old_div(1.0, np.sqrt(2 * np.pi * 2.0 ** 2))))

    def test_lpdf_scalar_N_components(self):
        llval = GMM1_lpdf(1.0,     # x
                          [0.25, 0.25, .5],  # weights
                          [0.0, 1.0, 2.0],   # mu
                          [1.0, 2.0, 5.0],   # sigma
                          )
        print(llval)

        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2) *
             np.exp(-.5 * (1.0) ** 2))
        a += (old_div(.25, np.sqrt(2 * np.pi * 2.0 ** 2)))
        a += (.5 / np.sqrt(2 * np.pi * 5.0 ** 2) *
              np.exp(-.5 * (old_div(1.0, 5.0)) ** 2))

    def test_lpdf_vector_N_components(self):
        llval = GMM1_lpdf([1.0, 0.0],     # x
                          [0.25, 0.25, .5],         # weights
                          [0.0, 1.0, 2.0],          # mu
                          [1.0, 2.0, 5.0],          # sigma
                          )

        # case x = 1.0
        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2) *
             np.exp(-.5 * (1.0) ** 2))
        a += (old_div(.25, np.sqrt(2 * np.pi * 2.0 ** 2)))
        a += (.5 / np.sqrt(2 * np.pi * 5.0 ** 2) *
              np.exp(-.5 * (old_div(1.0, 5.0)) ** 2))

        assert llval.shape == (2,)
        assert np.allclose(llval[0], np.log(a))

        # case x = 0.0
        a = (old_div(.25, np.sqrt(2 * np.pi * 1.0 ** 2)))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2) *
              np.exp(-.5 * (old_div(1.0, 2.0)) ** 2))
        a += (.5 / np.sqrt(2 * np.pi * 5.0 ** 2) *
              np.exp(-.5 * (old_div(2.0, 5.0)) ** 2))
        assert np.allclose(llval[1], np.log(a))

    def test_lpdf_matrix_N_components(self):
        llval = GMM1_lpdf(
            [
                [1.0, 0.0, 0.0],
                [0, 0, 1],
                [0, 0, 1000],
            ],
            [0.25, 0.25, .5],  # weights
            [0.0, 1.0, 2.0],   # mu
            [1.0, 2.0, 5.0],   # sigma
        )
        print(llval)
        assert llval.shape == (3, 3)

        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2) *
             np.exp(-.5 * (1.0) ** 2))
        a += (old_div(.25, np.sqrt(2 * np.pi * 2.0 ** 2)))
        a += (.5 / np.sqrt(2 * np.pi * 5.0 ** 2) *
              np.exp(-.5 * (old_div(1.0, 5.0)) ** 2))

        assert np.allclose(llval[0, 0], np.log(a))
        assert np.allclose(llval[1, 2], np.log(a))

        # case x = 0.0
        a = (old_div(.25, np.sqrt(2 * np.pi * 1.0 ** 2)))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2) *
              np.exp(-.5 * (old_div(1.0, 2.0)) ** 2))
        a += (.5 / np.sqrt(2 * np.pi * 5.0 ** 2) *
              np.exp(-.5 * (old_div(2.0, 5.0)) ** 2))

        assert np.allclose(llval[0, 1], np.log(a))
        assert np.allclose(llval[0, 2], np.log(a))
        assert np.allclose(llval[1, 0], np.log(a))
        assert np.allclose(llval[1, 1], np.log(a))
        assert np.allclose(llval[2, 0], np.log(a))
        assert np.allclose(llval[2, 1], np.log(a))

        assert np.isfinite(llval[2, 2])


class TestGMM1Math(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(234)
        self.weights = [.1, .3, .4, .2]
        self.mus = [1.0, 2.0, 3.0, 4.0]
        self.sigmas = [.1, .4, .8, 2.0]
        self.q = None
        self.low = None
        self.high = None
        self.n_samples = 10001
        self.samples_per_bin = 500
        self.show = False
        # -- triggers error if test case forgets to call work()
        self.worked = False

    def tearDown(self):
        assert self.worked

    def work(self):
        self.worked = True
        kwargs = dict(
            weights=self.weights,
            mus=self.mus,
            sigmas=self.sigmas,
            low=self.low,
            high=self.high,
            q=self.q,
        )
        samples = GMM1(rng=self.rng,
                       size=(self.n_samples,),
                       **kwargs)
        samples = np.sort(samples)
        edges = samples[::self.samples_per_bin]
        # print samples

        pdf = np.exp(GMM1_lpdf(edges[:-1], **kwargs))
        dx = edges[1:] - edges[:-1]
        y = 1 / dx / len(dx)

        if self.show:
            plt.scatter(edges[:-1], y)
            plt.plot(edges[:-1], pdf)
            plt.show()
        err = (pdf - y) ** 2
        print(np.max(err))
        print(np.mean(err))
        print(np.median(err))
        if not self.show:
            assert np.max(err) < .1
            assert np.mean(err) < .01
            assert np.median(err) < .01

    def test_basic(self):
        self.work()

    def test_bounded(self):
        self.low = 2.5
        self.high = 3.5
        self.work()


class TestQGMM1Math(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(234)
        self.weights = [.1, .3, .4, .2]
        self.mus = [1.0, 2.0, 3.0, 4.0]
        self.sigmas = [.1, .4, .8, 2.0]
        self.low = None
        self.high = None
        self.n_samples = 1001
        self.show = DO_SHOW  # or put a string
        # -- triggers error if test case forgets to call work()
        self.worked = False

    def tearDown(self):
        assert self.worked

    def work(self, **kwargs):
        self.__dict__.update(kwargs)
        del kwargs
        self.worked = True
        gkwargs = dict(
            weights=self.weights,
            mus=self.mus,
            sigmas=self.sigmas,
            low=self.low,
            high=self.high,
            q=self.q,
        )
        samples = old_div(GMM1(rng=self.rng,
                          size=(self.n_samples,),
                          **gkwargs), self.q)
        print('drew', len(samples), 'samples')
        assert np.all(samples == samples.astype('int'))
        min_max = int(samples.min()), int(samples.max())
        counts = np.bincount(samples.astype('int') - min_max[0])

        print(counts)
        xcoords = np.arange(min_max[0], min_max[1] + 1) * self.q
        prob = np.exp(GMM1_lpdf(xcoords, **gkwargs))
        assert counts.sum() == self.n_samples
        y = old_div(counts, float(self.n_samples))

        if self.show:
            plt.scatter(xcoords, y, c='r', label='empirical')
            plt.scatter(xcoords, prob, c='b', label='predicted')
            plt.legend()
            plt.title(str(self.show))
            plt.show()
        err = (prob - y) ** 2
        print(np.max(err))
        print(np.mean(err))
        print(np.median(err))
        if self.show:
            raise nose.SkipTest()
        else:
            assert np.max(err) < .1
            assert np.mean(err) < .01
            assert np.median(err) < .01

    def test_basic_1(self):
        self.work(q=1)

    def test_basic_2(self):
        self.work(q=2)

    def test_basic_pt5(self):
        self.work(q=0.5)

    def test_bounded_1(self):
        self.work(q=1, low=2, high=4)

    def test_bounded_2(self):
        self.work(q=2, low=2, high=4)

    def test_bounded_1b(self):
        self.work(q=1, low=1, high=4.1)

    def test_bounded_2b(self):
        self.work(q=2, low=1, high=4.1)

    def test_bounded_3(self):
        self.work(
            weights=[0.14285714, 0.28571429, 0.28571429, 0.28571429],
            mus=[5.505, 7., 2., 10.],
            sigmas=[8.99, 5., 8., 8.],
            q=1,
            low=1.01,
            high=10,
            n_samples=10000,
            # show='bounded_3',
        )

    def test_bounded_3b(self):
        self.work(
            weights=[0.33333333, 0.66666667],
            mus=[5.505, 5.],
            sigmas=[8.99, 5.19],
            q=1,
            low=1.01,
            high=10,
            n_samples=10000,
            # show='bounded_3b',
        )


class TestLGMM1Math(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(234)
        self.weights = [.1, .3, .4, .2]
        self.mus = [-2.0, 1.0, 0.0, 3.0]
        self.sigmas = [.1, .4, .8, 2.0]
        self.low = None
        self.high = None
        self.n_samples = 10001
        self.samples_per_bin = 200
        self.show = False
        # -- triggers error if test case forgets to call work()
        self.worked = False

    def tearDown(self):
        assert self.worked

    @property
    def LGMM1_kwargs(self):
        return dict(
            weights=self.weights,
            mus=self.mus,
            sigmas=self.sigmas,
            low=self.low,
            high=self.high,
        )

    def LGMM1_lpdf(self, samples):
        return self.LGMM1(samples, **self.LGMM1_kwargs)

    def work(self, **kwargs):
        self.__dict__.update(kwargs)
        self.worked = True
        samples = LGMM1(rng=self.rng,
                        size=(self.n_samples,),
                        **self.LGMM1_kwargs)
        samples = np.sort(samples)
        edges = samples[::self.samples_per_bin]
        centers = .5 * edges[:-1] + .5 * edges[1:]
        print(edges)

        pdf = np.exp(LGMM1_lpdf(centers, **self.LGMM1_kwargs))
        dx = edges[1:] - edges[:-1]
        y = 1 / dx / len(dx)

        if self.show:
            plt.scatter(centers, y)
            plt.plot(centers, pdf)
            plt.show()
        err = (pdf - y) ** 2
        print(np.max(err))
        print(np.mean(err))
        print(np.median(err))
        if not self.show:
            assert np.max(err) < .1
            assert np.mean(err) < .01
            assert np.median(err) < .01

    def test_basic(self):
        self.work()

    def test_bounded(self):
        self.work(low=2, high=4)


class TestQLGMM1Math(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(234)
        self.weights = [.1, .3, .4, .2]
        self.mus = [-2, 0.0, -3.0, 1.0]
        self.sigmas = [2.1, .4, .8, 2.1]
        self.low = None
        self.high = None
        self.n_samples = 1001
        self.show = DO_SHOW
        # -- triggers error if test case forgets to call work()
        self.worked = False

    def tearDown(self):
        assert self.worked

    @property
    def kwargs(self):
        return dict(
            weights=self.weights,
            mus=self.mus,
            sigmas=self.sigmas,
            low=self.low,
            high=self.high,
            q=self.q)

    def QLGMM1_lpdf(self, samples):
        return self.LGMM1(samples, **self.kwargs)

    def work(self, **kwargs):
        self.__dict__.update(kwargs)
        self.worked = True
        samples = old_div(LGMM1(rng=self.rng,
                                size=(self.n_samples,),
                                **self.kwargs), self.q)
        # -- we've divided the LGMM1 by self.q to get ints here
        assert np.all(samples == samples.astype('int'))
        min_max = int(samples.min()), int(samples.max())
        print('SAMPLES RANGE', min_max)
        counts = np.bincount(samples.astype('int') - min_max[0])

        # print samples
        # print counts
        xcoords = np.arange(min_max[0], min_max[1] + 0.5) * self.q
        prob = np.exp(LGMM1_lpdf(xcoords, **self.kwargs))
        print(xcoords)
        print(prob)
        assert counts.sum() == self.n_samples
        y = old_div(counts, float(self.n_samples))

        if self.show:
            plt.scatter(xcoords, y, c='r', label='empirical')
            plt.scatter(xcoords, prob, c='b', label='predicted')
            plt.legend()
            plt.show()
        # -- calculate errors on the low end, don't take a mean
        #    over all the range spanned by a few outliers.
        err = ((prob - y) ** 2)[:20]
        print(np.max(err))
        print(np.mean(err))
        print(np.median(err))
        if self.show:
            raise nose.SkipTest()
        else:
            assert np.max(err) < .1
            assert np.mean(err) < .01
            assert np.median(err) < .01

    def test_basic_1(self):
        self.work(q=1)

    def test_basic_2(self):
        self.work(q=2)

    def test_basic_pt5(self):
        self.work(q=0.5)

    def test_basic_pt125(self):
        self.work(q=0.125)

    def test_bounded_1(self):
        self.work(q=1, low=2, high=4)

    def test_bounded_2(self):
        self.work(q=2, low=2, high=4)

    def test_bounded_1b(self):
        self.work(q=1, low=1, high=4.1)

    def test_bounded_2b(self):
        self.work(q=2, low=1, high=4.1)


class TestSuggest(unittest.TestCase, CasePerDomain):

    def work(self):
        # -- smoke test that things simply run,
        #    for each type of several search spaces.
        trials = Trials()
        fmin(passthrough,
             space=self.bandit.expr,
             algo=partial(tpe.suggest, n_EI_candidates=3),
             trials=trials,
             max_evals=10)


class TestOpt(unittest.TestCase, CasePerDomain):
    thresholds = dict(
        quadratic1=1e-5,
        q1_lognormal=0.01,
        distractor=-1.96,
        gauss_wave=-2.0,
        gauss_wave2=-2.0,
        n_arms=-2.5,
        many_dists=.0005,
        branin=0.7,
    )

    LEN = dict(
        # -- running a long way out tests overflow/underflow
        #    to some extent
        quadratic1=1000,
        many_dists=200,
        distractor=100,
        # XXX
        q1_lognormal=250,
        gauss_wave2=75,  # -- boosted from 50 on Nov/2013 after new
        #  sampling order made thresh test fail.
        branin=200,
    )

    gammas = dict(
        distractor=.05,
    )

    prior_weights = dict(
        distractor=.01,
    )

    n_EIs = dict(
        # XXX
        # -- this can be low in a few dimensions
        quadratic1=5,
        # -- lower number encourages exploration
        # XXX: this is a damned finicky way to get TPE
        #      to solve the Distractor problem
        distractor=15,
    )

    def setUp(self):
        self.olderr = np.seterr('raise')
        np.seterr(under='ignore')

    def tearDown(self, *args):
        np.seterr(**self.olderr)

    def work(self):

        bandit = self.bandit
        assert bandit.name is not None
        algo = partial(tpe.suggest,
                       gamma=self.gammas.get(bandit.name,
                                             tpe._default_gamma),
                       prior_weight=self.prior_weights.get(bandit.name,
                                                           tpe._default_prior_weight),
                       n_EI_candidates=self.n_EIs.get(bandit.name,
                                                      tpe._default_n_EI_candidates),
                       )
        LEN = self.LEN.get(bandit.name, 50)

        trials = Trials()
        fmin(passthrough,
             space=bandit.expr,
             algo=algo,
             trials=trials,
             max_evals=LEN,
             rstate=np.random.RandomState(123),
             catch_eval_exceptions=False)
        assert len(trials) == LEN

        if 1:
            rtrials = Trials()
            fmin(passthrough,
                 space=bandit.expr,
                 algo=rand.suggest,
                 trials=rtrials,
                 max_evals=LEN)
            print('RANDOM MINS', list(sorted(rtrials.losses()))[:6])
            # logx = np.log([s['x'] for s in rtrials.specs])
            # print 'RND MEAN', np.mean(logx)
            # print 'RND STD ', np.std(logx)

        if 0:
            plt.subplot(2, 2, 1)
            plt.scatter(list(range(LEN)), trials.losses())
            plt.title('TPE losses')
            plt.subplot(2, 2, 2)
            plt.scatter(list(range(LEN)), ([s['x'] for s in trials.specs]))
            plt.title('TPE x')
            plt.subplot(2, 2, 3)
            plt.title('RND losses')
            plt.scatter(list(range(LEN)), rtrials.losses())
            plt.subplot(2, 2, 4)
            plt.title('RND x')
            plt.scatter(list(range(LEN)), ([s['x'] for s in rtrials.specs]))
            plt.show()
        if 0:
            plt.hist(
                [t['x'] for t in self.experiment.trials],
                bins=20)

        # print trials.losses()
        print('TPE    MINS', list(sorted(trials.losses()))[:6])
        # logx = np.log([s['x'] for s in trials.specs])
        # print 'TPE MEAN', np.mean(logx)
        # print 'TPE STD ', np.std(logx)
        thresh = self.thresholds[bandit.name]
        print('Thresh', thresh)
        assert min(trials.losses()) < thresh


@domain_constructor(loss_target=0)
def opt_q_uniform(target):
    rng = np.random.RandomState(123)
    x = hp.quniform('x', 1.01, 10, 1)
    return {'loss': (x - target) ** 2 + scope.normal(0, 1, rng=rng),
            'status': STATUS_OK}


class TestOptQUniform(object):

    show_steps = False
    show_vars = DO_SHOW
    LEN = 25

    def work(self, **kwargs):
        self.__dict__.update(kwargs)
        bandit = opt_q_uniform(self.target)
        prior_weight = 2.5
        gamma = 0.20
        algo = partial(tpe.suggest,
                       prior_weight=prior_weight,
                       n_startup_jobs=2,
                       n_EI_candidates=128,
                       gamma=gamma)
        # print algo.opt_idxs['x']
        # print algo.opt_vals['x']

        trials = Trials()
        fmin(passthrough,
             space=bandit.expr,
             algo=algo,
             trials=trials,
             max_evals=self.LEN)
        if self.show_vars:
            import hyperopt.plotting
            hyperopt.plotting.main_plot_vars(trials, bandit, do_show=1)

        idxs, vals = miscs_to_idxs_vals(trials.miscs)
        idxs = idxs['x']
        vals = vals['x']

        losses = trials.losses()

        from hyperopt.tpe import ap_filter_trials
        from hyperopt.tpe import adaptive_parzen_samplers

        qu = scope.quniform(1.01, 10, 1)
        fn = adaptive_parzen_samplers['quniform']
        fn_kwargs = dict(size=(4,), rng=np.random)
        s_below = pyll.Literal()
        s_above = pyll.Literal()
        b_args = [s_below, prior_weight] + qu.pos_args
        b_post = fn(*b_args, **fn_kwargs)
        a_args = [s_above, prior_weight] + qu.pos_args
        a_post = fn(*a_args, **fn_kwargs)

        # print b_post
        # print a_post
        fn_lpdf = getattr(scope, a_post.name + '_lpdf')
        print(fn_lpdf)
        # calculate the llik of b_post under both distributions
        a_kwargs = dict([(n, a) for n, a in a_post.named_args
                         if n not in ('rng', 'size')])
        b_kwargs = dict([(n, a) for n, a in b_post.named_args
                         if n not in ('rng', 'size')])
        below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
        above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)
        new_node = scope.broadcast_best(b_post, below_llik, above_llik)

        print('=' * 80)

        do_show = self.show_steps

        for ii in range(2, 9):
            if ii > len(idxs):
                break
            print('-' * 80)
            print('ROUND', ii)
            print('-' * 80)
            all_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            below, above = ap_filter_trials(idxs[:ii],
                                            vals[:ii], idxs[:ii], losses[:ii], gamma)
            below = below.astype('int')
            above = above.astype('int')
            print('BB0', below)
            print('BB1', above)
            # print 'BELOW',  zip(range(100), np.bincount(below, minlength=11))
            # print 'ABOVE',  zip(range(100), np.bincount(above, minlength=11))
            memo = {b_post: all_vals, s_below: below, s_above: above}
            bl, al, nv = pyll.rec_eval([below_llik, above_llik, new_node],
                                       memo=memo)
            # print bl - al
            print('BB2', dict(list(zip(all_vals, bl - al))))
            print('BB3', dict(list(zip(all_vals, bl))))
            print('BB4', dict(list(zip(all_vals, al))))
            print('ORIG PICKED', vals[ii])
            print('PROPER OPT PICKS:', nv)

            # assert np.allclose(below, [3, 3, 9])
            # assert len(below) + len(above) == len(vals)

            if do_show:
                plt.subplot(8, 1, ii)
                # plt.scatter(all_vals,
                #    np.bincount(below, minlength=11)[2:], c='b')
                # plt.scatter(all_vals,
                #    np.bincount(above, minlength=11)[2:], c='c')
                plt.scatter(all_vals, bl, c='g')
                plt.scatter(all_vals, al, c='r')
        if do_show:
            plt.show()

    def test4(self):
        self.work(target=4, LEN=100)

    def test2(self):
        self.work(target=2, LEN=100)

    def test6(self):
        self.work(target=6, LEN=100)

    def test10(self):
        self.work(target=10, LEN=100)
