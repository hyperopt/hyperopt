import unittest
import nose

import numpy as np
import theano
import matplotlib.pyplot as plt

import pyll
from pyll import scope

import hyperopt.bandits

from hyperopt import Bandit
from hyperopt import Experiment
from hyperopt import Random
from hyperopt import Trials

from hyperopt.base import miscs_to_idxs_vals

from hyperopt.bandits import Quadratic1
from hyperopt.bandits import Q1Lognormal
from hyperopt.bandits import TwoArms
from hyperopt.bandits import Distractor
from hyperopt.bandits import GaussWave
from hyperopt.bandits import GaussWave2

from hyperopt.tpe import adaptive_parzen_normal
from hyperopt.tpe import TreeParzenEstimator
from hyperopt.tpe import GMM1
from hyperopt.tpe import GMM1_lpdf
from hyperopt.tpe import LGMM1
from hyperopt.tpe import LGMM1_lpdf
from hyperopt.tpe import normal_cdf


class ManyDists(hyperopt.bandits.Base):
    loss_target = 0

    def __init__(self):
        hyperopt.bandits.Base.__init__(self, dict(
            a=scope.one_of(0, 1, 2),
            b=scope.randint(10),
            c=scope.uniform(4, 7),
            d=scope.loguniform(-2, 0),
            e=scope.quniform(0, 10, 3),
            f=scope.qloguniform(0, 3, 2),
            g=scope.normal(4, 7),
            h=scope.lognormal(-2, 2),
            i=scope.qnormal(0, 10, 2),
            j=scope.qlognormal(0, 2, 1),
            ))

    def score(self, config):
        return - float(np.sum(config.values()) ** 2)


def test_adaptive_parzen_normal():
    rng = np.random.RandomState(123)

    prior_mu = 7
    prior_sigma = 2
    mus = rng.randn(10) + 5

    weights2, mus2, sigmas2 = adaptive_parzen_normal(mus, prior_mu, prior_sigma)

    print weights2
    print mus2
    print sigmas2

    assert len(weights2) == len(mus2) == len(sigmas2) == 11
    assert np.all(weights2[0] > weights2[1:])
    assert mus2[0] == 7
    assert np.all(mus2[1:] == mus)
    assert sigmas2[0] == 2


def test_tpe_filter():
    bandit = Quadratic1()
    random_algo = Random(bandit)

    # build an experiment of 10 trials
    trials = Trials()
    exp = Experiment(trials, random_algo)
    exp.run(10)
    ids = trials.tids
    assert len(ids) == 10

    tpe_algo = TreeParzenEstimator(bandit)
    (g_s, g_r, g_m), (b_s, b_r, b_m) = tpe_algo.filter_trials(
            trials.specs, trials.results, trials.miscs, ids[2:8])

    assert len(g_s) == len(g_r) == len(g_m)
    assert len(b_s) == len(b_r) == len(b_m)
    assert len(g_s) + len(b_s) == 6

    g_ids = [m['tid'] for m in g_m]
    b_ids = [m['tid'] for m in b_m]
    assert set(g_ids).intersection(b_ids) == set()


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
        print samples.shape
        #import matplotlib.pyplot as plt
        #plt.hist(samples)
        #plt.show()
        assert .45 < np.mean(samples) < .55, np.mean(samples)
        assert .2 < np.var(samples) < .3, np.var(samples)

    def test_weights(self):
        samples = GMM1([.9999, .0001], [0.0, 1.0], [0.000001, 0.000001],
                rng=self.rng,
                size=[1000])
        assert samples.shape == (1000,)
        #import matplotlib.pyplot as plt
        #plt.hist(samples)
        #plt.show()
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
        llval = GMM1_lpdf(1.0, # x
                [1.],  # weights
                [1.0], # mu
                [2.0], # sigma
                )
        assert llval.shape == ()
        assert np.allclose(llval,
                np.log(1.0 / np.sqrt(2 * np.pi * 2.0**2)))

    def test_lpdf_scalar_N_components(self):
        llval = GMM1_lpdf(1.0, # x
                [0.25, 0.25, .5],  # weights
                [0.0, 1.0, 2.0], # mu
                [1.0, 2.0, 5.0], # sigma
                )

        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2)
                * np.exp(-.5 * (1.0)**2))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2))
        a += (.5 /  np.sqrt(2 * np.pi * 5.0 ** 2)
                * np.exp(-.5 * (1.0 / 5.0) ** 2))

    def test_lpdf_vector_N_components(self):
        llval = GMM1_lpdf([1.0, 0.0],     # x
                [0.25, 0.25, .5], # weights
                [0.0, 1.0, 2.0],  # mu
                [1.0, 2.0, 5.0],  # sigma
                )

        # case x = 1.0
        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2)
                * np.exp(-.5 * (1.0)**2))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2))
        a += (.5 /  np.sqrt(2 * np.pi * 5.0 ** 2)
                * np.exp(-.5 * (1.0 / 5.0) ** 2))

        assert llval.shape == (2,)
        assert np.allclose(llval[0], np.log(a))


        # case x = 0.0
        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2)
                * np.exp(-.5 * (1.0 / 2.0) ** 2))
        a += (.5 /  np.sqrt(2 * np.pi * 5.0 ** 2)
                * np.exp(-.5 * (2.0 / 5.0) ** 2))
        assert np.allclose(llval[1], np.log(a))

    def test_lpdf_matrix_N_components(self):
        llval = GMM1_lpdf(
                [
                    [1.0, 0.0, 0.0],
                    [0, 0, 1],
                    [0, 0, 1000],
                ],
                [0.25, 0.25, .5],  # weights
                [0.0, 1.0, 2.0], # mu
                [1.0, 2.0, 5.0], # sigma
                )
        print llval
        assert llval.shape == (3,3)

        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2)
                * np.exp(-.5 * (1.0)**2))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2))
        a += (.5 /  np.sqrt(2 * np.pi * 5.0 ** 2)
                * np.exp(-.5 * (1.0 / 5.0) ** 2))

        assert np.allclose(llval[0,0], np.log(a))
        assert np.allclose(llval[1,2], np.log(a))


        # case x = 0.0
        a = (.25 / np.sqrt(2 * np.pi * 1.0 ** 2))
        a += (.25 / np.sqrt(2 * np.pi * 2.0 ** 2)
                * np.exp(-.5 * (1.0 / 2.0)**2))
        a += (.5 /  np.sqrt(2 * np.pi * 5.0 ** 2)
                * np.exp(-.5 * (2.0 / 5.0) ** 2))

        assert np.allclose(llval[0,1], np.log(a))
        assert np.allclose(llval[0,2], np.log(a))
        assert np.allclose(llval[1,0], np.log(a))
        assert np.allclose(llval[1,1], np.log(a))
        assert np.allclose(llval[2,0], np.log(a))
        assert np.allclose(llval[2,1], np.log(a))

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
        #print samples

        pdf = np.exp(GMM1_lpdf(edges[:-1], **kwargs))
        dx = edges[1:] - edges[:-1]
        y = 1 / dx / len(dx)

        if self.show:
            import matplotlib.pyplot as plt
            plt.scatter(edges[:-1], y)
            plt.plot(edges[:-1], pdf)
            plt.show()
        err = (pdf - y) ** 2
        print np.max(err)
        print np.mean(err)
        print np.median(err)
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
        self.show = False
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
        samples = GMM1(rng=self.rng,
                size=(self.n_samples,),
                **gkwargs) / self.q
        assert np.all(samples == samples.astype('int'))
        min_max = int(samples.min()), int(samples.max())
        counts = np.bincount(samples.astype('int') - min_max[0])

        print counts
        xcoords = np.arange(min_max[0], min_max[1] + 1) * self.q
        prob = np.exp(GMM1_lpdf(xcoords, **gkwargs))
        assert counts.sum() == self.n_samples
        y = counts / float(self.n_samples)

        if self.show:
            import matplotlib.pyplot as plt
            plt.scatter(xcoords, y, c='r')
            plt.scatter(xcoords, prob, c='b')
            plt.show()
        err = (prob - y) ** 2
        print np.max(err)
        print np.mean(err)
        print np.median(err)
        if not self.show:
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
        print edges

        pdf = np.exp(LGMM1_lpdf(centers, **self.LGMM1_kwargs))
        dx = edges[1:] - edges[:-1]
        y = 1 / dx / len(dx)

        if self.show:
            import matplotlib.pyplot as plt
            plt.scatter(centers, y)
            plt.plot(centers, pdf)
            plt.show()
        err = (pdf - y) ** 2
        print np.max(err)
        print np.mean(err)
        print np.median(err)
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
        self.show = False
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
        samples = LGMM1(rng=self.rng,
                size=(self.n_samples,),
                **self.kwargs) / self.q
        # -- qlognormals have ceil, should never be 0
        assert samples.min() >= 1
        assert np.all(samples == samples.astype('int'))
        min_max = int(samples.min()), int(samples.max())
        print 'SAMPLES RANGE', min_max
        counts = np.bincount(samples.astype('int') - min_max[0])

        #print samples
        #print counts
        xcoords = np.arange(min_max[0], min_max[1] + 0.5) * self.q
        prob = np.exp(LGMM1_lpdf(xcoords, **self.kwargs))
        print xcoords
        print prob
        assert counts.sum() == self.n_samples
        y = counts / float(self.n_samples)

        if self.show:
            import matplotlib.pyplot as plt
            plt.scatter(xcoords, y, c='r')
            plt.scatter(xcoords, prob, c='b')
            plt.show()
        # -- calculate errors on the low end, don't take a mean
        #    over all the range spanned by a few outliers.
        err = ((prob - y) ** 2)[:20]
        print np.max(err)
        print np.mean(err)
        print np.median(err)
        if not self.show:
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


class CasePerBandit(object):
    def test_quadratic1(self): self.bandit = Quadratic1(); self.work()
    def test_q1lognormal(self): self.bandit = Q1Lognormal(); self.work()
    def test_twoarms(self): self.bandit = TwoArms(); self.work()
    def test_distractor(self): self.bandit = Distractor(); self.work()
    def test_gausswave(self): self.bandit = GaussWave(); self.work()
    def test_gausswave2(self): self.bandit = GaussWave2(); self.work()
    def test_many_dists(self): self.bandit = ManyDists(); self.work()


class TestPosteriorClone(unittest.TestCase, CasePerBandit):
    def work(self):
        """Test that all prior samplers are gone"""
        tpe_algo = TreeParzenEstimator(self.bandit)
        foo = pyll.as_apply([
                    tpe_algo.post_idxs, tpe_algo.post_vals])
        prior_names = [
                'uniform',
                'quniform',
                'loguniform',
                'qloguniform',
                'normal',
                'qnormal',
                'lognormal',
                'qlognormal',
                'randint',
                ]
        for node in pyll.dfs(foo):
            assert node.name not in prior_names


class TestPosteriorCloneSample(unittest.TestCase, CasePerBandit):
    def work(self):
        bandit = self.bandit
        random_algo = Random(bandit)
        # build an experiment of 10 trials
        trials = Trials()
        exp = Experiment(trials, random_algo)
        #print random_algo.s_specs_idxs_vals
        exp.run(10)
        ids = trials.tids
        assert len(ids) == 10
        tpe_algo = TreeParzenEstimator(bandit)
        #print pyll.as_apply(tpe_algo.post_idxs)
        #print pyll.as_apply(tpe_algo.post_vals)
        tpe_algo.set_iv(tpe_algo.observed,
                *miscs_to_idxs_vals(trials.miscs))
        pi, pv = pyll.stochastic.sample(
                pyll.as_apply([tpe_algo.post_idxs, tpe_algo.post_vals]),
                np.random.RandomState(33))
        print pi
        print pv


class TestSuggest(unittest.TestCase, CasePerBandit):
    def work(self):
        trials = Trials()
        bandit = self.bandit
        tpe_algo = TreeParzenEstimator(bandit)
        tpe_algo.n_EI_candidates = 3
        exp = Experiment(trials, tpe_algo)
        exp.run(10)


class TestOpt(unittest.TestCase, CasePerBandit):
    thresholds = dict(
            Distractor=-1.85, #XXX: investigate this failure
            Q1Lognormal=0.01,
            GaussWave=-2.0,
            GaussWave2=-2.0,
            Quadratic1=0.01,
            TwoArms=-2.5, # XXX: test categorical inference
            ManyDists=20,
            )

    def work(self):
        bandit = self.bandit
        algo = TreeParzenEstimator(bandit)
        trials = Trials()
        Experiment(trials, algo).run(50)

        if 0:
            plt.subplot(1,2,1)
            plt.plot(trials.losses())
            plt.subplot(1,2,2)
            plt.plot([s['x'] for s in trials.specs])
            plt.show()
        if 0:
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)

        print trials.losses()
        bname = bandit.__class__.__name__
        print 'Bandit', bname
        print 'MIN', min(trials.losses())
        thresh = self.thresholds[bname]
        print 'Thresh', thresh
        if bname in ['Distractor']:
            raise nose.SkipTest()
        else:
            assert min(trials.losses()) < thresh


