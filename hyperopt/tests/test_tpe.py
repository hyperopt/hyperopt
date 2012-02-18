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


    def test_lpdf_matches_hist(self):
        weights = [.1, .3, .4, .2]
        mus = [1.0, 2.0, 3.0, 4.0]
        sigmas = [.1, .4, .8, 2.0]

        n_samples = 1000

        samples = GMM1(weights, mus, sigmas)

        hist, edges = np.histogram(samples)
        pdf = np.exp(GMM1_lpdf(edges, weights, mus, sigmas))

        import matplotlib.pyplot as plt
        plt.scatter(edges[:-1], hist / hist.sum())
        plt.plot(edges, pdf)
        plot.show()



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



if 0:
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
                GM_BanditAlgo(self.bandit,
                        good_estimator=IndependentNullEstimator(),
                        bad_estimator=IndependentNullEstimator()))

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

            assert np.all(idxs0 == range(100))
            assert (vals0 == 0).sum() > 10
            assert (vals0 == 1).sum() > 10
            assert (vals0 == 1).sum()  + (vals0 == 0).sum() == 100

            assert len(idxs1) == (vals0 == 1).sum()
            assert vals1.min() >= .95

    class TestGM_Distractor(unittest.TestCase): # Tests normal
        def setUp(self):
            raise nose.SkipTest()
            self.experiment = SerialExperiment(
                bandit_algo=GM_BanditAlgo(
                        bandit=hyperopt.bandits.Distractor(),
                        good_estimator=IndependentAdaptiveParzenEstimator(),
                        bad_estimator=IndependentAdaptiveParzenEstimator()))

        def test_op_counts(self):
            # If everything is done right, there should be
            # 2 adaptive parzen estimators in the algorithm
            #  - one for fitting the good examples
            #  - one for fitting the rest of the examples
            # 1 GMM1 Op for drawing from the fit of good examples

            def gmms(fn):
                return [ap for ap in fn.maker.env.toposort()
                    if isinstance(ap.op, montetheano.distributions.GMM1)]

            def adaptive_parzens(fn):
                return [ap for ap in fn.maker.env.toposort()
                    if isinstance(ap.op, idxs_vals_rnd.AdaptiveParzen)]

            # touch property to compile fn
            self.experiment.bandit_algo._suggest_from_model_fn
            HL = self.experiment.bandit_algo.helper_locals
            if 1:
                f = theano.function(
                    [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                        + HL['s_obs'].flatten(),
                    HL['G_ll'],
                    allow_input_downcast=True,
                    )
                # theano.printing.debugprint(f)
                assert len(gmms(f)) == 1
                assert len(adaptive_parzens(f)) == 1

            if 1:
                f = theano.function(
                    [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                        + HL['s_obs'].flatten(),
                    HL['G_ll'] - HL['B_ll'],
                    allow_input_downcast=True,
                    )
                #print gmms(f)
                #print adaptive_parzens(f)
                assert len(gmms(f)) == 1
                assert len(adaptive_parzens(f)) == 2

            helper = self.experiment.bandit_algo._suggest_from_model_fn
            assert len(gmms(helper)) == 1
            assert len(adaptive_parzens(helper)) == 2


        def test_optimize_20(self):
            self.experiment.run(50)

            plt.subplot(1,2,1)
            plt.plot(self.experiment.losses())
            plt.subplot(1,2,2)
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)

            print self.experiment.losses()
            print 'MIN', min(self.experiment.losses())
            assert min(self.experiment.losses()) < -1.85

            if 0:
                plt.show()


    class TestGM_Q1Lognormal(unittest.TestCase): # Tests lognormal
        def setUp(self):
            raise nose.SkipTest()
            self.experiment = SerialExperiment(
                bandit_algo=GM_BanditAlgo(
                        bandit=hyperopt.bandits.Q1Lognormal(),
                        good_estimator=IndependentAdaptiveParzenEstimator(),
                        bad_estimator=IndependentAdaptiveParzenEstimator()))

        def test_optimize_20(self):
            self.experiment.run(50)

            plt.subplot(1,2,1)
            plt.plot(self.experiment.losses())
            plt.subplot(1,2,2)
            if 0:
                plt.hist(
                        [t['x'] for t in self.experiment.trials],
                        bins=20)
            else:
                plt.scatter(
                        [t['x'] for t in self.experiment.trials],
                        range(len(self.experiment.trials)))
            print self.experiment.losses()
            print 'MIN', min(self.experiment.losses())
            assert min(self.experiment.losses()) < .01
            if 0:
                plt.show()


    class TestGaussWave2(unittest.TestCase): # Tests nested search
        def setUp(self):
            raise nose.SkipTest()
            self.experiment = SerialExperiment(
                bandit_algo=GM_BanditAlgo(
                        bandit=hyperopt.bandits.GaussWave2(),
                        good_estimator=IndependentAdaptiveParzenEstimator(),
                        bad_estimator=IndependentAdaptiveParzenEstimator()))

        def test_op_counts_in_llik(self):
            HL = self.experiment.bandit_algo.helper_locals
            f = theano.function(
                    [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                        + HL['s_obs'].flatten(),
                    HL['log_EI'],
                    no_default_updates=True,
                    mode='FAST_RUN')                 # required for shape inference
            try:
                assert len(gmms(f)) == 0
                assert len(bgmms(f)) == 2            # sampling from good
                assert len(categoricals(f)) == 1     # sampling from good
                assert len(adaptive_parzens(f)) == 4 # fitting both good and bad
            except:
                theano.printing.debugprint(f)
                raise

        def test_op_counts_in_Gsamples(self):
            HL = self.experiment.bandit_algo.helper_locals
            f = theano.function(
                    [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                        + HL['s_obs'].flatten(),
                    HL['Gsamples'].flatten(),
                    no_default_updates=True,         # allow prune priors
                    mode='FAST_RUN')                 # required for shape inference
            try:
                assert len(gmms(f)) == 0
                assert len(bgmms(f)) == 2            # sampling from good
                assert len(categoricals(f)) == 1     # sampling from good
                assert len(adaptive_parzens(f)) == 2 # fitting both good and bad
            except:
                theano.printing.debugprint(f)
                raise


        def test_optimize_20(self):
            self.experiment.run(50)

            plt.subplot(1,2,1)
            plt.plot(self.experiment.losses())
            plt.subplot(1,2,2)
            plt.scatter(
                    [t['x'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
            print self.experiment.losses()
            print 'MIN', min(self.experiment.losses())
            assert min(self.experiment.losses()) < -1.75
            if 0:
                plt.show()

        def test_fit(self):
            self.experiment.run(150)
            plt.plot(
                    range(len(self.experiment.losses())),
                    self.experiment.losses())
            plt.figure()
            hyperopt.plotting.main_plot_vars(self.experiment,
                    end_with_show=True)


    class TestGM_DummyDBN(unittest.TestCase):
        def setUp(self):
            raise nose.SkipTest()
            self.experiment = SerialExperiment(
                bandit_algo=GM_BanditAlgo(
                        bandit=Dummy_DBN_Base(),
                        good_estimator=IndependentAdaptiveParzenEstimator(),
                        bad_estimator=IndependentAdaptiveParzenEstimator()))
            self._old = theano.gof.link.raise_with_op.print_thunk_trace
            theano.gof.link.raise_with_op.print_thunk_trace = True

        def tearDown(self):
            theano.gof.link.raise_with_op.print_thunk_trace = self._old

        def test_optimize_20(self):
            def callback(node, thunk, storage_map, compute_map):
                numeric_outputs = [storage_map[v][0]
                        for v in node.outputs
                        if isinstance(v.type, theano.tensor.TensorType)]
                numeric_inputs = [storage_map[v][0]
                        for v in node.inputs
                        if isinstance(v.type, theano.tensor.TensorType)]

                if not all([np.all(np.isfinite(n)) for n in numeric_outputs]):
                    theano.printing.debugprint(node, depth=8)
                    print 'inputs'
                    print numeric_inputs
                    print 'outputs'
                    print numeric_outputs
                    raise ValueError('non-finite created in', node)

            mode = theano.Mode(
                    optimizer='fast_compile',
                    linker=theano.gof.vm.VM_Linker(callback=callback))
            self.experiment.bandit_algo.mode = mode
            _helper = self.experiment.bandit_algo._suggest_from_model_fn
            theano.printing.debugprint(_helper)
            for i in range(50):
                print 'ITER', i
                try:
                    self.experiment.run(1)
                except:

                    raise

            if 0:
                plt.subplot(1,2,1)
                plt.plot(self.experiment.losses())
                plt.subplot(1,2,2)
                plt.scatter(
                        [t['x'] for t in self.experiment.trials],
                        range(len(self.experiment.trials)))
                plt.show()

    class TestGM_Quadratic1(unittest.TestCase): # Tests uniform
        def setUp(self):
            self.experiment = SerialExperiment(
                bandit_algo=GM_BanditAlgo(
                        bandit=hyperopt.bandits.Quadratic1(),
                        good_estimator=IndependentAdaptiveParzenEstimator(),
                        bad_estimator=IndependentAdaptiveParzenEstimator()))

        def test_op_counts(self):
            # If everything is done right, there should be
            # 2 adaptive parzen estimators in the algorithm
            #  - one for fitting the good examples
            #  - one for fitting the rest of the examples
            # 1 GMM1 Op for drawing from the fit of good examples

            def gmms(fn):
                return [ap for ap in fn.maker.env.toposort()
                    if isinstance(ap.op, montetheano.distributions.BGMM1)]

            def adaptive_parzens(fn):
                return [ap for ap in fn.maker.env.toposort()
                    if isinstance(ap.op, idxs_vals_rnd.AdaptiveParzen)]

            # touch property to compile fn
            self.experiment.bandit_algo._suggest_from_model_fn
            HL = self.experiment.bandit_algo.helper_locals
            if 1:
                f = theano.function(
                    [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                        + HL['s_obs'].flatten(),
                    HL['G_ll'],
                    allow_input_downcast=True,
                    )
                # theano.printing.debugprint(f)
                assert len(gmms(f)) == 1
                assert len(adaptive_parzens(f)) == 1

            if 1:
                f = theano.function(
                    [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                        + HL['s_obs'].flatten(),
                    HL['G_ll'] - HL['B_ll'],
                    allow_input_downcast=True,
                    )
                #print gmms(f)
                #print adaptive_parzens(f)
                assert len(gmms(f)) == 1
                assert len(adaptive_parzens(f)) == 2

            # touch property to compile fn
            _helper = self.experiment.bandit_algo._suggest_from_model_fn
            assert len(gmms(_helper)) == 1
            assert len(adaptive_parzens(_helper)) == 2


        def test_optimize_20(self):
            self.experiment.run(50)

            plt.subplot(1,2,1)
            plt.plot(self.experiment.losses())
            plt.subplot(1,2,2)
            if 0:
                plt.hist(
                        [t['x'] for t in self.experiment.trials],
                        bins=20)
            else:
                plt.scatter(
                        [t['x'] for t in self.experiment.trials],
                        range(len(self.experiment.trials)))
            print self.experiment.losses()
            print 'MIN', min(self.experiment.losses())
            assert min(self.experiment.losses()) < 0.01

            if 0:
                plt.show()



    class TestGM_TwoArms(unittest.TestCase): # Tests one_of
        def setUp(self):
            self.experiment = hyperopt.Experiment(
                bandit_algo=GM_BanditAlgo(
                        bandit=hyperopt.bandits.TwoArms(),
                        good_estimator=IndependentAdaptiveParzenEstimator(),
                        bad_estimator=IndependentAdaptiveParzenEstimator()))

        def test_optimize_20(self):
            self.experiment.bandit_algo.build_helpers()
            HL = self.experiment.bandit_algo.helper_locals
            assert len(HL['Gsamples']) == 1
            Gpseudocounts = HL['Gsamples'][0].vals.owner.inputs[1]
            Bpseudocounts = HL['Bsamples'][0].vals.owner.inputs[1]

            f = self.experiment.bandit_algo._suggest_from_model_fn
            debug = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                (HL['Gobs'].flatten()
                    + [Gpseudocounts]
                    + [Bpseudocounts]
                    + [HL['yvals'][where(HL['yvals'] < HL['y_thresh'])]]
                    + [HL['yvals'][where(HL['yvals'] >= HL['y_thresh'])]]
                    ),
                allow_input_downcast=True,
                )
            debug_rval = [None]
            def _helper(*args):
                rval = f(*args)
                debug_rval[0] = debug(*args)
                return rval
            self.experiment.bandit_algo._helper = _helper
            self.experiment.run(200)

            gobs_idxs, gobs_vals, Gpseudo, Bpseudo, Gyvals, Byvals = debug_rval[0]
            print gobs_idxs
            print 'Gpseudo', Gpseudo
            print 'Bpseudo', Bpseudo

            import matplotlib.pyplot as plt
            plt.subplot(1,4,1)
            Xs = [t['x'] for t in self.experiment.trials]
            Ys = self.experiment.losses()
            plt.plot(Ys)
            plt.xlabel('time')
            plt.ylabel('loss')

            plt.subplot(1,4,2)
            plt.scatter(Xs,Ys )
            plt.xlabel('X')
            plt.ylabel('loss')

            plt.subplot(1,4,3)
            plt.hist(Xs )
            plt.xlabel('X')
            plt.ylabel('freq')

            plt.subplot(1,4,4)
            plt.hist(Gyvals, bins=20)
            plt.hist(Byvals, bins=20)

            print self.experiment.losses()
            print 'MIN', min(self.experiment.losses())
            assert min(self.experiment.losses()) < -3.00

            if 0:
                plt.show()



