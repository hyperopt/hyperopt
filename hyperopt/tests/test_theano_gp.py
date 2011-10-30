"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra, Dan Yamins"
__copyright__ = "(c) 2011, James Bergstra, Dan Yamins"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import unittest

import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import theano
from theano import tensor
from theano.tests.unittest_tools import verify_grad, seed_rng

from hyperopt.idxs_vals_rnd import IdxsValsList
from hyperopt.bandits import TwoArms
from hyperopt.base import Bandit, BanditAlgo
from hyperopt.theano_gp import GP_BanditAlgo
from hyperopt.ht_dist2 import rSON2, normal
from hyperopt.genson_bandits import GensonBandit
from hyperopt.experiments import SerialExperiment
from hyperopt.dbn import Dummy_DBN_Base
import hyperopt.plotting


GPAlgo = GP_BanditAlgo


def show_bandit_algo(self, trials, results, xlim_low=-5, xlim_high=5,
        n_candidates=5):
    ivls = self.idxs_vals_by_status(trials, results)
    X_IVLs = ivls['x_IVLs']
    Ys = ivls['losses']
    Ys_var = ivls['losses_variance']
    prepared_data = self.prepare_GP_training_data(ivls)
    x_all, y_all, y_mean, y_var, y_std = prepared_data
    self.fit_GP(*prepared_data)

    candidates = self._prior_sampler(n_candidates)
    EI = self.GP_EI(IdxsValsList.fromflattened(candidates))
    print ''
    print 'Candidates'
    print 'idxs', candidates[0]
    print 'vals', candidates[1]
    print 'EI', EI
    #print 'optimizing candidates'
    candidates_opt = self.GP_EI_optimize(
            IdxsValsList.fromflattened(candidates))
    self.post_refinement(candidates_opt)
    EI_opt = self.GP_EI(candidates_opt)
    print ''
    print 'Optimized candidates'
    print 'idxs', candidates_opt[0].idxs
    print 'vals', candidates_opt[0].vals
    print "EI", EI_opt

    num = len(candidates_opt)

    plt.scatter(x_all[0].vals,
            y_all * self._GP_y_std + self._GP_y_mean)
    plt.scatter(candidates[1], numpy.zeros_like(candidates[1]),
        c='y')
    plt.scatter(candidates_opt[0].vals,
            numpy.zeros_like(candidates[1]) - .1,
            c='k')

    plt.figure()

    plt.xlim([xlim_low, xlim_high])
    xmesh = numpy.linspace(xlim_low, xlim_high)
    N = len(xmesh)
    XmeshN = [numpy.arange(N) for _ind in range(num)]
    Xmesh = [numpy.linspace(xlim_low, xlim_high)
            for _ind in range(num)]

    print Xmesh

    IVL = IdxsValsList.fromlists(XmeshN, Xmesh)
    gp_mean, gp_var = self.GP_mean_variance(IVL)
    gp_EI = self.GP_EI(IVL)

    print "GP_VAR", gp_var
    plt.plot(xmesh, gp_mean)
    plt.plot(xmesh, gp_mean + numpy.sqrt(gp_var), c='g')
    plt.plot(xmesh, gp_mean - numpy.sqrt(gp_var), c='g')
    plt.plot(xmesh, gp_EI, c='r')
    plt.show()


class TestGaussian1D(unittest.TestCase):
    def setUp(self):
        class GaussianBandit(GensonBandit):
            test_str = '{"x":gaussian(0,1)}'

            def __init__(self):
                super(GaussianBandit, self).__init__(
                        source_string=self.test_str)

            @classmethod
            def evaluate(cls, config, ctrl):
                return dict(
                        loss=(config['x'] - 2) ** 2,
                        status='ok')

            @classmethod
            def loss_variance(cls, result, config):
                return .1
        self.bandit = GaussianBandit()
        self.algo = GPAlgo(self.bandit)

    def test_basic(self):
        self.algo.n_startup_jobs = 7
        n_iter = 40
        serial_exp = SerialExperiment(self.algo)
        serial_exp.run(self.algo.n_startup_jobs)
        serial_exp.run(n_iter)
        assert min(serial_exp.losses()) < 1e-2


class TestUniform1D(unittest.TestCase):
    def setUp(self):
        class UniformBandit(GensonBandit):
            test_str = '{"x":uniform(-3,2)}'

            def __init__(self):
                super(UniformBandit, self).__init__(
                        source_string=self.test_str)

            def evaluate(cls, config, ctrl):
                return dict(loss=(config['x'] + 2.5) ** 2, status='ok')

            def loss_variance(cls, result, config):
                return 0 # test 0 variance for once
        self.bandit = UniformBandit()

    def test_fit_uniform(self):
        bandit_algo = GPAlgo(self.bandit)
        bandit_algo.n_startup_jobs = 5
        serial_exp = SerialExperiment(bandit_algo)
        serial_exp.run(bandit_algo.n_startup_jobs)
        serial_exp.run(20)

        # a grid spacing would have used 25 points to cover 5 units of
        # distance
        # so be no more than 1/5**2 == .04.  Here we test that the GP gets the
        # error below .005
        assert min(serial_exp.losses()) < 5e-3, serial_exp.results

        # assert that the sampler has not exceeded the boundaries
        assert min([t['x'] for t in serial_exp.trials]) >= bandit_algo.xlim_low
        assert min([t['x'] for t in serial_exp.trials]) <= bandit_algo.xlim_high

        # XXX: assert that variance has been reduced along the whole uniform
        # range

        # if showing a plot...
        # xlim_low = -3.0   #XXX match UniformBandit
        # xlim_high = 2.0   #XXX match UniformBandit


class TestLognormal1D(unittest.TestCase):
    def setUp(self):
        class LognormalBandit(GensonBandit):
            def __init__(self, test_str):
                super(LognormalBandit, self).__init__(
                        source_string=test_str)

            def evaluate(cls, config, ctrl):
                return dict(
                        loss=(numpy.log(config['x']) - numpy.log(2)) ** 2,
                            status='ok')

            def loss_variance(cls, result, config):
                return .00001
        self.ln_bandit = LognormalBandit('{"x":lognormal(0,1)}')
        self.qln_bandit = LognormalBandit('{"x":qlognormal(5,2)}')

    def test_fit_lognormal(self):
        bandit_algo = GPAlgo(self.ln_bandit)
        bandit_algo.n_startup_jobs = 5
        serial_exp = SerialExperiment(bandit_algo)
        serial_exp.run(bandit_algo.n_startup_jobs)

        # check that the Lognormal kernel has been
        # identified as refinable
        k = bandit_algo.kernels[0]
        assert bandit_algo.is_refinable[k]
        assert bandit_algo.bounds[k][0] > 0

        serial_exp.run(25)

        assert min(serial_exp.losses()) < .005

        # check that  all points were positive
        assert min([t['x'] for t in serial_exp.trials]) > 0

        # the lenscale is about 1.8  Is that about right? What's right?
        print bandit_algo.kernels[0].lenscale()

    def test_fit_quantized_lognormal(self):
        bandit_algo = GPAlgo(self.qln_bandit)
        bandit_algo.n_startup_jobs = 5
        serial_exp = SerialExperiment(bandit_algo)
        serial_exp.run(bandit_algo.n_startup_jobs)

        # check that the Lognormal kernel has been
        # identified as refinable
        k = bandit_algo.kernels[0]
        assert bandit_algo.is_refinable[k]
        assert bandit_algo.bounds[k][0] > 0

        serial_exp.run(25)
        xvec = numpy.asarray([t['x'] for t in serial_exp.trials])
        if 0:
            show_bandit_algo(bandit_algo,
                    serial_exp.trials, 
                    serial_exp.results,
                    xlim_low=1,
                    xlim_high=xvec.max() + 1,
                    )

        assert min(serial_exp.losses()) == 0, (
                serial_exp.losses(), min(serial_exp.losses()))

        # check that  all points were positive
        assert xvec.min() > 0

        # assert that the step size was respected
        assert numpy.all(numpy.fmod(xvec, 1) == 0)

        # the lenscale is about 1.8  Is that about right? What's right?
        print bandit_algo.kernels[0].lenscale()


class TestGaussian2D(unittest.TestCase):
    class Bandit(GensonBandit):
        test_str = '{"x":gaussian(0,1), "y":gaussian(0,1)}'

        def __init__(self, a, b):
            GensonBandit.__init__(self, source_string=self.test_str)
            self.a = a
            self.b = b
        def evaluate(self, config, ctrl):
            return dict(loss=(
                    self.a * (config['x'] - 2) ** 2 +
                    self.b * (config['y'] - 2) ** 2),
                status='ok')

        def loss_variance(cls, result, config):
            return 0

    def test_2var_equal(self):
        algo = GPAlgo(TestGaussian2D.Bandit(1, 1))
        algo.n_startup_jobs = 5
        se = SerialExperiment(algo)
        se.run(25)
        l0 = algo.kernels[0].lenscale()
        l1 = algo.kernels[1].lenscale()
        assert .85 < l0 / l1 < 1.15

    def test_2var_unequal(self):
        algo = GPAlgo(TestGaussian2D.Bandit(1, 0))
        algo.n_startup_jobs = 25 
        se = SerialExperiment(algo)
        se.run(50)
        l0 = algo.kernels[0].lenscale()
        l1 = algo.kernels[1].lenscale()
        #N.B. a ratio in log-length scales is a big difference!
        assert l1 / l0 > 3


class GaussianBandit4var(GensonBandit):
    """
    This bandit allows testing continuous distributions nested inside choice
    variables.

    The loss actually only depends on 'a' or 'd'. So the length scales of 'b'
    and 'd' should go to infinity.
    """
    test_str = """{"p0":choice([{"a":gaussian(0,1),"b":gaussian(0,1)},
                                 {"c":gaussian(0,1),"d":gaussian(0,1)}])}"""

    def __init__(self, a, b, c, d):
        super(GaussianBandit4var, self).__init__(source_string=self.test_str)

        # relevances to loss function:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def evaluate(self, config, ctrl):
        return dict(loss=self.a * (config['p0'].get("a", 2) - 2) ** 2 + \
                         self.b * (config['p0'].get("b", 2) - 2) ** 2 + \
                         self.c * (config['p0'].get("c", 2) - 2) ** 2 + \
                         self.d * (config['p0'].get("d", 2) - 2) ** 2 ,
                    status='ok')

    def loss_variance(self, result, config):
        """Return uncertainty in reported loss.

        The function is technically deterministic (var = 0), but
        overestimating is ok.
        """
        return .1


def test_4var_all_relevant():
    bandit_algo = GPAlgo(GaussianBandit4var(1, .5, 2, 1))
    serial_exp = SerialExperiment(bandit_algo)
    bandit_algo.n_startup_jobs = 10
    for i in range(50):
        serial_exp.run(1)
    l0 = bandit_algo.kernels[0].log_lenscale.get_value()
    l1 = bandit_algo.kernels[1].log_lenscale.get_value()
    l2 = bandit_algo.kernels[2].log_lenscale.get_value()
    l3 = bandit_algo.kernels[3].log_lenscale.get_value()
    l4 = bandit_algo.kernels[4].log_lenscale.get_value()
    for k in bandit_algo.kernels:
        print 'last kernel fit', k, k.lenscale()
    assert min(serial_exp.losses()) < .05
    hyperopt.plotting.main_plot_vars(serial_exp, end_with_show=True)


def test_4var_some_irrelevant():
    bandit_algo = GPAlgo(GaussianBandit4var(1, 0, 0, 1))
    serial_exp = SerialExperiment(bandit_algo)
    bandit_algo.n_startup_jobs = 10
    for i in range(50):
        serial_exp.run(1)
    l0 = bandit_algo.kernels[0].log_lenscale.get_value()
    l1 = bandit_algo.kernels[1].log_lenscale.get_value()
    l2 = bandit_algo.kernels[2].log_lenscale.get_value()
    l3 = bandit_algo.kernels[3].log_lenscale.get_value()
    l4 = bandit_algo.kernels[4].log_lenscale.get_value()
    for k in bandit_algo.kernels:
        print 'last kernel fit', k, k.lenscale()
    assert min(serial_exp.losses()) < .05
    hyperopt.plotting.main_plot_vars(serial_exp, end_with_show=True)


def test_fit_categorical():
    numpy.random.seed(555)
    serial_exp = SerialExperiment(GPAlgo(TwoArms()))
    serial_exp.bandit_algo.n_startup_jobs = 7
    serial_exp.run(100)
    arm0count = len([t for t in serial_exp.trials if t['x'] == 0])
    arm1count = len([t for t in serial_exp.trials if t['x'] == 1])
    print 'arm 0 count', arm0count
    print 'arm 1 count', arm1count
    # this is just a test of the gm_algo candidate proposal mechanism
    # since the GP doesn't apply to discrete variables.
    assert arm0count > 60

    k = bandit_algo.kernels[0]
    assert bandit_algo.is_refinable[k]
    assert bandit_algo.bounds[k] == (0, 1)
    bandit_algo.show = False
    bandit_algo.use_base_suggest = True
    serial_exp.run(15)

    assert min(serial_exp.losses()) < .005
    assert bandit_algo.kernels[0].lenscale() < .25

    assert min([t['x'] for t in serial_exp.trials]) >= 0
    assert min([t['x'] for t in serial_exp.trials]) <= 1


def test_fit_dummy_dbn():
    bandit = Dummy_DBN_Base()
    bandit_algo = GPAlgo(bandit)
    bandit_algo.n_startup_jobs = 20
    serial_exp = SerialExperiment(bandit_algo)
    bandit_algo.show = False
    bandit_algo.use_base_suggest = True

    serial_exp.run(bandit_algo.n_startup_jobs)
    serial_exp.run(50) # use the GP for some iterations

    # No assertion here.
    # If it runs this far, it's already something.
