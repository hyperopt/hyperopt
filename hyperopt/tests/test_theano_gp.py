"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra, Dan Yamins"
__copyright__ = "(c) 2011, James Bergstra, Dan Yamins"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import cPickle
import unittest
import nose

import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import theano
#from theano import tensor
#from theano.tests.unittest_tools import verify_grad, seed_rng

import hyperopt
#from hyperopt.idxs_vals_rnd import IdxsValsList
from hyperopt.bandits import TwoArms, GaussWave, GaussWave2
from hyperopt.base import Bandit, BanditAlgo
#from hyperopt.theano_gp import GP_BanditAlgo
#from hyperopt.theano_gm import AdaptiveParzenGM
#from hyperopt.ht_dist2 import rSON2, normal, uniform, one_of, lognormal
#from hyperopt.genson_bandits import GensonBandit
#from hyperopt.experiments import SerialExperiment
#from hyperopt.dbn import Dummy_DBN_Base, geom
import hyperopt.plotting


#GPAlgo = GP_BanditAlgo

#from hyperopt.theano_gp import SparseGramSet
#from hyperopt.theano_gp import SparseGramGet
#from hyperopt.theano_gp import sparse_gram_get
#from hyperopt.theano_gp import sparse_gram_set
#from hyperopt.theano_gp import sparse_gram_inc
#from hyperopt.theano_gp import sparse_gram_mul


class TestSparseUpdate(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()
        seed_rng()
        self.base = tensor.lmatrix()
        self.amt = tensor.lmatrix()
        self.i0 = tensor.lvector()
        self.i1 = tensor.lvector()
        self.zget = sparse_gram_get(self.base, self.i0, self.i1)
        self.zset = sparse_gram_set(self.base, self.amt, self.i0, self.i1)
        self.zinc = sparse_gram_inc(self.base, self.amt, self.i0, self.i1)
        self.zmul = sparse_gram_mul(self.base, self.amt, self.i0, self.i1)

        self.vbase0 = numpy.zeros((5, 6), dtype='int')
        self.vbase1 = (1 + numpy.zeros((5, 6), dtype='int'))
        self.vbase2 = (2 + numpy.zeros((5, 6), dtype='int'))
        self.vbase9 = (9 + numpy.zeros((5, 6), dtype='int'))
        self.vbaser = numpy.arange(30).reshape(5, 6).astype('int')
        self.vamt = (1 + numpy.arange(6)).reshape(2, 3).astype('int')
        self.vi0 = numpy.asarray([0, 3])
        self.vi1 = numpy.asarray([1, 2, 4])

    def test_extract_works(self):
        f = theano.function([self.base, self.i0, self.i1],
                self.zget)
        r = f(self.vbaser, self.vi0, self.vi1)
        assert r.shape == (2, 3), r
        assert numpy.all(r == [[1, 2, 4], [19, 20, 22]]), r

    def test_extract_with_negative_idxs(self):
        f = theano.function([self.base, self.i0, self.i1],
                self.zget)
        r = f(self.vbaser, [-5, -2], self.vi1)
        assert numpy.all(r == [[1, 2, 4], [19, 20, 22]]), r

        r = f(self.vbaser, [-5, -2], [1, 2, -2])
        assert numpy.all(r == [[1, 2, 4], [19, 20, 22]]), r

    def test_extract_works_with_dups(self):
        f = theano.function([self.base, self.i0, self.i1],
                self.zget)
        r = f(self.vbaser, [-5, -5], self.vi1)
        assert numpy.all(r == [[1, 2, 4], [1, 2, 4]]), r

    def test_extract_with_IndexError(self):
        f = theano.function([self.base, self.i0, self.i1],
                self.zget)
        self.assertRaises(IndexError,
                f, self.vbaser, [7], self.vi1)
        self.assertRaises(IndexError,
                f, self.vbaser, [-7], self.vi1)
        self.assertRaises(IndexError,
                f, self.vbaser, self.vi0, [7])
        self.assertRaises(IndexError,
                f, self.vbaser, self.vi0, [-7])

    def test_extract_with_wrong_rank(self):
        self.assertRaises(TypeError,
                sparse_gram_get, self.base, self.i0, tensor.lmatrix())
        self.assertRaises(TypeError,
                sparse_gram_get, self.base, tensor.lscalar(), self.i1)
        self.assertRaises(TypeError,
                sparse_gram_get, tensor.vector(), self.i0, self.i1)

    def test_extract_with_float_idxs(self):
        self.assertRaises(TypeError,
                sparse_gram_get, self.base, tensor.vector(), self.i1)
        self.assertRaises(TypeError,
                sparse_gram_get, self.base, self.i0, tensor.vector())

    def test_set_works(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zset)
        r = f(self.vbase9, self.vamt, self.vi0, self.vi1)
        assert r.shape == self.vbase0.shape, r
        assert numpy.all(r == [
            [9, 1, 2, 9, 3, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 4, 5, 9, 6, 9],
            [9, 9, 9, 9, 9, 9],
            ]), r

    def test_inc_works(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zinc)
        r = f(self.vbase2, self.vamt, self.vi0, self.vi1)
        assert r.shape == self.vbase0.shape, r
        assert numpy.all(r == [
            [2, 3, 4, 2, 5, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 6, 7, 2, 8, 2],
            [2, 2, 2, 2, 2, 2],
            ]), r

    def test_mul_works(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zset)
        r = f(self.vbase1, -self.vamt, self.vi0, self.vi1)
        assert r.shape == self.vbase0.shape, r
        assert numpy.all(r == [
            [1, -1, -2, 1, -3, 1],
            [1,  1,  1, 1,  1, 1],
            [1,  1,  1, 1,  1, 1],
            [1, -4, -5, 1, -6, 1],
            [1,  1,  1, 1,  1, 1],
            ]), r

    def test_inc_works_with_dups(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zinc)
        try:
            r = f(self.vbase2, self.vamt, [0, 0], self.vi1)
            assert r.shape == self.vbase0.shape, r
            assert numpy.all(r == [
                [2, 7, 9, 2, 11, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                ]), r
        except NotImplementedError:
            pass  # numpy doesn't support this for now

        try:
            r = f(self.vbase2, self.vamt, self.vi0, [1, 1, 1])
            assert r.shape == self.vbase0.shape, r
            assert numpy.all(r == [
                [2, 8, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 17, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                ]), r
        except NotImplementedError:
            pass  # numpy doesn't support this for now

    def test_inc_IndexError(self):
        f = theano.function(
                [self.base, self.amt, self.i0, self.i1],
                self.zinc)
        self.assertRaises(IndexError,
                f, self.vbase2, self.vamt, [10, 11], self.vi1)
        self.assertRaises(IndexError,
                f, self.vbase2, self.vamt, self.vi0, [10, 11, 12])
        self.assertRaises(IndexError,
                f, self.vbase2, self.vamt, [-10, 0], self.vi1)
        self.assertRaises(IndexError,
                f, self.vbase2, self.vamt, self.vi0, [0, 1, -10])

    def test_inc_size_mismatch(self):
        f = theano.function(
                [self.base, self.amt, self.i0, self.i1],
                self.zinc, allow_input_downcast=True)

        # the lengths of vi0 and vi1 have to match the shape of the increment
        # amount

        # the base condition works
        f(self.vbase2, numpy.ones((2, 3)), self.vi0, self.vi1)

        # but changing the size of amt triggers ValueError
        self.assertRaises(ValueError,
                f, self.vbase2, numpy.ones((2, 2)), self.vi0, self.vi1)
        self.assertRaises(ValueError,
                f, self.vbase2, numpy.ones((3, 3)), self.vi0, self.vi1)

    def test_inc_wrong_rank(self):
        self.assertRaises(TypeError,
              sparse_gram_inc, self.base, self.amt, self.i0, tensor.lmatrix())
        self.assertRaises(TypeError,
              sparse_gram_inc, self.base, self.amt, tensor.lscalar(), self.i1)
        self.assertRaises(TypeError,
              sparse_gram_inc, self.base, tensor.ltensor3(), self.i0, self.i1)
        self.assertRaises(TypeError,
              sparse_gram_inc, tensor.vector(), self.amt, self.i0, self.i1)

    def test_grad_get(self):
        def op(x):
            return sparse_gram_get(x, self.vi0, self.vi1)
        try:
            verify_grad(op, [1 + numpy.random.rand(4, 5)])
        except verify_grad.E_grad:
            print e.num_grad.gf
            print e.analytic_grad
            raise

    def test_grad_set(self):
        for oper in ('set', 'inc', 'mul'):
            def op(x, a):
                return SparseGramSet(oper)(x, a, self.vi0, self.vi1)
            try:
                base = numpy.random.rand(4, 5) + 1
                amt = numpy.random.rand(2, 3) + 1
                verify_grad(op, [base, amt])
            except verify_grad.E_grad, e:
                # This is supposed to work but doesn't :(
                #print e.num_grad.gf
                #print e.analytic_grad
                raise

    def test_grad_mul_0(self):
        for oper in ('set', 'inc', 'mul'):
            try:
                base = numpy.random.rand(4, 5) + 1
                amt = numpy.random.rand(2, 3) + 1
                amt[0] *= 0
                s_base = tensor.dmatrix()
                s_amt = tensor.dmatrix()
                verify_grad(
                        lambda b, a:
                            SparseGramSet(oper)(b, a, self.vi0, self.vi1),
                        [base, amt])
            except verify_grad.E_grad, e:
                # This is supposed to work but doesn't :(
                # print e.num_grad.gf
                # print e.analytic_grad
                raise




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
        raise nose.SkipTest()
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
        raise nose.SkipTest()
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
        self.xlim_low = -3
        self.xlim_high = 2

    def test_fit_uniform(self):
        bandit_algo = GPAlgo(self.bandit)
        bandit_algo.n_startup_jobs = 5
        serial_exp = SerialExperiment(bandit_algo)
        k = bandit_algo.kernels[0]
        assert bandit_algo.is_refinable[k]
        assert bandit_algo.bounds[k] == (self.xlim_low, self.xlim_high)

        serial_exp.run(bandit_algo.n_startup_jobs)
        serial_exp.run(20)

        # a grid spacing would have used 25 points to cover 5 units of
        # distance
        # so be no more than 1/5**2 == .04.  Here we test that the GP gets the
        # error below .005
        assert min(serial_exp.losses()) < 5e-3, serial_exp.results

        # assert that the sampler has not exceeded the boundaries
        assert min([t['x'] for t in serial_exp.trials]) >= self.xlim_low
        assert min([t['x'] for t in serial_exp.trials]) <= self.xlim_high

        # XXX: assert that variance has been reduced along the whole uniform
        # range

        # if showing a plot...
        # xlim_low = -3.0   #XXX match UniformBandit
        # xlim_high = 2.0   #XXX match UniformBandit


class TestLognormal1D(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()
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
    def setUp(self):
        raise nose.SkipTest()
    class Bandit(object):
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
        # XXX: consider using this tighter bound
        # when the mean and std are estimated from the
        # startup jobs.
        #assert min(se.losses()) < .005, min(se.losses())
        assert min(se.losses()) < .05, min(se.losses())

    def test_2var_unequal(self):
        algo = GPAlgo(TestGaussian2D.Bandit(1, 0))
        algo.n_startup_jobs = 25 
        se = SerialExperiment(algo)
        se.run(50)
        l0 = algo.kernels[0].lenscale()
        l1 = algo.kernels[1].lenscale()
        #N.B. a ratio in log-length scales is a big difference!
        assert l1 / l0 > 3
        assert min(se.losses()) < .005


class TestGaussian4D(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()

    class Bandit(object):
        """
        This bandit allows testing continuous distributions nested inside
        choice variables.

        The loss actually only depends on 'a' or 'd'. So the length scales of
        'b' and 'd' should go to infinity.
        """
        test_str = """{"p0":choice([{"a":gaussian(0,1),"b":gaussian(0,1)},
                                     {"c":gaussian(0,1),"d":gaussian(0,1)}])}"""

        def __init__(self, a, b, c, d):
            GensonBandit.__init__(self, source_string=self.test_str)
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
            return .1

    def test_4var_all_relevant(self):
        bandit_algo = GPAlgo(TestGaussian4D.Bandit(1, .5, 2, 1))
        serial_exp = SerialExperiment(bandit_algo)
        bandit_algo.n_startup_jobs = 10
        serial_exp.run(50)
        l0 = bandit_algo.kernels[0].lenscale()
        l1 = bandit_algo.kernels[1].lenscale()
        l2 = bandit_algo.kernels[2].lenscale()
        l3 = bandit_algo.kernels[3].lenscale()
        l4 = bandit_algo.kernels[4].lenscale()
        for k in bandit_algo.kernels:
            print 'last kernel fit', k, k.lenscale()
        assert min(serial_exp.losses()) < .05
        gauss_scales = numpy.asarray([l1, l2, l3, l4])
        assert gauss_scales.min() * 3 > gauss_scales.max()

    def test_4var_some_irrelevant(self):
        return # XXX enable when compilation is faster
        bandit_algo = GPAlgo(TestGaussian4D.Bandit(1, 0, 0, 1))
        serial_exp = SerialExperiment(bandit_algo)
        bandit_algo.n_startup_jobs = 10
        serial_exp.run(50)
        l0 = bandit_algo.kernels[0].lenscale()
        l1 = bandit_algo.kernels[1].lenscale()
        l2 = bandit_algo.kernels[2].lenscale()
        l3 = bandit_algo.kernels[3].lenscale()
        l4 = bandit_algo.kernels[4].lenscale()
        for k in bandit_algo.kernels:
            print 'last kernel fit', k, k.lenscale()
        assert min(serial_exp.losses()) < .05
        assert max(l1, l4) * 3 < min(l2, l3)


class TestGaussWave(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()
        numpy.random.seed(555)
        self.algo = GPAlgo(GaussWave())
        self.algo.n_startup_jobs = 20
        self.serial_exp = SerialExperiment(self.algo)

    def test_fit(self):
        for i in range(100):
            self.serial_exp.run(1)
            if i > self.algo.n_startup_jobs:
                print [k.lenscale() for k in self.algo.kernels]
                assert numpy.allclose(
                        numpy.diag(self.algo.GP_train_K()),
                        1.0)
        plt.plot(
                range(len(self.serial_exp.losses())),
                self.serial_exp.losses())
        plt.figure()
        hyperopt.plotting.main_plot_vars(self.serial_exp,
                end_with_show=False)


class TestGaussWave2(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()
        numpy.random.seed(555)
        self.algo = GPAlgo(GaussWave2())
        self.algo.n_startup_jobs = 20
        self.algo.EI_ambition = 0.75
        self.serial_exp = SerialExperiment(self.algo)

    def test_fit(self):
        for i in range(75):
            self.serial_exp.run(1)
            if i > self.algo.n_startup_jobs:
                print [k.lenscale() for k in self.algo.kernels]
                assert numpy.allclose(
                        numpy.diag(self.algo.GP_train_K()),
                        1.0)
        plt.plot(
                range(len(self.serial_exp.losses())),
                self.serial_exp.losses())
        plt.figure()
        hyperopt.plotting.main_plot_vars(self.serial_exp,
                end_with_show=False)


class TestGaussWave3(unittest.TestCase):
    """
    GP_BanditAlgo has different code paths for mulsets of one choice vs
    mulsets with multiple choices.  This tests both kinds.
    """
    def setUp(self):
        raise nose.SkipTest()
        class Bandit(GensonBandit):
            loss_target = -3
            test_str = """ {
                "x": uniform(-20, 20),
                "hf": choice([
                    {"kind": "raw"},
                    {"kind": "negcos", "amp": uniform(0, 1)}]),
                "y": choice([0,
                    uniform(3, 4),
                    uniform(2, 5),
                    uniform(1, 6),
                    choice([uniform(5, 6), uniform(4, 6.5)])])
                }
            """

            def __init__(self):
                GensonBandit.__init__(self, source_string=self.test_str)

            def evaluate(self, config, ctrl):
                r = numpy.random.randn() * .1
                x = config['x']
                r -= 2 * numpy.exp(-(x/5.0)**2) # up to 2
                if config['hf']['kind'] == 'negcos':
                    r -= numpy.sin(x) * config['hf']['amp']
                r -= config['y']
                return dict(loss=r, status='ok')

            def loss_variance(self, result, config=None):
                return 0.01

        self.algo = GPAlgo(Bandit())
        self.algo.n_startup_jobs = 5
        self.serial_exp = SerialExperiment(self.algo)

    def test_fit(self):
        for i in range(50):
            self.serial_exp.run(1)
            if i > self.algo.n_startup_jobs:
                print [k.lenscale() for k in self.algo.kernels]
                d = numpy.diag(self.algo.GP_train_K())
                #print 'max abs err', numpy.max(abs(d - 1))
                assert numpy.max(abs(d-1)) < .001
                assert 'float64' == str(d.dtype)
        plt.plot(
                range(len(self.serial_exp.losses())),
                self.serial_exp.losses())
        plt.figure()
        hyperopt.plotting.main_plot_vars(self.serial_exp,
                end_with_show=False)


class TestDummyDBN(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()

    def dbn_template0(self,
            dataset_name='skdata.larochelle_etal_2007.Rectangles',
            sup_min_epochs=300,
            sup_max_epochs=4000):
        template = rSON2(
            'preprocessing', one_of(
                rSON2(
                    'kind', 'raw'),
                rSON2(
                    'kind', 'zca',
                    'energy', uniform(0.5, 1.0))),
            'dataset_name', dataset_name,
            'sup_max_epochs', sup_max_epochs,
            'sup_min_epochs', sup_min_epochs,
            'iseed', one_of(5, 6, 7, 8),
            'batchsize', one_of(20, 100),
            'lr', lognormal(numpy.log(.01), 3),
            'lr_anneal_start', geom(100, 10000),
            'l2_penalty', one_of(0, lognormal(numpy.log(1.0e-6), 2)),
            'next_layer', None)
        return template

    def dbn_template1(self,
            dataset_name='skdata.larochelle_etal_2007.Rectangles',
            sup_min_epochs=300,
            sup_max_epochs=4000):
        template = rSON2(
            'preprocessing', one_of(
                rSON2(
                    'kind', 'raw'),
                rSON2(
                    'kind', 'zca',
                    'energy', uniform(0.5, 1.0))),
            'dataset_name', dataset_name,
            'sup_max_epochs', sup_max_epochs,
            'sup_min_epochs', sup_min_epochs,
            'iseed', one_of(5, 6, 7, 8),
            'batchsize', one_of(20, 100),
            'lr', lognormal(numpy.log(.01), 3),
            'lr_anneal_start', geom(100, 10000),
            'l2_penalty', one_of(0, lognormal(numpy.log(1.0e-6), 2)),
            'next_layer', one_of(None,
            rSON2(
                'n_hid', geom(2**7, 2**12, round=16),
                'W_init_dist', one_of('uniform', 'normal'),
                'W_init_algo', one_of('old', 'Xavier'),
                'W_init_algo_old_multiplier', lognormal(0.0, 1.0),
                'cd_epochs', geom(1, 3000),
                'cd_batchsize', 100,
                'cd_sample_v0s', one_of(False, True),
                'cd_lr', lognormal(numpy.log(.01), 2),
                'cd_lr_anneal_start', geom(10, 10000),
                'next_layer', None)))
        return template

    def dbn_template2(self,
            dataset_name='skdata.larochelle_etal_2007.Rectangles',
            sup_min_epochs=300,
            sup_max_epochs=4000):
        template = rSON2(
            'preprocessing', one_of(
                rSON2(
                    'kind', 'raw'),
                rSON2(
                    'kind', 'zca',
                    'energy', uniform(0.5, 1.0))),
            'dataset_name', dataset_name,
            'sup_max_epochs', sup_max_epochs,
            'sup_min_epochs', sup_min_epochs,
            'iseed', one_of(5, 6, 7, 8),
            'batchsize', one_of(20, 100),
            'lr', lognormal(numpy.log(.01), 3),
            'lr_anneal_start', geom(100, 10000),
            'l2_penalty', one_of(0, lognormal(numpy.log(1.0e-6), 2)),
            'next_layer', one_of(None,
            rSON2(
                'n_hid', geom(2**7, 2**12, round=16),
                'W_init_dist', one_of('uniform', 'normal'),
                'W_init_algo', one_of('old', 'Xavier'),
                'W_init_algo_old_multiplier', lognormal(0.0, 1.0),
                'cd_epochs', geom(1, 3000),
                'cd_batchsize', 100,
                'cd_sample_v0s', one_of(False, True),
                'cd_lr', lognormal(numpy.log(.01), 2),
                'cd_lr_anneal_start', geom(10, 10000),
                'next_layer', one_of(None,
                    rSON2(
                        'n_hid', geom(2**7, 2**12, round=16),
                        'W_init_dist', one_of('uniform', 'normal'),
                        'W_init_algo', one_of('old', 'Xavier'),
                        'W_init_algo_old_multiplier', lognormal(0.0, 1.0),
                        'cd_epochs', geom(1, 2000),
                        'cd_batchsize', 100,
                        'cd_sample_v0s', one_of(False, True),
                        'cd_lr', lognormal(numpy.log(.01), 2),
                        'cd_lr_anneal_start', geom(10, 10000),
                        'next_layer', None)))))
        return template

    def dbn_template3(self,
            dataset_name='skdata.larochelle_etal_2007.Rectangles',
            sup_min_epochs=300,
            sup_max_epochs=4000):
        template = rSON2(
            'preprocessing', one_of(
                rSON2(
                    'kind', 'raw'),
                rSON2(
                    'kind', 'zca',
                    'energy', uniform(0.5, 1.0))),
            'dataset_name', dataset_name,
            'sup_max_epochs', sup_max_epochs,
            'sup_min_epochs', sup_min_epochs,
            'iseed', one_of(5, 6, 7, 8),
            'batchsize', one_of(20, 100),
            'lr', lognormal(numpy.log(.01), 3),
            'lr_anneal_start', geom(100, 10000),
            'l2_penalty', one_of(0, lognormal(numpy.log(1.0e-6), 2)),
            'next_layer', one_of(None,
            rSON2(
                'n_hid', geom(2**7, 2**12, round=16),
                'W_init_dist', one_of('uniform', 'normal'),
                'W_init_algo', one_of('old', 'Xavier'),
                'W_init_algo_old_multiplier', lognormal(0.0, 1.0),
                'cd_epochs', geom(1, 3000),
                'cd_batchsize', 100,
                'cd_sample_v0s', one_of(False, True),
                'cd_lr', lognormal(numpy.log(.01), 2),
                'cd_lr_anneal_start', geom(10, 10000),
                'next_layer', one_of(None,
                    rSON2(
                        'n_hid', geom(2**7, 2**12, round=16),
                        'W_init_dist', one_of('uniform', 'normal'),
                        'W_init_algo', one_of('old', 'Xavier'),
                        'W_init_algo_old_multiplier', lognormal(0.0, 1.0),
                        'cd_epochs', geom(1, 2000),
                        'cd_batchsize', 100,
                        'cd_sample_v0s', one_of(False, True),
                        'cd_lr', lognormal(numpy.log(.01), 2),
                        'cd_lr_anneal_start', geom(10, 10000),
                        'next_layer', one_of(None,
                            rSON2(
                                'n_hid', geom(2**7, 2**12, round=16),
                                'W_init_dist', one_of('uniform', 'normal'),
                                'W_init_algo', one_of('old', 'Xavier'),
                                'W_init_algo_old_multiplier', lognormal(0., 1.),
                                'cd_epochs', geom(1, 1500),
                                'cd_batchsize', 100,
                                'cd_sample_v0s', one_of(False, True),
                                'cd_lr', lognormal(numpy.log(.01), 2),
                                'cd_lr_anneal_start', geom(10, 10000),
                                'next_layer', None,
                                )))))))
        return template

    def bandit(self, template):
        class Bandit(hyperopt.base.Bandit):
            def __init__(self, template):
                hyperopt.base.Bandit.__init__(self, template=template)
                self.rng = numpy.random.RandomState(234)
            def evaluate(self, argd, ctrl):
                rval = dict(dbn_train_fn_version=-1)
                # XXX: TODO: make up a loss function that depends on argd.
                rval['status'] = 'ok'
                rval['best_epoch_valid'] = float(self.rng.rand())
                rval['loss'] = 1.0 - rval['best_epoch_valid']
                return rval
        return Bandit(template)

    def test_fit0(self):
        bandit = self.bandit(self.dbn_template0())
        bandit_algo = GPAlgo(bandit)
        bandit_algo.n_startup_jobs = 20
        serial_exp = SerialExperiment(bandit_algo)
        for i in range(50):
            serial_exp.run(1)
            if i > bandit_algo.n_startup_jobs:
                #print 'LENSCALES',
                #print [k.lenscale() for k in bandit_algo.kernels]
                d = numpy.diag(bandit_algo.GP_train_K())
                #print 'max abs err', numpy.max(abs(d - 1))
                assert numpy.max(abs(d - 1)) < .0001
                assert 'float64' == str(d.dtype)

    def test_fit1(self):
        bandit = self.bandit(self.dbn_template1())
        bandit_algo = GPAlgo(bandit)
        bandit_algo.n_startup_jobs = 20
        serial_exp = SerialExperiment(bandit_algo)
        for i in range(50):
            serial_exp.run(1)
            if i > bandit_algo.n_startup_jobs:
                d = numpy.diag(bandit_algo.GP_train_K())
                #print 'max abs err', numpy.max(abs(d - 1))
                assert numpy.max(abs(d - 1)) < .0001
                assert 'float64' == str(d.dtype)

    def test_fit2(self):
        bandit = self.bandit(self.dbn_template2())
        bandit_algo = GPAlgo(bandit)
        bandit_algo.n_startup_jobs = 20
        serial_exp = SerialExperiment(bandit_algo)
        for i in range(50):
            serial_exp.run(1)
            if i > bandit_algo.n_startup_jobs:
                d = numpy.diag(bandit_algo.GP_train_K())
                #print 'max abs err', numpy.max(abs(d - 1))
                assert numpy.max(abs(d - 1)) < .0001
                assert 'float64' == str(d.dtype)

    def test_fit3(self):
        bandit = self.bandit(self.dbn_template3())
        bandit_algo = GPAlgo(bandit)
        bandit_algo.n_startup_jobs = 20
        serial_exp = SerialExperiment(bandit_algo)
        for i in range(50):
            serial_exp.run(1)
            if i > bandit_algo.n_startup_jobs:
                d = numpy.diag(bandit_algo.GP_train_K())
                #print 'max abs err', numpy.max(abs(d - 1))
                assert numpy.max(abs(d - 1)) < .0001
                assert 'float64' == str(d.dtype)
        # just getting to this point means that no NaNs were produced during
        # the calculations.


class TestPickle(unittest.TestCase):
    def setUp(self):
        raise nose.SkipTest()
        numpy.random.seed(555)
        self.algo_a = GPAlgo(GaussWave2())
        self.algo_a.n_startup_jobs = 10
        self.algo_a.EI_ambition = 0.75
        self.algo_b = GPAlgo(GaussWave2())
        self.algo_b.n_startup_jobs = 10
        self.algo_b.EI_ambition = 0.75
        self.exp_a = SerialExperiment(self.algo_a)
        self.exp_b = SerialExperiment(self.algo_b)

    def test_reproducible(self):
        self.exp_a.run(21)
        self.exp_b.run(21)
        for i, (ta, tb) in enumerate(zip(
                self.exp_a.trials,
                self.exp_b.trials)):
            print i, ta, tb
        print self.exp_a.losses()
        print self.exp_b.losses()
        # N.B. exact comparison, not approximate
        assert numpy.all(self.exp_a.losses() == self.exp_b.losses())

    def test_reproducible_w_recompiling(self):
        for i in range(21):
            self.exp_b.run(1)
            if not i % 5:
                todel = [k for k, v in self.algo_b.__dict__.items()
                        if isinstance(v, theano.compile.Function)]
                for name in todel:
                    delattr(self.algo_b, name)
        self.exp_a.run(21)
        for i, (ta, tb) in enumerate(zip(
                self.exp_a.trials,
                self.exp_b.trials)):
            print i, ta, tb
        print self.exp_a.losses()
        print self.exp_b.losses()
        # N.B. exact comparison, not approximate
        assert numpy.all(self.exp_a.losses() == self.exp_b.losses())

    def test_reproducible_w_pickling(self):
        self.exp_a.bandit_algo.trace_on = True
        self.exp_b.bandit_algo.trace_on = True
        ITERS = 12
        for i in range(ITERS):
            print 'running experiment b', i
            self.exp_b.run(1)
            if not i % 5:
                # This knocks out the theano functions
                # (see test_reproducible_w_recompiling)
                # but also deep-copies the rest of the experiment
                ####print 'pickling'
                pstr = cPickle.dumps(self.exp_b)
                ####print 'unpickling'
                self.exp_b = cPickle.loads(pstr)
        self.exp_a.run(ITERS)

        trace_a = self.exp_a.bandit_algo._trace
        trace_b = self.exp_b.bandit_algo._trace

        for ta, tb in zip(trace_a, trace_b):
            assert ta[0] == tb[0], (ta[0], tb[0])
            print 'matching', ta[0]
            na = numpy.asarray(ta[1])
            nb = numpy.asarray(tb[1])
            if not numpy.all(na == nb):
                print ta[0]
                print ''
                print na.shape
                print na
                print ''
                print nb.shape
                print nb
                print ''
                print (na - nb)
                assert 0
        for i, (ta, tb) in enumerate(zip(
                self.exp_a.trials,
                self.exp_b.trials)):
            ###print 'trial', i
            ###print '  exp a', ta
            ###print '  exp b', tb
            pass
        print self.exp_a.losses()
        print self.exp_b.losses()
        assert numpy.allclose(self.exp_a.losses(), self.exp_b.losses())


def test_fit_categorical():
    raise nose.SkipTest()
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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGaussian4D)
    unittest.TextTestRunner(verbosity=2).run(suite)
