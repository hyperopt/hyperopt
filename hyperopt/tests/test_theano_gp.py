"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
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
import hyperopt.plotting

from hyperopt.theano_gp import SparseGramSet
from hyperopt.theano_gp import SparseGramGet
from hyperopt.theano_gp import sparse_gram_get
from hyperopt.theano_gp import sparse_gram_set
from hyperopt.theano_gp import sparse_gram_inc

from hyperopt.theano_gp import sparse_gram_mul


class TestSparseUpdate(unittest.TestCase):
    def setUp(self):
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


class GPAlgo(GP_BanditAlgo):
    use_base_suggest = True
    xlim_low = -5
    xlim_high = 5
    def suggest_from_model(self, trials, results, N):
        if self.use_base_suggest:
            return GP_BanditAlgo.suggest_from_model(self,
                    trials, results, N)

        ivls = self.idxs_vals_by_status(trials, results)
        X_IVLs = ivls['x_IVLs']
        Ys = ivls['losses']
        Ys_var = ivls['losses_variance']
        prepared_data = self.prepare_GP_training_data(ivls)
        x_all, y_all, y_mean, y_var, y_std = prepared_data
        self.fit_GP(*prepared_data)

        candidates = self._prior_sampler(5)
        EI = self.GP_EI(IdxsValsList.fromflattened(candidates))
        print ''
        print 'Candidates'
        print candidates[0]
        print candidates[1]
        print EI
        #print 'optimizing candidates'
        candidates_opt = self.GP_EI_optimize(
                IdxsValsList.fromflattened(candidates))
        EI_opt = self.GP_EI(candidates_opt)
        print ''
        print 'Optimized candidates'
        print candidates_opt[0].idxs
        print candidates_opt[0].vals
        print EI_opt

        num = len(candidates_opt)

        if self.show:

            plt.scatter(x_all[0].vals,
                    y_all * self._GP_y_std + self._GP_y_mean)
            plt.scatter(candidates[1], numpy.zeros_like(candidates[1]),
                c='y')
            plt.scatter(candidates_opt[0].vals,
                    numpy.zeros_like(candidates[1]) - .1,
                    c='k')


            plt.figure()

            plt.xlim([self.xlim_low, self.xlim_high])
            xmesh = numpy.linspace(self.xlim_low, self.xlim_high)
            N = len(xmesh)
            XmeshN = [numpy.arange(N) for _ind in range(num)]
            Xmesh = [numpy.linspace(self.xlim_low, self.xlim_high)
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

        best_idx = numpy.argmax(EI_opt)
        args = []
        for c_opt in candidates_opt:
            args.append([c_opt.idxs[best_idx]])
            args.append([c_opt.vals[best_idx]])
        rval = IdxsValsList.fromflattened(tuple(args))
        return rval


class GaussianBandit(GensonBandit):
    test_str = '{"x":gaussian(0,1)}'

    def __init__(self):
        super(GaussianBandit, self).__init__(source_string=self.test_str)

    @classmethod
    def evaluate(cls, config, ctrl):
        return dict(loss=(config['x'] - 2) ** 2, status='ok')

    @classmethod
    def loss_variance(cls, result, config):
        return .1


class UniformBandit(GensonBandit):
    test_str = '{"x":uniform(0,1)}'

    def __init__(self):
        super(UniformBandit, self).__init__(source_string=self.test_str)

    @classmethod
    def evaluate(cls, config, ctrl):
        return dict(loss=(config['x'] - .5) ** 2, status='ok')

    @classmethod
    def loss_variance(cls, result, config):
        return .01 ** 2


class LognormalBandit(GensonBandit):
    test_str = '{"x":lognormal(0,1)}'

    def __init__(self):
        super(LognormalBandit, self).__init__(source_string=self.test_str)

    @classmethod
    def evaluate(cls, config, ctrl):
        return dict(loss=(config['x'] - 2) ** 2, status='ok')

    @classmethod
    def loss_variance(cls, result, config):
        return .1        


class QLognormalBandit(GensonBandit):
    test_str = '{"x":qlognormal(0,1)}'

    def __init__(self):
        super(QLognormalBandit, self).__init__(source_string=self.test_str)

    @classmethod
    def evaluate(cls, config, ctrl):
        return dict(loss=(config['x'] - 2) ** 2, status='ok')

    @classmethod
    def loss_variance(cls, result, config):
        return .1  


class GaussianBandit2var(GensonBandit):
    test_str = '{"x":gaussian(0,1), "y":gaussian(0,1)}'

    def __init__(self, a, b):
        super(GaussianBandit2var, self).__init__(source_string=self.test_str)
        GaussianBandit2var.a = a
        GaussianBandit2var.b = b
    @classmethod
    def evaluate(cls, config, ctrl):
        return dict(loss=cls.a * (config['x'] - 2) ** 2 + \
                                   cls.b * (config['y'] - 2) ** 2, status='ok')

    @classmethod
    def loss_variance(cls, result, config):
        return .1


def fit_base(A, B, *args, **kwargs):
    A.n_startup_jobs = 7

    n_iter = kwargs.pop('n_iter', 40)
    serial_exp = SerialExperiment(A(B(*args, **kwargs)))
    serial_exp.run(A.n_startup_jobs)

    assert len(serial_exp.trials) == len(serial_exp.results)
    assert len(serial_exp.trials) == A.n_startup_jobs

    def run_then_show(N):
        if N > 1:
            A.show = False
            A.use_base_suggest = True
            serial_exp.run(N - 1)
        A.show = True
        A.use_base_suggest = False
        serial_exp.run(1)
        return serial_exp

    return run_then_show(n_iter)


def test_fit_normal():
    fit_base(GPAlgo, GaussianBandit)


def test_2var_equal():
    se = fit_base(GPAlgo, GaussianBandit2var, 1, 1)
    l0 = se.bandit_algo.kernels[0].log_lenscale.get_value()
    l1 = se.bandit_algo.kernels[1].log_lenscale.get_value()
    assert .85 < l0 / l1 < 1.15


def test_2var_unequal():
    se = fit_base(GPAlgo, GaussianBandit2var, 1, 0)
    l0 = se.bandit_algo.kernels[0].log_lenscale.get_value()
    l1 = se.bandit_algo.kernels[1].log_lenscale.get_value()
    #N.B. a ratio in log-length scales is a big difference!
    assert l1 / l0 > 5


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
    assert arm0count > 60


def test_fit_uniform():
    bandit = UniformBandit()
    bandit_algo = GPAlgo(bandit)
    bandit_algo.n_startup_jobs = 5
    serial_exp = SerialExperiment(bandit_algo)
    serial_exp.run(bandit_algo.n_startup_jobs)
    bandit_algo.xlim_low = 0.0   #XXX match UniformBandit
    bandit_algo.xlim_high = 1.0   #XXX match UniformBandit

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


def test_fit_lognormal():
    fit_base(GPAlgo, LognormalBandit)


def test_fit_quantized_lognormal():
    fit_base(GPAlgo, QLognormalBandit)
