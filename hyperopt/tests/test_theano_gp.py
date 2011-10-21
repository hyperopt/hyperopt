"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import unittest

import numpy

import matplotlib.pyplot as plt

import theano
from theano import tensor
from theano.tests.unittest_tools import verify_grad, seed_rng

from hyperopt.idxs_vals_rnd import IdxsValsList
from hyperopt.base import Bandit, BanditAlgo
from hyperopt.theano_gp import GP_BanditAlgo
from hyperopt.ht_dist2 import rSON2, normal
from hyperopt.experiments import SerialExperiment

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
                amt =  numpy.random.rand(2, 3) + 1
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
                amt =  numpy.random.rand(2, 3) + 1
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

# test that it can fit a GP to each of the simple variable types:
#  - normal
#  - uniform
#  - lognormal
#  - quantized lognormal
#  - categorical



def test_fit_normal():
    class B(Bandit):
        def __init__(self):
            Bandit.__init__(self, rSON2('x', normal(0, 1)))
        @classmethod
        def evaluate(cls, config, ctrl):
            return dict(loss=(config['x'] - 2)**2, status='ok')

        @classmethod
        def loss_variance(cls, result, config):
            return .1

    class A(GP_BanditAlgo):
        def suggest_from_model(self, trials, results, N):
            ivls = self.idxs_vals_by_status(trials, results)
            X_IVLs = ivls['x_IVLs']
            Ys = ivls['losses']
            Ys_var = ivls['losses_variance']
            prepared_data = self.prepare_GP_training_data(
                    X_IVLs, Ys, Ys_var)
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

            if self.show:

                plt.scatter(x_all[0].vals,
                        y_all * self._GP_y_std + self._GP_y_mean)
                plt.scatter(candidates[1], numpy.zeros_like(candidates[1]),
                    c='y')
                plt.scatter(candidates_opt[0].vals,
                        numpy.zeros_like(candidates[1]) - .1,
                        c='k')
                plt.xlim([-5, 5])
                xmesh = numpy.arange(-5, 5, .1)
                gp_mean, gp_var = self.GP_mean_variance(
                        IdxsValsList.fromlists([numpy.arange(len(xmesh))], [xmesh]))
                gp_EI = self.GP_EI(IdxsValsList.fromlists([numpy.arange(len(xmesh))], [xmesh]))
                print "GP_VAR", gp_var
                plt.plot(xmesh, gp_mean)
                plt.plot(xmesh, gp_mean + numpy.sqrt(gp_var), c='g')
                plt.plot(xmesh, gp_mean - numpy.sqrt(gp_var), c='g')
                plt.plot(xmesh, gp_EI, c='r')
                plt.show()


            best_idx = numpy.argmax(EI_opt)
            rval = IdxsValsList.fromflattened((
                    [candidates_opt[0].idxs[best_idx]],
                    [candidates_opt[0].vals[best_idx]]))
            return rval

    A.n_startup_jobs = 10
    se = SerialExperiment(A(B()))
    se.run(A.n_startup_jobs)

    assert len(se.trials) == len(se.results) == A.n_startup_jobs

    # now trigger the use of the GP, EI, etc.
    #A.show = False; se.run(6)
    A.show = True; se.run(4)

    #A.show = True; se.run(1)

# for a Bandit of two variables, of which one doesn't do anything
# test that the learned length scales are appropriate


# for a Bandit with
#    template one_of({'a':normal, 'b':normal}, {'c':normal, 'd':normal})
# and an evaluate that depends only on a or d,
# show that the length scales of b and c go to inf.
