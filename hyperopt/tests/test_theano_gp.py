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

from hyperopt.idxs_vals_rnd import IdxsValsList
from hyperopt.base import Bandit, BanditAlgo
from hyperopt.theano_gp import GP_BanditAlgo
from hyperopt.ht_dist2 import rSON2, normal
from hyperopt.experiments import SerialExperiment

from hyperopt.theano_gp import sparse_gram_get
from hyperopt.theano_gp import sparse_gram_set
from hyperopt.theano_gp import sparse_gram_inc
from hyperopt.theano_gp import sparse_gram_mul

class TestSparseUpdate(unittest.TestCase):
    def setUp(self):
        self.base = tensor.lmatrix()
        self.amt = tensor.lmatrix()
        self.i0 = tensor.lvector()
        self.i1 = tensor.lvector()
        self.zget = sparse_gram_get(self.base, self.i0, self.i1)
        self.zset = sparse_gram_set(self.base, self.amt, self.i0, self.i1)
        self.zinc = sparse_gram_inc(self.base, self.amt, self.i0, self.i1)
        self.zmul = sparse_gram_mul(self.base, self.amt, self.i0, self.i1)

        self.vbase0 = numpy.zeros((5, 6), dtype='int')
        self.vbase1 = numpy.arange(30).reshape(5, 6).astype('int')
        self.vamt = numpy.arange(6).reshape(2, 3).astype('int')
        self.vi0 = numpy.asarray([0, 3])
        self.vi1 = numpy.asarray([1, 2, 4])

    def test_extract_works(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zget)
        r = f(self.vbase1, self.vamt, self.vi0, self.vi1)
        assert r.shape == self.vamt.shape, r
        assert numpy.all(r == [[1, 2, 4], [19, 20, 22]]), r

    def test_extract_with_negative_idxs(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zget)
        r = f(self.vbase1, self.vamt, [-5, -2], self.vi1)
        assert numpy.all(r == [[1, 2, 4], [19, 20, 22]]), r

        r = f(self.vbase1, self.vamt, [-5, -2], [1, 2, -2])
        assert numpy.all(r == [[1, 2, 4], [19, 20, 22]]), r

    def test_extract_works_with_dups(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zget)
        r = f(self.vbase1, self.vamt, [-5, -5], self.vi1)
        assert numpy.all(r == [[1, 2, 4], [1, 2, 4]]), r

    def test_extract_with_IndexError(self):
        f = theano.function([self.base, self.amt, self.i0, self.i1],
                self.zget)
        self.assertRaises(IndexError,
                f, self.vbase1, self.vamt, [7], self.vi1)
        self.assertRaises(IndexError,
                f, self.vbase1, self.vamt, [-7], self.vi1)
        self.assertRaises(IndexError,
                f, self.vbase1, self.vamt, self.vi0, [7])
        self.assertRaises(IndexError,
                f, self.vbase1, self.vamt, self.vi0, [-7])

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

    def test_set_works(self): pass
    def test_mul_works(self): pass
    def test_inc_works(self): pass
    def test_inc_works_with_dups(self): pass
    def test_inc_IndexError(self): pass
    def test_inc_wrong_rank(self): pass
    def test_grad_extract(self): pass
    def test_grad_set(self): pass
    def test_grad_inc(self): pass
    def test_grad_mul(self): pass


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
