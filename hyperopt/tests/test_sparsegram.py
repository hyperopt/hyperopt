"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra, Dan Yamins"
__copyright__ = "(c) 2011, James Bergstra, Dan Yamins"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import unittest

import numpy

import theano
from theano import tensor
from theano.tests.unittest_tools import verify_grad, seed_rng

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


