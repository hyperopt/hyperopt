from __future__ import print_function
from builtins import range
from functools import partial
import numpy as np
import unittest
from hyperopt import hp, Trials, fmin, tpe, anneal, rand
import hyperopt.pyll.stochastic


class TestPChoice(unittest.TestCase):

    def test_basic(self):

        space = hp.pchoice('naive_type',
                           [(.14, 'gaussian'),
                            (.02, 'multinomial'),
                               (.84, 'bernoulli')])
        a, b, c = 0, 0, 0
        rng = np.random.RandomState(123)
        for i in range(0, 1000):
            nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
            if nesto == 'gaussian':
                a += 1
            elif nesto == 'multinomial':
                b += 1
            elif nesto == 'bernoulli':
                c += 1
        print((a, b, c))
        assert a + b + c == 1000
        assert 120 < a < 160
        assert 0 < b < 40
        assert 800 < c < 900

    def test_basic2(self):
        space = hp.choice('normal_choice', [
            hp.pchoice('fsd',
                       [(.1, 'first'),
                        (.8, 'second'),
                           (.1, 2)]),
            hp.choice('something_else', [10, 20])
        ])
        a, b, c = 0, 0, 0
        rng = np.random.RandomState(123)
        for i in range(0, 1000):
            nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
            if nesto == 'first':
                a += 1
            elif nesto == 'second':
                b += 1
            elif nesto == 2:
                c += 1
            elif nesto in (10, 20):
                pass
            else:
                assert 0, nesto
        print((a, b, c))
        assert b > 2 * a
        assert b > 2 * c

    def test_basic3(self):
        space = hp.pchoice('something', [
            (.2, hp.pchoice('number', [(.8, 2), (.2, 1)])),
            (.8, hp.pchoice('number1', [(.7, 5), (.3, 6)]))
        ])
        a, b, c, d = 0, 0, 0, 0
        rng = np.random.RandomState(123)
        for i in range(0, 2000):
            nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
            if nesto == 2:
                a += 1
            elif nesto == 1:
                b += 1
            elif nesto == 5:
                c += 1
            elif nesto == 6:
                d += 1
            else:
                assert 0, nesto
        print((a, b, c, d))
        assert a + b + c + d == 2000
        assert 300 < a + b < 500
        assert 1500 < c + d < 1700
        assert a * .3 > b  # a * 1.2 > 4 * b
        assert c * 3 * 1.2 > d * 7


class TestSimpleFMin(unittest.TestCase):
    # test that that a space with a pchoice in it is
    # (a) accepted by various algos and
    # (b) handled correctly.
    #

    def setUp(self):
        self.space = hp.pchoice('a', [
            (.1, 0),
            (.2, 1),
            (.3, 2),
            (.4, 3)])
        self.trials = Trials()

    def objective(self, a):
        return [1, 1, 1, 0][a]

    def test_random(self):
        # test that that a space with a pchoice in it is
        # (a) accepted by tpe.suggest and
        # (b) handled correctly.
        N = 150
        fmin(self.objective,
             space=self.space,
             trials=self.trials,
             algo=rand.suggest,
             max_evals=N)

        a_vals = [t['misc']['vals']['a'][0] for t in self.trials.trials]
        counts = np.bincount(a_vals)
        print(counts)
        assert counts[3] > N * .35
        assert counts[3] < N * .60

    def test_tpe(self):
        N = 100
        fmin(self.objective,
             space=self.space,
             trials=self.trials,
             algo=partial(tpe.suggest, n_startup_jobs=10),
             max_evals=N)

        a_vals = [t['misc']['vals']['a'][0] for t in self.trials.trials]
        counts = np.bincount(a_vals)
        print(counts)
        assert counts[3] > N * .6

    def test_anneal(self):
        N = 100
        fmin(self.objective,
             space=self.space,
             trials=self.trials,
             algo=partial(anneal.suggest),
             max_evals=N)

        a_vals = [t['misc']['vals']['a'][0] for t in self.trials.trials]
        counts = np.bincount(a_vals)
        print(counts)
        assert counts[3] > N * .6


def test_bug1_rand():
    space = hp.choice('preprocess_choice', [
        {'pwhiten': hp.pchoice('whiten_randomPCA',
                               [(.3, False), (.7, True)])},
        {'palgo': False},
        {'pthree': 7}])
    fmin(fn=lambda x: 1,
         space=space,
         algo=rand.suggest,
         max_evals=50)


def test_bug1_tpe():
    space = hp.choice('preprocess_choice', [
        {'pwhiten': hp.pchoice('whiten_randomPCA',
                               [(.3, False), (.7, True)])},
        {'palgo': False},
        {'pthree': 7}])
    fmin(fn=lambda x: 1,
         space=space,
         algo=tpe.suggest,
         max_evals=50)


def test_bug1_anneal():
    space = hp.choice('preprocess_choice', [
        {'pwhiten': hp.pchoice('whiten_randomPCA',
                               [(.3, False), (.7, True)])},
        {'palgo': False},
        {'pthree': 7}])
    fmin(fn=lambda x: 1,
         space=space,
         algo=anneal.suggest,
         max_evals=50)
