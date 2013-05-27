import numpy as np
import unittest
from sklearn import datasets
from hyperopt import hp
import hyperopt.pyll.stochastic


class TestPChoice(unittest.TestCase):
    def setUp(self):

        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        self.X, self.y = X[y != 0, :2], y[y != 0]
        X_og, y_og = X, y

    def test_basic(self):

        space = hp.pchoice('naive_type', [.14, .02, .84],
                           ['gaussian', 'multinomial', 'bernoulli'])
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
            else:
                assert 0, nesto
        print(a, b, c)
        assert 120 < a < 160
        assert 0 < b < 40
        assert 800 < c < 900

    def test_basic2(self):
        space = hp.choice('normal_choice', [
            hp.pchoice('fsd', [.1, .8, .1], ['first', 'second', 2]),
            hp.choice('something_else', [10, 20])
        ])
        a, b, c = 0, 0, 0
        rng=np.random.RandomState(123)
        for i in range(0, 1000):
            nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
            if nesto == 'first':
                a += 1
            elif nesto == 'second':
                b += 1
            elif nesto == 2:
                c += 1
        print(a, b, c)
        assert b > 2 * a
        assert b > 2 * c
