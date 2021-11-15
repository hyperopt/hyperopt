import unittest
from functools import partial

import numpy as np
from hyperopt import hp, Trials, fmin, rand, tpe
import hyperopt.pyll.stochastic


def test_basic():

    space = hp.randint("a", 5)
    x = np.zeros(5)
    rng = np.random.default_rng(123)
    for i in range(0, 1000):
        nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
        x[nesto] += 1

    print(x)
    for i in x:
        assert 100 < i < 300


def test_basic2():

    space = hp.randint("a", 5, 15)
    x = np.zeros(15)
    rng = np.random.default_rng(123)
    for i in range(0, 1000):
        nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
        x[nesto] += 1

    print(x)
    for i in range(5):
        assert x[i] == 0
    for i in range(5, 15):
        assert 80 < x[i] < 120


class TestSimpleFMin(unittest.TestCase):
    # test that that a space with a randint in it is
    # (a) accepted for each algo (random, tpe)
    # and
    # (b) handled correctly in fmin, finding the solution in the constrained space
    #

    def setUp(self):
        self.space = hp.randint("t", 2, 100)
        self.trials = Trials()

    def objective(self, a):
        # an objective function with roots at 3, 10, 50
        return abs(np.poly1d([1, -63, 680, -1500])(a))

    def test_random_runs(self):
        max_evals = 150
        fmin(
            self.objective,
            space=self.space,
            trials=self.trials,
            algo=rand.suggest,
            rstate=np.random.default_rng(4),
            max_evals=max_evals,
        )

        values = [t["misc"]["vals"]["t"][0] for t in self.trials.trials]
        counts = np.bincount(values, minlength=100)
        assert counts[:2].sum() == 0

    def test_tpe_runs(self):
        max_evals = 100
        fmin(
            self.objective,
            space=self.space,
            trials=self.trials,
            algo=partial(tpe.suggest, n_startup_jobs=10),
            rstate=np.random.default_rng(4),
            max_evals=max_evals,
        )

        values = [t["misc"]["vals"]["t"][0] for t in self.trials.trials]
        counts = np.bincount(values, minlength=100)
        assert counts[:2].sum() == 0

    def test_random_finds_constrained_solution(self):
        max_evals = 150

        # (2, 7), (2, 30), (2, 100), (5, 30), (5, 100), (20, 100)
        for lower, upper in zip([2, 2, 2, 5, 5, 20], [7, 30, 100, 30, 100, 100]):
            best = fmin(
                self.objective,
                space=hp.randint("t", lower, upper),
                algo=rand.suggest,
                rstate=np.random.default_rng(4),
                max_evals=max_evals,
            )
            expected = [i for i in [3, 10, 50] if lower <= i < upper]
            assert best["t"] in expected

    def test_tpe_finds_constrained_solution(self):
        max_evals = 150

        # (2, 7), (2, 30), (2, 100), (5, 30), (5, 100), (20, 100)
        for lower, upper in zip([2, 2, 2, 5, 5, 20], [7, 30, 100, 30, 100, 100]):
            best = fmin(
                self.objective,
                space=hp.randint("t", lower, upper),
                algo=tpe.suggest,
                rstate=np.random.default_rng(4),
                max_evals=max_evals,
            )
            expected = [i for i in [3, 10, 50] if lower <= i < upper]
            assert best["t"] in expected
