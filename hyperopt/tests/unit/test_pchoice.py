from functools import partial
import numpy as np
import unittest
from hyperopt import hp, Trials, fmin, tpe, anneal, rand
import hyperopt.pyll.stochastic


class TestPChoice(unittest.TestCase):
    def test_basic(self):

        space = hp.pchoice(
            "naive_type",
            [(0.14, "gaussian"), (0.02, "multinomial"), (0.84, "bernoulli")],
        )
        a, b, c = 0, 0, 0
        rng = np.random.default_rng(123)
        for i in range(0, 1000):
            nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
            if nesto == "gaussian":
                a += 1
            elif nesto == "multinomial":
                b += 1
            elif nesto == "bernoulli":
                c += 1
        print((a, b, c))
        assert a + b + c == 1000
        assert 120 < a < 160
        assert 0 < b < 40
        assert 800 < c < 900

    def test_basic2(self):
        space = hp.choice(
            "normal_choice",
            [
                hp.pchoice("fsd", [(0.1, "first"), (0.8, "second"), (0.1, 2)]),
                hp.choice("something_else", [10, 20]),
            ],
        )
        a, b, c = 0, 0, 0
        rng = np.random.default_rng(123)
        for i in range(0, 1000):
            nesto = hyperopt.pyll.stochastic.sample(space, rng=rng)
            if nesto == "first":
                a += 1
            elif nesto == "second":
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
        space = hp.pchoice(
            "something",
            [
                (0.2, hp.pchoice("number", [(0.8, 2), (0.2, 1)])),
                (0.8, hp.pchoice("number1", [(0.7, 5), (0.3, 6)])),
            ],
        )
        a, b, c, d = 0, 0, 0, 0
        rng = np.random.default_rng(123)
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
        assert a * 0.3 > b  # a * 1.2 > 4 * b
        assert c * 3 * 1.2 > d * 7


class TestSimpleFMin(unittest.TestCase):
    # test that that a space with a pchoice in it is
    # (a) accepted for each algo (random, tpe, anneal)
    # and
    # (b) handled correctly.
    #

    def setUp(self):
        self.space = hp.pchoice("a", [(0.1, 0), (0.2, 1), (0.3, 2), (0.4, 3)])
        self.trials = Trials()

    def objective(self, a):
        return [1, 1, 1, 0][a]

    def test_random(self):
        max_evals = 150
        fmin(
            self.objective,
            space=self.space,
            trials=self.trials,
            algo=rand.suggest,
            rstate=np.random.default_rng(4),
            max_evals=max_evals,
        )

        a_vals = [t["misc"]["vals"]["a"][0] for t in self.trials.trials]
        counts = np.bincount(a_vals)
        assert counts[3] > max_evals * 0.35
        assert counts[3] < max_evals * 0.60

    def test_tpe(self):
        max_evals = 100
        fmin(
            self.objective,
            space=self.space,
            trials=self.trials,
            algo=partial(tpe.suggest, n_startup_jobs=10),
            rstate=np.random.default_rng(4),
            max_evals=max_evals,
        )

        a_vals = [t["misc"]["vals"]["a"][0] for t in self.trials.trials]
        counts = np.bincount(a_vals)
        assert counts[3] > max_evals * 0.6

    def test_anneal(self):
        max_evals = 100
        fmin(
            self.objective,
            space=self.space,
            trials=self.trials,
            algo=partial(anneal.suggest),
            rstate=np.random.default_rng(4),
            max_evals=max_evals,
        )

        a_vals = [t["misc"]["vals"]["a"][0] for t in self.trials.trials]
        counts = np.bincount(a_vals)
        assert counts[3] > max_evals * 0.6


def test_constant_fn_rand():
    space = hp.choice(
        "preprocess_choice",
        [
            {"pwhiten": hp.pchoice("whiten_randomPCA", [(0.3, False), (0.7, True)])},
            {"palgo": False},
            {"pthree": 7},
        ],
    )
    fmin(fn=lambda x: 1, space=space, algo=rand.suggest, max_evals=50)


def test_constant_fn_tpe():
    space = hp.choice(
        "preprocess_choice",
        [
            {"pwhiten": hp.pchoice("whiten_randomPCA", [(0.3, False), (0.7, True)])},
            {"palgo": False},
            {"pthree": 7},
        ],
    )
    fmin(
        fn=lambda x: 1,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        rstate=np.random.default_rng(44),
    )


def test_constant_fn_anneal():
    space = hp.choice(
        "preprocess_choice",
        [
            {"pwhiten": hp.pchoice("whiten_randomPCA", [(0.3, False), (0.7, True)])},
            {"palgo": False},
            {"pthree": 7},
        ],
    )
    fmin(fn=lambda x: 1, space=space, algo=anneal.suggest, max_evals=50)
