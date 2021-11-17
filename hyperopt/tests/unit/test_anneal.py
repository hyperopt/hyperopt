from functools import partial
import unittest
import numpy as np
from hyperopt import anneal
from hyperopt import rand
from hyperopt import Trials, fmin


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .test_domains import CasePerDomain


def passthrough(x):
    return x


class TestItJustRuns(unittest.TestCase, CasePerDomain):
    def work(self):
        trials = Trials()
        space = self.bandit.expr
        fmin(
            fn=passthrough,
            space=space,
            trials=trials,
            algo=anneal.suggest,
            max_evals=10,
        )


class TestItAtLeastSortOfWorks(unittest.TestCase, CasePerDomain):
    thresholds = dict(
        quadratic1=1e-5,
        q1_lognormal=0.01,
        distractor=-0.96,  # -- anneal is a strategy that can really
        #   get tricked by the distractor.
        gauss_wave=-2.0,
        gauss_wave2=-2.0,
        n_arms=-2.5,
        many_dists=0.0005,
        branin=0.7,
    )

    iters_thresholds = dict(
        # -- running a long way out tests overflow/underflow
        #    to some extent
        quadratic1=1000,
        many_dists=200,
        # -- anneal is pretty bad at this kind of function
        distractor=150,
        branin=200,
    )

    def setUp(self):
        self.olderr = np.seterr("raise")
        np.seterr(under="ignore")

    def tearDown(self, *args):
        np.seterr(**self.olderr)

    def work(self):
        bandit = self.bandit
        assert bandit.name is not None
        algo = partial(anneal.suggest)
        iters_thresholds = self.iters_thresholds.get(bandit.name, 50)

        trials = Trials()
        fmin(
            fn=passthrough,
            space=self.bandit.expr,
            trials=trials,
            algo=algo,
            max_evals=iters_thresholds,
            rstate=np.random.default_rng(8),
        )
        assert len(trials) == iters_thresholds

        rtrials = Trials()
        fmin(
            fn=passthrough,
            space=self.bandit.expr,
            trials=rtrials,
            algo=rand.suggest,
            max_evals=iters_thresholds,
            rstate=np.random.default_rng(8),
        )

        thresh = self.thresholds[bandit.name]
        assert min(trials.losses()) < thresh
