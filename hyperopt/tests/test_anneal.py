from __future__ import print_function
from __future__ import absolute_import
from builtins import range
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
            max_evals=10)


class TestItAtLeastSortOfWorks(unittest.TestCase, CasePerDomain):
    thresholds = dict(
        quadratic1=1e-5,
        q1_lognormal=0.01,
        distractor=-0.96,  # -- anneal is a strategy that can really
        #   get tricked by the distractor.
        gauss_wave=-2.0,
        gauss_wave2=-2.0,
        n_arms=-2.5,
        many_dists=.0005,
        branin=0.7,
    )

    LEN = dict(
        # -- running a long way out tests overflow/underflow
        #    to some extent
        quadratic1=1000,
        many_dists=200,
        # -- anneal is pretty bad at this kind of function
        distractor=150,
        # q1_lognormal=100,
        branin=200,
    )

    def setUp(self):
        self.olderr = np.seterr('raise')
        np.seterr(under='ignore')

    def tearDown(self, *args):
        np.seterr(**self.olderr)

    def work(self):
        bandit = self.bandit
        assert bandit.name is not None
        algo = partial(
            anneal.suggest,
        )
        LEN = self.LEN.get(bandit.name, 50)

        trials = Trials()
        fmin(fn=passthrough,
             space=self.bandit.expr,
             trials=trials,
             algo=algo,
             max_evals=LEN)
        assert len(trials) == LEN

        if 1:
            rtrials = Trials()
            fmin(fn=passthrough,
                 space=self.bandit.expr,
                 trials=rtrials,
                 algo=rand.suggest,
                 max_evals=LEN)
            print('RANDOM BEST 6:', list(sorted(rtrials.losses()))[:6])

        if 0:
            plt.subplot(2, 2, 1)
            plt.scatter(list(range(LEN)), trials.losses())
            plt.title('TPE losses')
            plt.subplot(2, 2, 2)
            plt.scatter(list(range(LEN)), ([s['x'] for s in trials.specs]))
            plt.title('TPE x')
            plt.subplot(2, 2, 3)
            plt.title('RND losses')
            plt.scatter(list(range(LEN)), rtrials.losses())
            plt.subplot(2, 2, 4)
            plt.title('RND x')
            plt.scatter(list(range(LEN)), ([s['x'] for s in rtrials.specs]))
            plt.show()
        if 0:
            plt.hist(
                [t['x'] for t in self.experiment.trials],
                bins=20)

        # print trials.losses()
        print('ANNEAL BEST 6:', list(sorted(trials.losses()))[:6])
        # logx = np.log([s['x'] for s in trials.specs])
        # print 'TPE MEAN', np.mean(logx)
        # print 'TPE STD ', np.std(logx)
        thresh = self.thresholds[bandit.name]
        print('Thresh', thresh)
        assert min(trials.losses()) < thresh
