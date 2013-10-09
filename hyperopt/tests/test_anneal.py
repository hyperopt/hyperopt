import unittest
from hyperopt import anneal
from hyperopt import Trials, fmin

from test_bandits import CasePerBandit

def passthrough(x):
    return x

class TestItJustRuns(unittest.TestCase, CasePerBandit):
    def work(self):
        trials = Trials()
        space = self.bandit.expr
        fmin(
            fn=passthrough,
            space=space,
            trials=trials,
            algo=anneal.suggest,
            max_evals=10)


class TestItActuallyWorks(unittest.TestCase, CasePerBandit):
    thresholds = dict(
            quadratic1=1e-5,
            q1_lognormal=0.01,
            distractor=-1.96,
            gauss_wave=-2.0,
            gauss_wave2=-2.0,
            n_arms=-2.5,
            many_dists=.0005,
            )

    LEN = dict(
            # -- running a long way out tests overflow/underflow
            #    to some extent
            quadratic1=1000,
            many_dists=200,
            distractor=100,
            #XXX
            q1_lognormal=250,
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
        exp = Experiment(trials, algo)
        exp.catch_bandit_exceptions = False
        exp.run(LEN)
        assert len(trials) == LEN

        if 1:
            rtrials = Trials()
            exp = Experiment(rtrials, Random(bandit))
            exp.run(LEN)
            print 'RANDOM MINS', list(sorted(rtrials.losses()))[:6]
            #logx = np.log([s['x'] for s in rtrials.specs])
            #print 'RND MEAN', np.mean(logx)
            #print 'RND STD ', np.std(logx)

        if 0:
            plt.subplot(2, 2, 1)
            plt.scatter(range(LEN), trials.losses())
            plt.title('TPE losses')
            plt.subplot(2, 2, 2)
            plt.scatter(range(LEN), ([s['x'] for s in trials.specs]))
            plt.title('TPE x')
            plt.subplot(2, 2, 3)
            plt.title('RND losses')
            plt.scatter(range(LEN), rtrials.losses())
            plt.subplot(2, 2, 4)
            plt.title('RND x')
            plt.scatter(range(LEN), ([s['x'] for s in rtrials.specs]))
            plt.show()
        if 0:
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)

        #print trials.losses()
        print 'TPE    MINS', list(sorted(trials.losses()))[:6]
        #logx = np.log([s['x'] for s in trials.specs])
        #print 'TPE MEAN', np.mean(logx)
        #print 'TPE STD ', np.std(logx)
        thresh = self.thresholds[bandit.name]
        print 'Thresh', thresh
        assert min(trials.losses()) < thresh


