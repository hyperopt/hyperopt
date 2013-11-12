"""
Verify that the sample bandits in bandits.py run, and and that a random
experiment proceeds as expected
"""
import unittest

from hyperopt import Trials
import hyperopt.bandits
from hyperopt.fmin import fmin
from hyperopt.rand import suggest

class BanditExperimentMixin(object):
    def test_basic(self):
        bandit = self._bandit_cls()
        #print 'bandit params', bandit.params, bandit
        #print 'algo params', algo.vh.params
        trials = Trials()
        fmin(lambda x: x, bandit.expr,
             trials=trials,
             algo=suggest,
             max_evals=self._n_steps)
        assert trials.average_best_error(bandit) - bandit.loss_target  < .2

    @classmethod
    def make(cls, bandit_cls, n_steps=500):
        class Tester(unittest.TestCase, cls):
            def setUp(self):
                self._n_steps = n_steps
                self._bandit_cls = bandit_cls
        Tester.__name__ = bandit_cls.__name__ + 'Tester'
        return Tester


quadratic1Tester = BanditExperimentMixin.make(hyperopt.bandits.quadratic1)
q1_lognormalTester = BanditExperimentMixin.make(hyperopt.bandits.q1_lognormal)
q1_choiceTester = BanditExperimentMixin.make(hyperopt.bandits.q1_choice)
n_armsTester = BanditExperimentMixin.make(hyperopt.bandits.n_arms)
distractorTester = BanditExperimentMixin.make(hyperopt.bandits.distractor)
gauss_waveTester = BanditExperimentMixin.make(hyperopt.bandits.gauss_wave)
gauss_wave2Tester = BanditExperimentMixin.make(hyperopt.bandits.gauss_wave2,
        n_steps=5000)


class CasePerBandit(object):
    # -- this is a mixin
    # -- Override self.work to execute a test for each kind of self.bandit

    def test_quadratic1(self):
        self.bandit = hyperopt.bandits.quadratic1()
        self.work()

    def test_q1lognormal(self):
        self.bandit = hyperopt.bandits.q1_lognormal()
        self.work()

    def test_twoarms(self):
        self.bandit = hyperopt.bandits.n_arms()
        self.work()

    def test_distractor(self):
        self.bandit = hyperopt.bandits.distractor()
        self.work()

    def test_gausswave(self):
        self.bandit = hyperopt.bandits.gauss_wave()
        self.work()

    def test_gausswave2(self):
        self.bandit = hyperopt.bandits.gauss_wave2()
        self.work()

    def test_many_dists(self):
        self.bandit = hyperopt.bandits.many_dists()
        self.work()


