"""
Verify that the sample bandits in bandits.py run, and and that a random
experiment proceeds as expected
"""
import unittest

from hyperopt import Random, Experiment, Trials
import hyperopt.bandits

class BanditExperimentMixin(object):
    def test_basic(self):
        bandit = self._bandit_cls()
        algo = Random(bandit)
        trials = Trials()
        experiment = Experiment(trials, algo, async=False)
        experiment.max_queue_len = 50
        experiment.run(self._n_steps)
        print
        print self._bandit_cls
        print bandit.loss_target
        print trials.average_best_error(bandit)
        assert trials.average_best_error(bandit) - bandit.loss_target  < .2
        print


    @classmethod
    def make(cls, bandit_cls, n_steps=500):
        class Tester(unittest.TestCase, cls):
            _n_steps = n_steps
            _bandit_cls = bandit_cls
        Tester.__name__ = bandit_cls.__name__ + 'Tester'
        return Tester

Quadratic1Tester = BanditExperimentMixin.make(hyperopt.bandits.Quadratic1)
Q1LognormalTester = BanditExperimentMixin.make(hyperopt.bandits.Q1Lognormal)
TwoArmsTester = BanditExperimentMixin.make(hyperopt.bandits.TwoArms)
DistractorTester = BanditExperimentMixin.make(hyperopt.bandits.Distractor)
GaussWaveTester = BanditExperimentMixin.make(hyperopt.bandits.GaussWave)
GaussWave2Tester = BanditExperimentMixin.make(hyperopt.bandits.GaussWave2,
        n_steps=5000)
