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
        print 'bandit params', bandit.params
        algo = Random(bandit)
        print 'algo params', algo.vh.params
        trials = Trials()
        experiment = Experiment(trials, algo, async=False)
        experiment.catch_bandit_exceptions = False
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

