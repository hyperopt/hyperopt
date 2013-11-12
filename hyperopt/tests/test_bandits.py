import unittest

from hyperopt import Trials
from hyperopt.tests import bandits
from hyperopt.fmin import fmin
from hyperopt.rand import suggest

class DomainExperimentMixin(object):
    def test_basic(self):
        domain = self._domain_cls()
        #print 'domain params', domain.params, domain
        #print 'algo params', algo.vh.params
        trials = Trials()
        fmin(lambda x: x, domain.expr,
             trials=trials,
             algo=suggest,
             max_evals=self._n_steps)
        assert trials.average_best_error(domain) - domain.loss_target  < .2

    @classmethod
    def make(cls, domain_cls, n_steps=500):
        class Tester(unittest.TestCase, cls):
            def setUp(self):
                self._n_steps = n_steps
                self._domain_cls = domain_cls
        Tester.__name__ = domain_cls.__name__ + 'Tester'
        return Tester


quadratic1Tester = DomainExperimentMixin.make(bandits.quadratic1)
q1_lognormalTester = DomainExperimentMixin.make(bandits.q1_lognormal)
q1_choiceTester = DomainExperimentMixin.make(bandits.q1_choice)
n_armsTester = DomainExperimentMixin.make(bandits.n_arms)
distractorTester = DomainExperimentMixin.make(bandits.distractor)
gauss_waveTester = DomainExperimentMixin.make(bandits.gauss_wave)
gauss_wave2Tester = DomainExperimentMixin.make(bandits.gauss_wave2,
        n_steps=5000)


class CasePerDomain(object):
    # -- this is a mixin
    # -- Override self.work to execute a test for each kind of self.bandit

    def test_quadratic1(self):
        self.bandit = bandits.quadratic1()
        self.work()

    def test_q1lognormal(self):
        self.bandit = bandits.q1_lognormal()
        self.work()

    def test_twoarms(self):
        self.bandit = bandits.n_arms()
        self.work()

    def test_distractor(self):
        self.bandit = bandits.distractor()
        self.work()

    def test_gausswave(self):
        self.bandit = bandits.gauss_wave()
        self.work()

    def test_gausswave2(self):
        self.bandit = bandits.gauss_wave2()
        self.work()

    def test_many_dists(self):
        self.bandit = bandits.many_dists()
        self.work()


