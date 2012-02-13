import unittest
import numpy as np

from hyperopt import STATUS_STRINGS
from hyperopt import STATUS_OK
from hyperopt.base import Ctrl
from hyperopt.base import Trials
from hyperopt.base import CoinFlip
from hyperopt.base import Random
from hyperopt.base import Experiment
from hyperopt.base import Bandit
from hyperopt.vectorize import pretty_names

class BanditMixin(object):

    def test_dry_run(self):
        rval = self.bandit_cls.main_dryrun()
        assert 'loss' in rval
        assert 'status' in rval
        assert rval['status'] in STATUS_STRINGS

    @classmethod
    def make(cls, bandit_cls_to_test):
        class Tester(unittest.TestCase, cls):
            bandit_cls = bandit_cls_to_test
        Tester.__name__ = bandit_cls_to_test.__name__ + 'Tester'
        return Tester


CoinFlipTester = BanditMixin.make(CoinFlip)


class TestRandom(unittest.TestCase):
    def setUp(self):
        self.bandit = CoinFlip()
        self.algo = Random(self.bandit)

    def test_suggest_1(self):
        specs, idxs, vals = self.algo.suggest([0], [], [], [], [])
        print specs
        print idxs
        print vals
        assert len(specs) == 1
        assert len(idxs) == 1
        assert len(vals) == 1
        idxs['node_4'] == [0]

    def test_suggest_5(self):
        specs, idxs, vals = self.algo.suggest(range(5), [], [], [], [])
        print specs
        print idxs
        print vals
        assert len(specs) == 5
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_4'] == range(5)
        assert np.all(vals['node_4'] == [0, 1, 0, 0, 0])

    def test_arbitrary_range(self):
        new_ids = [-2, 0, 7, 'a', '007']
        specs, idxs, vals = self.algo.suggest(new_ids, [], [], [], [])
        assert len(specs) == 5
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_4'] == new_ids
        assert np.all(vals['node_4'] == [0, 1, 0, 0, 0])

# XXX: Test experiment loss code

class TestCoinFlipExperiment(unittest.TestCase):

    def setUp(self):
        self.bandit = CoinFlip()
        self.algo = Random(self.bandit)
        self.trials = Trials()
        self.experiment = Experiment(self.trials, self.algo, async=False)
        self.ctrl = Ctrl()

    def test_run_1(self):
        self.experiment.run(1)
        assert len(self.trials._trials) == 1

    def test_run_1_1_1(self):
        self.experiment.run(1)
        self.experiment.run(1)
        self.experiment.run(1)
        assert len(self.trials._trials) == 3
        print self.trials.idxs
        print self.trials.vals
        assert self.trials.idxs['node_4'] == [0, 1, 2]
        assert self.trials.vals['node_4'] == [0, 1, 0]


class ZeroBandit(Bandit):
    def __init__(self, template):
        Bandit.__init__(self, template)

    def evaluate(self, config, ctrl):
        return dict(loss=0.0, status=STATUS_OK)


from pyll import as_apply, scope, rec_eval, clone, dfs
uniform = scope.uniform
normal = scope.normal
one_of = scope.one_of



class TestConfigs(unittest.TestCase):
    def foo(self):
        bandit = ZeroBandit(self.expr)
        algo = Random(bandit)
        if hasattr(self, 'n_randints'):
            n_randints = len([nn for nn in algo.vh.name_by_id().values()
                if nn == 'randint'])
            assert n_randints == self.n_randints

        trials = Trials()
        experiment = Experiment(trials, algo, async=False)
        experiment.run(5)
        for trial in trials:
            print ''
            for nid in trial['idxs']:
                print algo.doc_coords[nid], trial['idxs'][nid], trial['vals'][nid]


    def test0(self):
        self.expr = as_apply(dict(p0=uniform(0, 1)))
        self.target = {'root["p0"]': 1.0}
        self.foo()

    def test1(self):
        self.expr = as_apply(dict(p0=normal(0, 1)))
        self.foo()

    def test2(self):
        self.expr = as_apply(dict(p0=one_of(0, 1)))
        self.foo()

    def test3(self):
        self.expr = as_apply(dict(p0=uniform(0, 1), p1=normal(0, 1)))
        self.foo()

    def test4(self):
        self.expr = as_apply(dict(p0=uniform(0, 1) + normal(0, 1)))
        self.foo()

    def test5(self):
        p0 = uniform(0, 1)
        self.expr = as_apply(dict(p0=p0, p1=p0))
        self.foo()

    def test6(self):
        p0 = uniform(0, 1)
        self.expr = as_apply(dict(p0=p0, p1=normal(p0, 1)))
        self.foo()

    def test7(self):
        p0 = uniform(0, 1)
        p1 = normal(0, 1)
        self.expr = as_apply(dict(
            p0=p0,
            p1=p1,
            p2=one_of(1, p0),
            p3=one_of(2, p1)))
        self.n_randints = 2
        self.foo()


