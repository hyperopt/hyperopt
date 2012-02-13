import unittest
import numpy as np

from hyperopt import STATUS_STRINGS
from hyperopt.base import Ctrl
from hyperopt.base import CoinFlip
from hyperopt.base import Random
from hyperopt.base import SerialExperiment

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
        idxs['node_2'] == [0]

    def test_suggest_5(self):
        specs, idxs, vals = self.algo.suggest(range(5), [], [], [], [])
        print specs
        print idxs
        print vals
        assert len(specs) == 5
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_2'] == range(5)
        assert np.all(vals['node_2'] == [0, 1, 0, 0, 0])

    def test_arbitrary_range(self):
        new_ids = [-2, 0, 7, 'a', '007']
        specs, idxs, vals = self.algo.suggest(new_ids, [], [], [], [])
        assert len(specs) == 5
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_2'] == new_ids
        assert np.all(vals['node_2'] == [0, 1, 0, 0, 0])

# XXX: Test experiment loss code

class TestSerialExperiment(unittest.TestCase):

    def setUp(self):
        self.bandit = CoinFlip()
        self.algo = Random(self.bandit)
        self.experiment = SerialExperiment(self.algo)
        self.ctrl = Ctrl()

    def test_run_1(self):
        self.experiment.run(1)
