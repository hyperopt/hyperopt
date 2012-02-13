import unittest
import numpy as np
from hyperopt.base import CoinFlip, Random
from hyperopt import STATUS_STRINGS

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
        docs, idxs, vals = self.algo.suggest_docs_idxs_vals([], [], [], [], 1)
        print docs
        print idxs
        print vals
        assert len(docs) == 1
        assert len(idxs) == 1
        assert len(vals) == 1

    def test_suggest_5(self):
        docs, idxs, vals = self.algo.suggest_docs_idxs_vals([], [], [], [], 5)
        print docs
        print idxs
        print vals
        assert len(docs) == 5
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_2'] == range(5)
        assert np.all(vals['node_2'] == [0, 1, 0, 0, 0])
