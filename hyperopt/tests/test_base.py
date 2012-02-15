import unittest
import numpy as np

from pyll import as_apply, scope, rec_eval, clone, dfs
uniform = scope.uniform
normal = scope.normal
one_of = scope.one_of

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
        specs, idxs, vals = self.algo.suggest([0], [], [], {}, {})
        print specs
        print idxs
        print vals
        assert len(specs) == 1
        assert len(idxs) == 1
        assert len(vals) == 1
        idxs['node_4'] == [0]

    def test_suggest_5(self):
        specs, idxs, vals = self.algo.suggest(range(5), [], [], {}, {})
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
        specs, idxs, vals = self.algo.suggest(new_ids, [], [], {}, {})
        assert len(specs) == 5
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_4'] == new_ids
        assert np.all(vals['node_4'] == [0, 1, 0, 0, 0])


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



class TestConfigs(unittest.TestCase):
    def foo(self):
        self.bandit = bandit = ZeroBandit(self.expr)
        self.algo = algo = Random(bandit)
        if hasattr(self, 'n_randints'):
            n_randints = len([nn for nn in algo.vh.name_by_id().values()
                if nn == 'randint'])
            assert n_randints == self.n_randints

        self.trials = trials = Trials()
        self.experiment = Experiment(trials, algo, async=False)
        self.experiment.run(5)
        trials = list(self.trials)
        self.output = output = []
        for trial in trials:
            print ''
            tmp = []
            for nid in trial['idxs']:
                thing = self.algo.doc_coords[nid], trial['idxs'][nid], trial['vals'][nid]
                print thing
                tmp.append(thing)
            tmp.sort()
            output.append(tmp)
        print repr(output)
        print repr(self.wanted)
        assert output == self.wanted

    def test0(self):
        self.expr = as_apply(dict(p0=uniform(0, 1)))
        self.wanted = [
                [('p0', [0], [0.69646918559786164])],
                [('p0', [1], [0.28613933495037946])],
                [('p0', [2], [0.22685145356420311])],
                [('p0', [3], [0.55131476908289123])],
                [('p0', [4], [0.71946896978556307])]]
        self.foo()

    def test1(self):
        self.expr = as_apply(dict(p0=normal(0, 1)))
        self.wanted = [
                [('p0', [0], [-1.0856306033005612])],
                [('p0', [1], [0.99734544658358582])],
                [('p0', [2], [0.28297849805199204])],
                [('p0', [3], [-1.506294713918092])],
                [('p0', [4], [-0.57860025196853637])]]
        self.foo()

    def test2(self):
        self.expr = as_apply(dict(p0=one_of(0, 1)))
        self.wanted = [
                [('p0.randint', [0], [0])], [('p0.randint', [1], [1])],
                [('p0.randint', [2], [0])], [('p0.randint', [3], [0])],
                [('p0.randint', [4], [0])]]
        self.foo()

    def test3(self):
        self.expr = as_apply(dict(p0=uniform(0, 1), p1=normal(0, 1)))
        self.wanted = [
                [('p0', [0], [0.69646918559786164]),
                    ('p1', [0], [-0.95209720686132215])],
                [('p0', [1], [0.55131476908289123]),
                    ('p1', [1], [-0.74544105948265826])],
                [('p0', [2], [0.71946896978556307]),
                    ('p1', [2], [0.32210606833962163])],
                [('p0', [3], [0.68482973858486329]),
                    ('p1', [3], [-0.0515177209393851])],
                [('p0', [4], [0.48093190148436094]),
                    ('p1', [4], [-1.6193000650367457])]]
        self.foo()

    def test4(self):
        self.expr = as_apply(dict(p0=uniform(0, 1) + normal(0, 1)))
        self.wanted = [
                [('p0.arg:0', [0], [0.69646918559786164]),
                    ('p0.arg:1', [0], [-0.95209720686132215])],
                [('p0.arg:0', [1], [0.55131476908289123]),
                    ('p0.arg:1', [1], [-0.74544105948265826])],
                [('p0.arg:0', [2], [0.71946896978556307]),
                    ('p0.arg:1', [2], [0.32210606833962163])],
                [('p0.arg:0', [3], [0.68482973858486329]),
                    ('p0.arg:1', [3], [-0.0515177209393851])],
                [('p0.arg:0', [4], [0.48093190148436094]),
                    ('p0.arg:1', [4], [-1.6193000650367457])]]
        self.foo()

    def test5(self):
        p0 = uniform(0, 1)
        self.expr = as_apply(dict(p0=p0, p1=p0))
        self.wanted = [[('p0', [0], [0.69646918559786164])], [('p0', [1],
                    [0.28613933495037946])], [('p0', [2],
                        [0.22685145356420311])], [('p0', [3],
                            [0.55131476908289123])], [('p0', [4],
                                [0.71946896978556307])]]
        self.foo()

    def test6(self):
        p0 = uniform(0, 1)
        self.expr = as_apply(dict(p0=p0, p1=normal(p0, 1)))
        self.wanted = [
            [('p0', [0], [0.69646918559786164]), ('p1', [0], [-0.25562802126346051])],
            [('p0', [1], [0.55131476908289123]), ('p1', [1], [-0.19412629039976703])],
            [('p0', [2], [0.71946896978556307]), ('p1', [2], [1.0415750381251847])],
            [('p0', [3], [0.68482973858486329]), ('p1', [3], [0.63331201764547818])],
            [('p0', [4], [0.48093190148436094]), ('p1', [4], [-1.1383681635523848])]]
        self.foo()

    def test7(self):
        p0 = uniform(0, 1)
        p1 = normal(0, 1)
        self.expr = as_apply(dict(
            p0=p0,
            p1=p1,
            p2=one_of(1, p0),
            p3=one_of(2, p1, uniform(2, 3))))
        self.n_randints = 2
        self.wanted = [
                [
                    ('p0', [0], [0.71295532052322719]),
                    ('p1', [0], [0.28297849805199204]),
                    ('p2.randint', [0], [0]),
                    ('p3.arg:2', [0], [2.719468969785563]),
                    ('p3.randint', [0], [2])],
                [
                    ('p0', [1], [0.78002776191207912]),
                    ('p1', [1], [-1.506294713918092]),
                    ('p2.randint', [1], [1]),
                    ('p3.arg:2', [], []),
                    ('p3.randint', [1], [1])],
                [
                    ('p0', [2], [0.57969429702261011]),
                    ('p1', [2], [1.6796003743035337]),
                    ('p2.randint', [2], [0]),
                    ('p3.arg:2', [], []),
                    ('p3.randint', [2], [1])],
                [
                    ('p0', [3], [0.43857224467962441]),
                    ('p1', [3], [-1.3058031267484451]),
                    ('p2.randint', [3], [1]),
                    ('p3.arg:2', [], []),
                    ('p3.randint', [3], [1])],
                [
                    ('p0', [4], [0.39804425533043142]),
                    ('p1', [4], [-0.91948540682140967]),
                    ('p2.randint', [4], [0]),
                    ('p3.arg:2', [], []),
                    ('p3.randint', [4], [0])]]
        self.foo()

