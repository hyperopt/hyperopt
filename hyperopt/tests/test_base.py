import copy
import unittest
import numpy as np
import nose
import bson

from pyll import as_apply, scope, rec_eval, clone, dfs
uniform = scope.uniform
normal = scope.normal
one_of = scope.one_of

from hyperopt import STATUS_STRINGS
from hyperopt import STATUS_OK
from hyperopt.base import JOB_STATE_NEW
from hyperopt.base import JOB_STATE_ERROR
from hyperopt.base import TRIAL_KEYS
from hyperopt.base import TRIAL_MISC_KEYS
from hyperopt.base import Bandit
from hyperopt.base import Ctrl
from hyperopt.base import Experiment
from hyperopt.base import InvalidTrial
from hyperopt.base import Trials
from hyperopt.base import CoinFlip
from hyperopt.base import Random
from hyperopt.base import SONify
from hyperopt.base import miscs_to_idxs_vals
from hyperopt.vectorize import pretty_names


def ok_trial(tid, *args, **kwargs):
    return dict(
        tid=tid,
        result={'status': 'algo, ok'},
        spec={'a':1, 'foo': (args, kwargs)},
        misc={
            'tid':tid,
            'cmd':("some cmd",),
            'idxs':{'z':[tid]},
            'vals':{'z':[1]}},
        extra='extra', # -- more stuff here is ok
        owner=None,
        state=JOB_STATE_NEW,
        version=0,
        book_time=None,
        refresh_time=None,
        exp_key='my_experiment',
        )


class TestTrials(unittest.TestCase):
    def setUp(self):
        self.trials = Trials()

    def test_valid(self):
        trials = self.trials
        f = trials.insert_trial_doc
        fine = ok_trial('ID', 1, 2, 3)

        # --original runs fine
        f(fine)

        # -- take out each mandatory root key
        def knockout(key):
            rval = copy.deepcopy(fine)
            del rval[key]
            return rval
        for key in TRIAL_KEYS:
            self.assertRaises(InvalidTrial, f, knockout(key))

        # -- take out each mandatory misc key
        def knockout2(key):
            rval = copy.deepcopy(fine)
            del rval['misc'][key]
            return rval
        for key in TRIAL_MISC_KEYS:
            self.assertRaises(InvalidTrial, f, knockout2(key))

    def test_insert_sync(self):
        trials = self.trials
        assert len(trials) == 0
        trials.insert_trial_doc(ok_trial('a', 8))
        assert len(trials) == 0
        trials.insert_trial_doc(ok_trial(5, a=1, b=3))
        assert len(trials) == 0
        trials.insert_trial_docs(
                [ok_trial(tid=4, a=2, b=3), ok_trial(tid=9, a=4, b=3)])
        assert len(trials) == 0
        trials.refresh()

        assert len(trials) == 4, len(trials)
        assert len(trials) == len(trials.specs)
        assert len(trials) == len(trials.results)
        assert len(trials) == len(trials.miscs)

        trials.insert_trial_docs(
                trials.new_trial_docs(
                    ['id0', 'id1'],
                    [dict(a=1), dict(a=2)],
                    [dict(status='new'), dict(status='new')],
                    [dict(tid='id0', idxs={}, vals={}, cmd=None),
                        dict(tid='id1', idxs={}, vals={}, cmd=None)],))

        assert len(trials) == 4
        assert len(trials) == len(trials.specs)
        assert len(trials) == len(trials.results)
        assert len(trials) == len(trials.miscs)

        trials.refresh()
        assert len(trials) == 6
        assert len(trials) == len(trials.specs)
        assert len(trials) == len(trials.results)
        assert len(trials) == len(trials.miscs)


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
        specs, results, miscs = self.algo.suggest([0], [], [], [])
        print specs
        print results
        print miscs
        assert len(specs) == len(results) == len(miscs) == 1
        assert miscs[0]['idxs']['node_4'] == [0]
        idxs, vals = miscs_to_idxs_vals(miscs)
        assert idxs['node_4'] == [0]

    def test_suggest_5(self):
        specs, results, miscs = self.algo.suggest(range(5), [], [], [])
        print specs
        print miscs
        assert len(specs) == len(results) == len(miscs) == 5
        idxs, vals = miscs_to_idxs_vals(miscs)
        print idxs
        print vals
        assert len(idxs) == 1
        assert len(vals) == 1
        assert idxs['node_4'] == range(5)
        assert np.all(vals['node_4'] == [0, 1, 0, 0, 0])

    def test_arbitrary_range(self):
        new_ids = [-2, 0, 7, 'a', '007']
        specs, results, miscs = self.algo.suggest(new_ids, [], [], [])
        idxs, vals = miscs_to_idxs_vals(miscs)
        assert len(specs) == len(results) == len(miscs) == 5
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
        self.ctrl = Ctrl(self.trials)

    def test_run_1(self):
        self.experiment.run(1)
        assert len(self.trials._trials) == 1

    def test_run_1_1_1(self):
        self.experiment.run(1)
        self.experiment.run(1)
        self.experiment.run(1)
        assert len(self.trials._trials) == 3
        print self.trials.miscs
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
        trials = self.trials._trials
        self.output = output = []
        for trial in trials:
            print ''
            tmp = []
            for nid in trial['misc']['idxs']:
                thing = self.algo.doc_coords[nid], trial['misc']['idxs'][nid], trial['misc']['vals'][nid]
                print thing
                tmp.append(thing)
            tmp.sort()
            output.append(tmp)
        print repr(output)
        print repr(self.wanted)
        # -- think of a more robust way to test these things
        #    or, if the sampling style is to be nailed down,
        #    put it in and be sure of it.
        raise nose.SkipTest()
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


class TestSONify(unittest.TestCase):

    def SONify(self, foo):
        rval = SONify(foo)
        assert bson.BSON.encode(dict(a=rval))
        return rval

    def test_int(self):
        assert self.SONify(1) == 1

    def test_float(self):
        assert self.SONify(1.1) == 1.1

    def test_np_int(self):
        assert self.SONify(np.int(1)) == 1

    def test_np_float(self):
        assert self.SONify(np.float(1.1)) == 1.1

    def test_np_1d_int(self):
        assert np.all(self.SONify(np.asarray([1, 2, 3]))
                == [1, 2, 3])

    def test_np_1d_float(self):
        assert np.all(self.SONify(np.asarray([1, 2, 3.4]))
                == [1, 2, 3.4])

    def test_np_1d_str(self):
        assert np.all(self.SONify(np.asarray(['a', 'b', 'ccc']))
                == ['a', 'b', 'ccc'])

    def test_np_2d_int(self):
        assert np.all(self.SONify(np.asarray([[1, 2], [3, 4]]))
                == [[1, 2], [3, 4]])

    def test_np_2d_float(self):
        assert np.all(self.SONify(np.asarray([[1, 2], [3, 4.5]]))
                == [[1, 2], [3, 4.5]])




def test_failure():
    class BanditE(Exception):
        pass
    class DummyBandit(Bandit):
        param_gen = {"a":10}
        def __init__(self):
            super(DummyBandit, self).__init__(self.param_gen)

        def evaluate(self, config, ctrl):
            raise BanditE()

    trials = Trials()
    bandit_algo = Random(DummyBandit())
    exp = Experiment(trials, bandit_algo, async=False)

    exp.run(1)
    trials.refresh()
    assert len(trials) == 1
    assert trials.trials[0]['state'] == JOB_STATE_ERROR
    assert trials.trials[0]['misc']['error'] != None

    exp.catch_bandit_exceptions = False
    nose.tools.assert_raises(BanditE, exp.run, 1)
    trials.refresh()
    # -- judgement call: even passed-through errors should show up in db
    assert len(trials) == 2
    assert trials.trials[1]['state'] == JOB_STATE_ERROR
    assert trials.trials[1]['misc']['error'] != None



