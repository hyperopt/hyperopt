import copy
import unittest
import numpy as np
import bson

from hyperopt.pyll import scope

from hyperopt.base import JOB_STATE_DONE, JOB_STATE_NEW
from hyperopt.base import TRIAL_KEYS
from hyperopt.base import TRIAL_MISC_KEYS
from hyperopt.base import InvalidTrial
from hyperopt.base import miscs_to_idxs_vals
from hyperopt.base import SONify
from hyperopt.base import STATUS_OK
from hyperopt.base import Trials
from hyperopt.base import trials_from_docs

from hyperopt.exceptions import AllTrialsFailed

uniform = scope.uniform
normal = scope.normal
one_of = scope.one_of


def ok_trial(tid, *args, **kwargs):
    return dict(
        tid=tid,
        result={"status": "algo, ok"},
        spec={"a": 1, "foo": (args, kwargs)},
        misc={
            "tid": tid,
            "cmd": ("some cmd",),
            "idxs": {"z": [tid]},
            "vals": {"z": [1]},
        },
        extra="extra",  # -- more stuff here is ok
        owner=None,
        state=JOB_STATE_NEW,
        version=0,
        book_time=None,
        refresh_time=None,
        exp_key=None,
    )


def create_fake_trial(tid, loss=None, status=STATUS_OK, state=JOB_STATE_DONE):
    return dict(
        tid=tid,
        result={"status": status, "loss": loss}
        if loss is not None
        else {"status": status},
        spec={"a": 1},
        misc={
            "tid": tid,
            "cmd": ("some cmd",),
            "idxs": {"z": [tid]},
            "vals": {"z": [1]},
        },
        extra="extra",  # -- more stuff here is ok
        owner=None,
        state=state,
        version=0,
        book_time=None,
        refresh_time=None,
        exp_key=None,
    )


class Suggest_API:
    """
    Run some generic sanity-checks of a suggest algorithm to make sure that
    it respects the semantics expected by e.g. fmin.

    Use it like this:

        TestRand = Suggest_API.make_test_class(rand.suggest, 'TestRand')

    """

    @classmethod
    def make_tst_class(cls, suggest, domain, name):
        class Tester(unittest.TestCase, cls):
            def suggest(self, *args, **kwargs):
                print(args, kwargs)
                return suggest(*args, **kwargs)

            def setUp(self):
                self.domain = domain

        Tester.__name__ = name
        return Tester

    seed_randomizes = True

    def idxs_vals_from_ids(self, ids, seed):
        docs = self.suggest(ids, self.domain, Trials(), seed)
        trials = trials_from_docs(docs)
        idxs, vals = miscs_to_idxs_vals(trials.miscs)
        return idxs, vals

    def test_arbitrary_ids(self):
        # -- suggest implementations should work for arbitrary ID
        #    values (possibly assuming they are hashable), and the
        #    ID values should have no effect on the return values.
        ids_1 = [-2, 0, 7, "a", "007", 66, "a3", "899", 23, 2333]
        ids_2 = ["a", "b", "c", "d", 1, 2, 3, 0.1, 0.2, 0.3]
        idxs_1, vals_1 = self.idxs_vals_from_ids(ids=ids_1, seed=45)
        idxs_2, vals_2 = self.idxs_vals_from_ids(ids=ids_2, seed=45)
        all_ids_1 = set()
        for var, ids in list(idxs_1.items()):
            all_ids_1.update(ids)
        all_ids_2 = set()
        for var, ids in list(idxs_2.items()):
            all_ids_2.update(ids)
        self.assertEqual(all_ids_1, set(ids_1))
        self.assertEqual(all_ids_2, set(ids_2))
        self.assertEqual(vals_1, vals_2)

    def test_seed_randomizes(self):
        #
        # suggest() algorithms can be either stochastic (e.g. random search)
        # or deterministic (e.g. grid search).  If an suggest implementation
        # is stochastic, then changing the seed argument should change the
        # return value.
        #
        if not self.seed_randomizes:
            return

        # -- sample 20 points to make sure we get some differences even
        #    for small search spaces (chance of false failure is 1/million).
        idxs_1, vals_1 = self.idxs_vals_from_ids(ids=list(range(20)), seed=45)
        idxs_2, vals_2 = self.idxs_vals_from_ids(ids=list(range(20)), seed=46)
        self.assertNotEqual((idxs_1, vals_1), (idxs_2, vals_2))


class TestTrials(unittest.TestCase):
    def setUp(self):
        self.trials = Trials()

    def test_valid(self):
        trials = self.trials
        f = trials.insert_trial_doc
        fine = ok_trial("ID", 1, 2, 3)

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
            del rval["misc"][key]
            return rval

        for key in TRIAL_MISC_KEYS:
            self.assertRaises(InvalidTrial, f, knockout2(key))

    def test_insert_sync(self):
        trials = self.trials
        assert len(trials) == 0
        trials.insert_trial_doc(ok_trial("a", 8))
        assert len(trials) == 0
        trials.insert_trial_doc(ok_trial(5, a=1, b=3))
        assert len(trials) == 0
        trials.insert_trial_docs([ok_trial(tid=4, a=2, b=3), ok_trial(tid=9, a=4, b=3)])
        assert len(trials) == 0
        trials.refresh()

        assert len(trials) == 4, len(trials)
        assert len(trials) == len(trials.specs)
        assert len(trials) == len(trials.results)
        assert len(trials) == len(trials.miscs)

        trials.insert_trial_docs(
            trials.new_trial_docs(
                ["id0", "id1"],
                [dict(a=1), dict(a=2)],
                [dict(status="new"), dict(status="new")],
                [
                    dict(tid="id0", idxs={}, vals={}, cmd=None),
                    dict(tid="id1", idxs={}, vals={}, cmd=None),
                ],
            )
        )

        assert len(trials) == 4
        assert len(trials) == len(trials.specs)
        assert len(trials) == len(trials.results)
        assert len(trials) == len(trials.miscs)

        trials.refresh()
        assert len(trials) == 6
        assert len(trials) == len(trials.specs)
        assert len(trials) == len(trials.results)
        assert len(trials) == len(trials.miscs)

    def test_best_trial(self):
        trials = self.trials
        assert len(trials) == 0
        # It should throw a reasonable error when no valid trials exist.
        trials.insert_trial_doc(create_fake_trial(0, loss=np.NaN))
        trials.refresh()
        with self.assertRaises(AllTrialsFailed):
            assert trials.best_trial is None

        # It should work even with some trials with NaN losses.
        trials.insert_trial_doc(create_fake_trial(1, loss=1.0))
        trials.insert_trial_doc(create_fake_trial(2, loss=np.NaN))
        trials.insert_trial_doc(create_fake_trial(3, loss=0.5))
        trials.refresh()

        best_trial = trials.best_trial
        self.assertEquals(best_trial["tid"], 3)


class TestSONify(unittest.TestCase):
    def SONify(self, foo):
        rval = SONify(foo)
        assert bson.BSON.encode(dict(a=rval))
        return rval

    def test_int(self):
        assert self.SONify(1) == 1

    def test_float(self):
        assert self.SONify(1.1) == 1.1

    def test_np_1d_int(self):
        assert np.all(self.SONify(np.asarray([1, 2, 3])) == [1, 2, 3])

    def test_np_1d_float(self):
        assert np.all(self.SONify(np.asarray([1, 2, 3.4])) == [1, 2, 3.4])

    def test_np_1d_str(self):
        assert np.all(self.SONify(np.asarray(["a", "b", "ccc"])) == ["a", "b", "ccc"])

    def test_np_2d_int(self):
        assert np.all(self.SONify(np.asarray([[1, 2], [3, 4]])) == [[1, 2], [3, 4]])

    def test_np_2d_float(self):
        assert np.all(self.SONify(np.asarray([[1, 2], [3, 4.5]])) == [[1, 2], [3, 4.5]])

    def test_nested_w_bool(self):
        thing = dict(a=1, b="2", c=True, d=False, e=int(3), f=[1])
        assert thing == SONify(thing)
