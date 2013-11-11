import unittest
import numpy as np
import nose.tools

from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval
from hyperopt.base import JOB_STATE_ERROR


def test_quadratic1_rand():
    trials = Trials()

    argmin = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp.uniform('x', -5, 5),
            algo=rand.suggest,
            max_evals=500,
            trials=trials)

    assert len(trials) == 500
    assert abs(argmin['x'] - 3.0) < .25


def test_quadratic1_tpe():
    trials = Trials()

    argmin = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp.uniform('x', -5, 5),
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

    assert len(trials) == 50, len(trials)
    assert abs(argmin['x'] - 3.0) < .25, argmin


def test_quadratic1_anneal():
    trials = Trials()
    import hyperopt.anneal

    N = 30
    def fn(x):
        return (x - 3) ** 2

    argmin = fmin(
            fn=fn,
            space=hp.uniform('x', -5, 5),
            algo=hyperopt.anneal.suggest,
            max_evals=N,
            trials=trials)

    print argmin

    assert len(trials) == N
    assert abs(argmin['x'] - 3.0) < .25


@nose.tools.raises(exceptions.DuplicateLabel)
def test_duplicate_label_is_error():
    trials = Trials()

    def fn(xy):
        x, y = xy
        return x ** 2 + y ** 2

    fmin(fn=fn,
            space=[
                hp.uniform('x', -5, 5),
                hp.uniform('x', -5, 5),
                ],
            algo=rand.suggest,
            max_evals=500,
            trials=trials)


def test_space_eval():
    space = hp.choice('a',
        [
            ('case 1', 1 + hp.lognormal('c1', 0, 1)),
            ('case 2', hp.uniform('c2', -10, 10))
        ])

    assert space_eval(space, {'a': 0, 'c1': 1.0}) == ('case 1', 2.0)
    assert space_eval(space, {'a': 1, 'c2': 3.5}) == ('case 2', 3.5)

def test_set_fmin_rstate():
    lossfn = lambda x: (x - 3) ** 2
    trials_seed0 = Trials()
    argmin_seed0 = fmin(
            fn=lossfn,
            space=hp.uniform('x', -5, 5),
            algo=rand.suggest,
            max_evals=1,
            trials=trials_seed0,
            rstate=np.random.RandomState(0))
    assert len(trials_seed0) == 1
    trials_seed1 = Trials()
    argmin_seed1 = fmin(
            fn=lossfn,
            space=hp.uniform('x', -5, 5),
            algo=rand.suggest,
            max_evals=1,
            trials=trials_seed1,
            rstate=np.random.RandomState(1))
    assert len(trials_seed1) == 1
    assert argmin_seed0 != argmin_seed1


class TestFmin(unittest.TestCase):
    class BanditE(Exception):
        #XXX also test Bandit.exceptions mechanism that actually catches this
        pass

    def eval_fn(self, space):
        raise TestFmin.BanditE()

    def setUp(self):
        self.trials = Trials()

    def test_catch_eval_exceptions_True(self):

        # -- should go to max_evals, catching all exceptions, so all jobs
        #    should have JOB_STATE_ERROR
        fmin(self.eval_fn,
             space=hp.uniform('x', 0, 1),
             algo=rand.suggest,
             trials=self.trials,
             max_evals=2,
             catch_eval_exceptions=True,
             return_argmin=False,)
        trials = self.trials
        assert len(trials) == 0
        assert len(trials._dynamic_trials) == 2
        assert trials._dynamic_trials[0]['state'] == JOB_STATE_ERROR
        assert trials._dynamic_trials[0]['misc']['error'] != None
        assert trials._dynamic_trials[1]['state'] == JOB_STATE_ERROR
        assert trials._dynamic_trials[1]['misc']['error'] != None

    def test_catch_eval_exceptions_False(self):
        with self.assertRaises(TestFmin.BanditE):
            fmin(self.eval_fn,
                 space=hp.uniform('x', 0, 1),
                 algo=rand.suggest,
                 trials=self.trials,
                 max_evals=2,
                 catch_eval_exceptions=False)
        print len(self.trials)
        assert len(self.trials) == 0
        assert len(self.trials._dynamic_trials) == 1


