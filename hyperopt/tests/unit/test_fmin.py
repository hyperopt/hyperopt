import unittest
import numpy as np
from timeit import default_timer as timer
import time
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate
import pytest

from hyperopt import (
    fmin,
    rand,
    tpe,
    hp,
    Trials,
    exceptions,
    space_eval,
    STATUS_FAIL,
    STATUS_OK,
)
from hyperopt.base import JOB_STATE_ERROR


def test_quadratic1_rand():
    trials = Trials()

    argmin = fmin(
        fn=lambda x: (x - 3) ** 2,
        space=hp.uniform("x", -5, 5),
        algo=rand.suggest,
        max_evals=500,
        trials=trials,
        rstate=np.random.default_rng(np.random.PCG64(123)),
    )

    assert len(trials) == 500
    assert abs(argmin["x"] - 3.0) < 0.25


def test_quadratic1_tpe(trials=Trials()):

    argmin = fmin(
        fn=lambda x: (x - 3) ** 2,
        space=hp.uniform("x", -5, 5),
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(np.random.PCG64(123)),
    )

    assert len(trials) == 50, len(trials)
    assert abs(argmin["x"] - 3.0) < 0.25, argmin


def test_quadratic1_anneal():
    trials = Trials()
    import hyperopt.anneal

    N = 30

    def fn(x):
        return (x - 3) ** 2

    argmin = fmin(
        fn=fn,
        space=hp.uniform("x", -5, 5),
        algo=hyperopt.anneal.suggest,
        max_evals=N,
        trials=trials,
        rstate=np.random.default_rng(np.random.PCG64(123)),
    )

    print(argmin)

    assert len(trials) == N
    assert abs(argmin["x"] - 3.0) < 0.25


def test_duplicate_label_is_error():
    trials = Trials()

    def fn(xy):
        x, y = xy
        return x ** 2 + y ** 2

    with pytest.raises(exceptions.DuplicateLabel):
        fmin(
            fn=fn,
            space=[hp.uniform("x", -5, 5), hp.uniform("x", -5, 5)],
            algo=rand.suggest,
            max_evals=500,
            trials=trials,
            rstate=np.random.default_rng(0),
        )


def test_space_eval():
    space = hp.choice(
        "a",
        [
            ("case 1", 1 + hp.lognormal("c1", 0, 1)),
            ("case 2", hp.uniform("c2", -10, 10)),
        ],
    )

    assert space_eval(space, {"a": 0, "c1": 1.0}) == ("case 1", 2.0)
    assert space_eval(space, {"a": 1, "c2": 3.5}) == ("case 2", 3.5)


def test_set_fmin_rstate():
    def lossfn(x):
        return (x - 3) ** 2

    trials_seed0 = Trials()
    argmin_seed0 = fmin(
        fn=lossfn,
        space=hp.uniform("x", -5, 5),
        algo=rand.suggest,
        max_evals=1,
        trials=trials_seed0,
        rstate=np.random.default_rng(0),
    )
    assert len(trials_seed0) == 1
    trials_seed1 = Trials()
    argmin_seed1 = fmin(
        fn=lossfn,
        space=hp.uniform("x", -5, 5),
        algo=rand.suggest,
        max_evals=1,
        trials=trials_seed1,
        rstate=np.random.default_rng(1),
    )
    assert len(trials_seed1) == 1
    assert argmin_seed0 != argmin_seed1


def test_fmin_return_argmin():
    def fn(x):
        return x

    space = hp.choice("x", [100, 5, 10])
    # With return_argmin=False it should return the
    # best parameter values
    best_parameter = fmin(
        fn=fn,
        space=space,
        max_evals=10,
        algo=rand.suggest,
        return_argmin=False,
        rstate=np.random.default_rng(0),
    )
    assert best_parameter == 5

    # With return_argmin=True it should return the
    # optimal point in the sample space
    best_args = fmin(
        fn=fn,
        space=space,
        max_evals=10,
        algo=rand.suggest,
        return_argmin=True,
        rstate=np.random.default_rng(0),
    )
    assert best_args["x"] == 1


class TestFmin(unittest.TestCase):
    class SomeError(Exception):
        # XXX also test domain.exceptions mechanism that actually catches this
        pass

    def eval_fn(self, space):
        raise TestFmin.SomeError()

    def setUp(self):
        self.trials = Trials()

    def test_catch_eval_exceptions_True(self):

        # -- should go to max_evals, catching all exceptions, so all jobs
        #    should have JOB_STATE_ERROR
        fmin(
            self.eval_fn,
            space=hp.uniform("x", 0, 1),
            algo=rand.suggest,
            trials=self.trials,
            max_evals=2,
            catch_eval_exceptions=True,
            return_argmin=False,
        )
        trials = self.trials
        assert len(trials) == 0
        assert len(trials._dynamic_trials) == 2
        assert trials._dynamic_trials[0]["state"] == JOB_STATE_ERROR
        assert trials._dynamic_trials[0]["misc"]["error"] != None
        assert trials._dynamic_trials[1]["state"] == JOB_STATE_ERROR
        assert trials._dynamic_trials[1]["misc"]["error"] != None

    def test_catch_eval_exceptions_False(self):
        with self.assertRaises(TestFmin.SomeError):
            fmin(
                self.eval_fn,
                space=hp.uniform("x", 0, 1),
                algo=rand.suggest,
                trials=self.trials,
                max_evals=2,
                catch_eval_exceptions=False,
            )
        print(len(self.trials))
        assert len(self.trials) == 0
        assert len(self.trials._dynamic_trials) == 1


def test_status_fail_tpe():
    trials = Trials()

    argmin = fmin(
        fn=lambda x: (
            {"loss": (x - 3) ** 2, "status": STATUS_OK}
            if (x < 0)
            else {"status": STATUS_FAIL}
        ),
        space=hp.uniform("x", -5, 5),
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )

    assert len(trials) == 50, len(trials)
    assert argmin["x"] < 0, argmin
    assert "loss" in trials.best_trial["result"], "loss" in trials.best_trial["result"]
    assert trials.best_trial["result"]["loss"] >= 9, trials.best_trial["result"]["loss"]


class TestGenerateTrialsToCalculate(unittest.TestCase):
    def test_generate_trials_to_calculate(self):
        points = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
        best = fmin(
            fn=lambda space: space["x"] ** 2 + space["y"] ** 2,
            space={"x": hp.uniform("x", -10, 10), "y": hp.uniform("y", -10, 10)},
            algo=tpe.suggest,
            max_evals=10,
            points_to_evaluate=points,
        )
        assert best["x"] == 0.0
        assert best["y"] == 0.0


def test_timeout():
    def fn(x):
        return [time.sleep(1), x][1]

    space = hp.choice("x", range(20))

    start_time_1 = timer()
    fmin(
        fn=fn,
        space=space,
        max_evals=10,
        timeout=1,
        algo=rand.suggest,
        return_argmin=False,
        rstate=np.random.default_rng(0),
    )
    end_time_1 = timer()
    assert (end_time_1 - start_time_1) < 2
    assert (end_time_1 - start_time_1) > 0.9

    start_time_5 = timer()
    fmin(
        fn=fn,
        space=space,
        max_evals=10,
        timeout=5,
        algo=rand.suggest,
        return_argmin=False,
        rstate=np.random.default_rng(0),
    )
    end_time_5 = timer()
    assert (end_time_5 - start_time_5) < 6
    assert (end_time_5 - start_time_5) > 4.9


def test_invalid_timeout():
    def fn(x):
        return [time.sleep(1), x][1]

    space = hp.choice("x", range(20))

    for wrong_timeout in [-1, True]:
        expected_message = "The timeout argument should be None or a positive value. Given value: {m}".format(
            m=wrong_timeout
        )
        try:
            fmin(
                fn=fn,
                space=space,
                max_evals=10,
                timeout=wrong_timeout,
                algo=rand.suggest,
                return_argmin=False,
                rstate=np.random.default_rng(0),
            )
        except Exception as e:
            assert str(e) == expected_message


def test_loss_threshold():
    loss_threshold = 0.001
    hypopt_trials = Trials()
    fmin(
        fn=lambda x: x ** 2,
        space=hp.uniform("x", -10, 10),
        loss_threshold=loss_threshold,
        algo=rand.suggest,
        trials=hypopt_trials,
        rstate=np.random.default_rng(0),
    )
    best_loss = hypopt_trials.best_trial["result"]["loss"]
    assert best_loss <= loss_threshold
    assert len(hypopt_trials) > 0


def test_invalid_loss_threshold():
    def fn(x):
        return [time.sleep(1), x][1]

    space = hp.choice("x", range(20))

    for wrong_loss_threshold in ["a", True]:
        expected_message = (
            "The loss_threshold argument should be None "
            "or a numeric value. Given value: {m}".format(m=wrong_loss_threshold)
        )
        try:
            fmin(
                fn=fn,
                space=space,
                max_evals=10,
                loss_threshold=wrong_loss_threshold,
                algo=rand.suggest,
                return_argmin=False,
                rstate=np.random.default_rng(0),
            )
        except Exception as e:
            assert str(e) == expected_message


def test_early_stop():
    trials = Trials()

    # basic stop after 100 trials
    def stop(trial, count=0):
        return count + 1 >= 100, [count + 1]

    fmin(
        fn=lambda x: x,
        space=hp.uniform("x", -5, 5),
        algo=rand.suggest,
        max_evals=500,
        trials=trials,
        early_stop_fn=stop,
    )

    assert len(trials) == 100


def test_early_stop_no_progress_loss():
    trials = generate_trials_to_calculate([{"x": -100}])
    fmin(
        fn=lambda x: x,
        space=hp.uniform("x", -5, 5),
        algo=rand.suggest,
        max_evals=500,
        trials=trials,
        early_stop_fn=no_progress_loss(10),
    )

    assert len(trials) == 10


def test_annotated_params_space():
    def objective(x: hp.uniform("x", -10, 10), y: hp.uniform("y", -10, 10)):
        return (x * y) ** 2

    trials = Trials()
    fmin(objective, space="annotated", algo=tpe.suggest, max_evals=10, trials=trials)

    assert len(trials) == 10


def test_invalid_annotated_params_space():
    def objective(x: hp.uniform("x", -10, 10), y: float):
        return (x * y) ** 2

    with pytest.raises(exceptions.InvalidAnnotatedParameter):
        fmin(objective, space="annotated", algo=tpe.suggest, max_evals=10)
