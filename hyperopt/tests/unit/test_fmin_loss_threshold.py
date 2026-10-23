import numpy as np
import pytest

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, rand


@pytest.mark.parametrize(
    "failed_result",
    [
        {"status": STATUS_FAIL},
        {"status": STATUS_FAIL, "loss": None},
        {"status": STATUS_FAIL, "loss": float("nan")},
        {"status": STATUS_FAIL, "loss": -100},
    ],
    ids=["missing", "none", "nan", "numeric"],
)
@pytest.mark.parametrize("loss_threshold", [None, 0])
def test_loss_threshold_ignores_failed_results(failed_result, loss_threshold):
    trials = Trials()
    results = iter(
        [
            {"status": STATUS_OK, "loss": 4},
            failed_result,
            {"status": STATUS_OK, "loss": -1},
            {"status": STATUS_OK, "loss": 2},
        ]
    )

    best = fmin(
        fn=lambda x: next(results),
        space=hp.uniform("x", -5, 5),
        algo=rand.suggest,
        max_evals=4,
        trials=trials,
        loss_threshold=loss_threshold,
        rstate=np.random.default_rng(0),
        show_progressbar=False,
    )

    assert len(trials) == (4 if loss_threshold is None else 3)
    assert trials.best_trial["result"]["loss"] == -1
    assert best == {"x": trials.trials[2]["misc"]["vals"]["x"][0]}
