from copy import deepcopy

import numpy as np
import pytest

from hyperopt import Trials, anneal, hp, rand, tpe
from hyperopt.base import (
    JOB_STATE_DONE,
    STATUS_FAIL,
    STATUS_NEW,
    STATUS_OK,
    STATUS_RUNNING,
    STATUS_SUSPENDED,
    Domain,
)


def make_history(status, loss):
    domain = Domain(lambda x: x, hp.uniform("x", -10, 10))
    trials = Trials()
    docs = rand.suggest(list(range(20)), domain, trials, 7)
    for i, doc in enumerate(docs):
        doc["state"] = JOB_STATE_DONE
        doc["misc"]["vals"]["x"] = [-9 + i * 0.05 if i < 10 else 7 + (i - 10) * 0.05]
        doc["result"] = {
            "status": status if i < 10 else STATUS_OK,
            "loss": loss if i < 10 else i - 10,
        }
    trials.insert_trial_docs(docs)
    trials.refresh()
    return domain, trials


def suggest_value(algorithm, domain, trials):
    if algorithm is tpe:
        kwargs = {"n_startup_jobs": 0, "verbose": False}
    else:
        kwargs = {"avg_best_idx": 1}
    suggestion = algorithm.suggest([20], domain, trials, 11, **kwargs)
    return suggestion[0]["misc"]["vals"]["x"]


@pytest.mark.parametrize("algorithm", [tpe, anneal])
@pytest.mark.parametrize(
    "status", [STATUS_FAIL, STATUS_NEW, STATUS_RUNNING, STATUS_SUSPENDED]
)
def test_unsuccessful_loss_does_not_change_the_next_suggestion(algorithm, status):
    domain, trials = make_history(status, -1e6)
    reference = deepcopy(trials)
    for doc in reference.trials[:10]:
        doc["result"].pop("loss")

    actual = suggest_value(algorithm, domain, trials)
    expected = suggest_value(algorithm, domain, reference)

    np.testing.assert_array_equal(actual, expected)
    assert all(doc["result"]["loss"] == -1e6 for doc in trials.trials[:10])


@pytest.mark.parametrize("algorithm", [tpe, anneal])
def test_successful_loss_still_changes_the_next_suggestion(algorithm):
    domain, successful = make_history(STATUS_OK, -1e6)
    _, failed = make_history(STATUS_FAIL, None)

    assert suggest_value(algorithm, domain, successful) != suggest_value(
        algorithm, domain, failed
    )
