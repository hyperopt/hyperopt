import numpy as np
import pytest
from scipy.stats import norm

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, hp
from hyperopt.base import JOB_STATE_DONE, Domain


def _trials_with_results(results):
    trials = Trials()
    tids = list(range(len(results)))
    documents = trials.new_trial_docs(
        tids,
        [None] * len(results),
        results,
        [{"tid": tid, "cmd": None, "idxs": {}, "vals": {}} for tid in tids],
    )
    for document in documents:
        document["state"] = JOB_STATE_DONE
    trials.insert_trial_docs(documents)
    trials.refresh()
    return trials


@pytest.mark.parametrize("best_variance", [0.0, 1e-6, 1.0])
@pytest.mark.parametrize("with_domain", [False, True])
def test_average_best_error_accounts_for_uncertain_competing_trials(
    best_variance, with_domain
):
    trials = _trials_with_results(
        [
            {
                "status": STATUS_OK,
                "loss": 0.0,
                "loss_variance": best_variance,
                "true_loss": 4.0,
            },
            {
                "status": STATUS_OK,
                "loss": 1.0,
                "loss_variance": 100.0,
                "true_loss": 10.0,
            },
            {"status": STATUS_FAIL},
        ]
    )
    domain = Domain(lambda x: x, hp.uniform("x", 0, 1)) if with_domain else None
    probability_competitor_is_best = norm.cdf(-1 / np.sqrt(best_variance + 100))
    expected = 4 + 6 * probability_competitor_is_best

    actual = trials.average_best_error(domain)

    # The helper uses 1,000 samples; allow Monte Carlo error around the analytic result.
    assert actual == pytest.approx(expected, abs=0.3)


@pytest.mark.parametrize("with_domain", [False, True])
def test_average_best_error_keeps_the_deterministic_best_trial(with_domain):
    trials = _trials_with_results(
        [
            {"status": STATUS_OK, "loss": 2.0, "true_loss": 20.0},
            {"status": STATUS_OK, "loss": 1.0, "true_loss": 10.0},
            {"status": STATUS_FAIL},
        ]
    )
    domain = Domain(lambda x: x, hp.uniform("x", 0, 1)) if with_domain else None

    assert trials.average_best_error(domain) == 10.0
