import numpy as np
import pytest

from hyperopt import JOB_STATE_DONE, STATUS_OK, Domain, Trials, fmin, hp, rand
from hyperopt.base import trials_from_docs


def imported_trials(ids):
    domain = Domain(lambda x: x**2, hp.uniform("x", -1, 1))
    docs = rand.suggest(ids, domain, Trials(), 42)
    for doc in docs:
        doc["state"] = JOB_STATE_DONE
        doc["result"] = {"status": STATUS_OK, "loss": doc["misc"]["vals"]["x"][0] ** 2}
    return trials_from_docs(docs)


@pytest.mark.parametrize(
    "ids", [[], [0, 1, 2], [0, 2], [2], [1, 3, 5], [3, "saved"], [-1, 2], [0, 1000]]
)
def test_new_ids_never_overlap_imported_or_reserved_ids(ids):
    trials = imported_trials(ids)
    first = trials.new_trial_ids(4)
    second = trials.new_trial_ids(4)
    assert len(first) == len(second) == 4
    assert len(set(first + second)) == 8
    assert set(ids).isdisjoint(first + second)


def test_contiguous_ids_keep_existing_sequence():
    trials = imported_trials([0, 1])
    assert trials.new_trial_ids(0) == []
    assert trials.new_trial_ids(3) == [2, 3, 4]
    assert trials.new_trial_ids(2) == [5, 6]


def test_views_share_new_id_reservations():
    trials = imported_trials([0, 2])
    view = trials.view()
    ids = trials.new_trial_ids(3) + view.new_trial_ids(3)
    assert len(set(ids)) == 6
    assert set([0, 2]).isdisjoint(ids)


@pytest.mark.parametrize("queue_length", [1, 3])
def test_fmin_continues_sparse_imported_history_without_duplicate_ids(queue_length):
    trials = imported_trials([0, 2])
    fmin(
        lambda x: x**2,
        hp.uniform("x", -1, 1),
        algo=rand.suggest,
        trials=trials,
        max_evals=7,
        max_queue_len=queue_length,
        rstate=np.random.default_rng(12),
        show_progressbar=False,
        verbose=False,
    )
    assert len(trials) == len(set(trials.tids)) == 7
    for doc in trials:
        assert doc["tid"] == doc["misc"]["tid"]
        assert doc["misc"]["idxs"]["x"] == [doc["tid"]]
