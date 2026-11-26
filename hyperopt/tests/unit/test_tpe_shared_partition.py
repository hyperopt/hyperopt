import numpy as np
import pytest

from hyperopt import JOB_STATE_DONE, STATUS_OK, Domain, Trials, hp, rand, tpe


@pytest.mark.parametrize("dimensions", [1, 4, 8])
@pytest.mark.parametrize("conditional", [False, True])
def test_suggest_partitions_losses_once(monkeypatch, dimensions, conditional):
    space = {str(i): hp.uniform(str(i), -1, 1) for i in range(dimensions)}
    if conditional:
        space["conditional"] = hp.choice(
            "branch", [hp.uniform("left", -1, 1), hp.uniform("right", -1, 1)]
        )
    domain = Domain(lambda x: x, space)
    trials = Trials()
    ntrials = 32
    losses = 100.0 + np.arange(ntrials) * 17 % ntrials
    docs = rand.suggest(list(range(ntrials)), domain, trials, 42)
    for doc, loss in zip(docs, losses):
        doc["state"] = JOB_STATE_DONE
        doc["result"] = {"status": STATUS_OK, "loss": loss}
    trials.insert_trial_docs(docs)
    trials.refresh()
    original_argsort = np.argsort
    loss_sorts = []

    def tracked_argsort(values, *args, **kwargs):
        if np.array_equal(values, losses):
            loss_sorts.append(True)
        return original_argsort(values, *args, **kwargs)

    monkeypatch.setattr(np, "argsort", tracked_argsort)
    suggestion = tpe.suggest([ntrials], domain, trials, 13, verbose=False)
    assert len(suggestion) == 1
    assert set(suggestion[0]["misc"]["vals"]) == set(domain.params)
    assert len(loss_sorts) == 1


@pytest.mark.parametrize(
    "observation_ids,values,loss_ids,losses,gamma,cap,below,above",
    [
        ([8, 3, 5], [0.8, 0.3, 0.5], [3, 5, 8], [3, 1, 2], 1, 25, [0.8, 0.5], [0.3]),
        ([8, 3, 5], [0.8, 0.3, 0.5], [3, 5, 8], [3, 1, 2], 1, 1, [0.5], [0.8, 0.3]),
        ([1, 2, 3], [10, 20, 30], [1, 2, 3], [4, 1, 2], 0, 25, [], [10, 20, 30]),
        ([4, 2], [40, 20], [1, 2, 3, 4], [4, 1, 1, 3], 1, 25, [20], [40]),
        ([2, 2, 7], [20, 21, 70], [1, 2, 3], [3, 1, 2], 1, 25, [20, 21], []),
        ([], [], [1, 2], [2, 1], 1, 25, [], []),
        ([1, 2], [10, 20], [], [], 1, 25, [], []),
        (
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [1, 2, 3, 4],
            [-np.inf, 7, np.inf, np.nan],
            1,
            25,
            [10, 20],
            [30, 40],
        ),
    ],
)
def test_split_preserves_observation_order_and_membership(
    observation_ids, values, loss_ids, losses, gamma, cap, below, above
):
    actual_below, actual_above = tpe.ap_split_trials(
        observation_ids, values, loss_ids, losses, gamma, gamma_cap=cap
    )
    np.testing.assert_array_equal(actual_below, below)
    np.testing.assert_array_equal(actual_above, above)
