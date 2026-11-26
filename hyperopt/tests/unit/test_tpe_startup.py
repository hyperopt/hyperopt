from functools import partial

import numpy as np
import pytest

from hyperopt import JOB_STATE_DONE, STATUS_OK, Domain, Trials, fmin, hp, rand, tpe


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize(
    "history_size,startup", [(0, 20), (3, 20), (19, 20), (20, 20), (3, 0), (0, 0)]
)
def test_startup_only_builds_posterior_when_needed(
    monkeypatch, conditional, history_size, startup
):
    space = {"x": hp.uniform("x", -1, 1)}
    if conditional:
        space["branch"] = hp.choice(
            "branch", [hp.uniform("left", -2, 0), hp.uniform("right", 0, 2)]
        )
    domain = Domain(lambda value: value, space)
    trials = Trials()
    history = rand.suggest(list(range(history_size)), domain, trials, 42)
    for doc in history:
        doc["state"] = JOB_STATE_DONE
        doc["result"] = {"status": STATUS_OK, "loss": float(doc["tid"])}
    trials.insert_trial_docs(history)
    trials.refresh()

    original_build = tpe.build_posterior_wrapper
    builds = []

    def tracked_build(*args, **kwargs):
        builds.append(True)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(tpe, "build_posterior_wrapper", tracked_build)
    actual = tpe.suggest(
        [history_size], domain, trials, 17, n_startup_jobs=startup, verbose=False
    )
    assert len(actual) == 1
    if history_size < startup:
        expected = rand.suggest([history_size], domain, trials, 17)
        assert actual[0]["misc"] == expected[0]["misc"]
        assert builds == []
    else:
        assert len(builds) == 1
        assert set(actual[0]["misc"]["vals"]) == set(domain.params)


def test_fmin_builds_posterior_after_startup(monkeypatch):
    original_build = tpe.build_posterior_wrapper
    builds = []

    def tracked_build(*args, **kwargs):
        builds.append(True)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(tpe, "build_posterior_wrapper", tracked_build)
    trials = Trials()
    fmin(
        lambda value: value**2,
        hp.uniform("x", -1, 1),
        algo=partial(tpe.suggest, n_startup_jobs=3, verbose=False),
        max_evals=5,
        trials=trials,
        rstate=np.random.default_rng(7),
        show_progressbar=False,
    )
    assert len(trials) == 5
    assert len(builds) == 2
