import numpy as np
import pytest

from hyperopt import Trials, base, fmin, hp, rand
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import FMinIter


def make_iterator(stop, max_evals=20):
    trials = Trials()
    domain = base.Domain(lambda value: 1.0, hp.uniform("x", 0, 1))
    iterator = FMinIter(
        rand.suggest,
        domain,
        trials,
        np.random.default_rng(5),
        max_evals=max_evals,
        show_progressbar=False,
        early_stop_fn=stop,
    )
    return iterator, trials


@pytest.mark.parametrize("stop_after", [1, 3, 4])
def test_iterator_calls_early_stop_once_per_evaluation(stop_after):
    calls = []

    def stop(trials, count=0):
        calls.append((len(trials), count))
        return count + 1 >= stop_after, [count + 1]

    iterator, trials = make_iterator(stop)

    for _ in iterator:
        pass

    assert len(trials) == stop_after
    assert calls == [(i + 1, i) for i in range(stop_after)]
    assert iterator.early_stop_args == [stop_after]


def test_iterator_honors_the_original_stop_decision_without_rechecking():
    calls = []

    def stop(trials):
        calls.append(len(trials))
        return len(calls) == 1, []

    iterator, trials = make_iterator(stop)

    with pytest.raises(StopIteration):
        next(iterator)

    assert len(trials) == 1
    assert calls == [1]


def test_stopped_iterator_does_not_schedule_more_work_or_call_the_callback_again():
    calls = []

    def stop(trials):
        calls.append(len(trials))
        return True, []

    iterator, trials = make_iterator(stop)
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(iterator)

    assert len(trials) == 1
    assert calls == [1]


@pytest.mark.parametrize("patience", [4, 7])
def test_iterator_and_fmin_use_the_same_no_progress_patience(patience):
    iterator, trials = make_iterator(no_progress_loss(patience))
    for _ in iterator:
        pass
    reference_trials = Trials()
    fmin(
        lambda value: 1.0,
        hp.uniform("x", 0, 1),
        algo=rand.suggest,
        trials=reference_trials,
        rstate=np.random.default_rng(5),
        max_evals=20,
        show_progressbar=False,
        early_stop_fn=no_progress_loss(patience),
    )

    assert len(trials) == len(reference_trials) == patience
    assert trials.vals == reference_trials.vals


def test_iterator_keeps_nonstopping_callback_state():
    calls = []

    def stop(trials, count=0):
        calls.append(len(trials))
        return False, [count + 1]

    iterator, trials = make_iterator(stop, max_evals=3)
    for _ in iterator:
        pass

    assert len(trials) == 3
    assert calls == [1, 2, 3]
    assert iterator.early_stop_args == [3]


def test_an_explicit_run_can_resume_after_an_early_stop():
    def stop(trials, count=0):
        return count == 0, [count + 1]

    iterator, trials = make_iterator(stop)
    with pytest.raises(StopIteration):
        next(iterator)

    iterator.run(1)

    assert len(trials) == 2
    assert iterator.early_stop_args == [2]


def test_iterator_without_early_stopping_keeps_its_evaluation_limit():
    iterator, trials = make_iterator(None, max_evals=3)

    for _ in iterator:
        pass

    assert len(trials) == 3
