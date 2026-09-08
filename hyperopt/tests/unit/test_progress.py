import contextlib
import sys

import numpy as np

from hyperopt import Trials, fmin, hp, progress, rand
from hyperopt.fmin import FMinIter
from hyperopt.progress import tqdm_progress_callback


def test_tqdm_progress_callback_restores_stdout():
    real_stdout = sys.stdout
    with tqdm_progress_callback(initial=0, total=100) as ctx:
        assert sys.stdout != real_stdout
        ctx.postfix = "best loss: 4711"
        ctx.update(42)
    assert sys.stdout == real_stdout


def test_fmin_iter_progress_callback_selection():
    @contextlib.contextmanager
    def custom_progress_callback(initial, total):
        yield

    def make_fmin_iter(*, verbose=False, **kwargs):
        return FMinIter(
            algo=None,
            domain=None,
            trials=Trials(),
            rstate=np.random.default_rng(0),
            verbose=verbose,
            **kwargs,
        )

    assert make_fmin_iter().progress_callback is progress.no_progress_callback
    assert make_fmin_iter(verbose=True).progress_callback is progress.default_callback

    cases = [
        (None, progress.no_progress_callback),
        (False, progress.no_progress_callback),
        (True, progress.default_callback),
        (custom_progress_callback, custom_progress_callback),
    ]
    for show_progressbar, expected_callback in cases:
        assert (
            make_fmin_iter(show_progressbar=show_progressbar).progress_callback
            is expected_callback
        )


def test_trials_progress_callback_selection(monkeypatch):
    calls = []
    omitted = object()

    def make_progress_callback(name):
        @contextlib.contextmanager
        def progress_callback(initial, total):
            calls.append(name)
            yield

        return progress_callback

    default_progress_callback = make_progress_callback("default")
    no_progress_callback = make_progress_callback("disabled")
    custom_progress_callback = make_progress_callback("custom")
    monkeypatch.setattr(progress, "default_callback", default_progress_callback)
    monkeypatch.setattr(progress, "no_progress_callback", no_progress_callback)

    def run_trials(*, verbose=False, show_progressbar=omitted):
        calls.clear()
        progress_kwargs = {}
        if show_progressbar is not omitted:
            progress_kwargs["show_progressbar"] = show_progressbar
        Trials().fmin(
            fn=lambda value: value,
            space=hp.uniform("value", 0, 1),
            algo=rand.suggest,
            max_evals=0,
            return_argmin=False,
            verbose=verbose,
            **progress_kwargs,
        )
        return list(calls)

    assert run_trials() == ["disabled"]
    assert run_trials(show_progressbar=None) == ["disabled"]
    assert run_trials(verbose=True, show_progressbar=None) == ["default"]
    assert run_trials(show_progressbar=False) == ["disabled"]
    assert run_trials(show_progressbar=True) == ["default"]
    assert run_trials(show_progressbar=custom_progress_callback) == ["custom"]


def test_custom_progress_callback_contract():
    events = []

    class ProgressContext:
        @property
        def postfix(self):
            raise AssertionError("postfix is write-only in this test")

        @postfix.setter
        def postfix(self, value):
            events.append(("postfix", value))

        def update(self, value):
            events.append(("update", value))

    @contextlib.contextmanager
    def custom_progress_callback(initial, total):
        events.append(("enter", initial, total))
        try:
            yield ProgressContext()
        finally:
            events.append(("exit",))

    fmin(
        fn=lambda value: 0.0,
        space=hp.uniform("value", 0, 1),
        algo=rand.suggest,
        max_evals=1,
        rstate=np.random.default_rng(0),
        verbose=False,
        show_progressbar=custom_progress_callback,
    )

    assert events == [
        ("enter", 0, 1),
        ("postfix", "best loss: 0.0"),
        ("update", 1),
        ("exit",),
    ]


def test_fmin_resolves_progress_before_custom_trials_dispatch():
    class CustomTrials:
        def fmin(self, *args, **kwargs):
            self.show_progressbar = kwargs["show_progressbar"]

    @contextlib.contextmanager
    def custom_progress_callback(initial, total):
        yield

    def dispatch(*, verbose, **kwargs):
        trials = CustomTrials()
        fmin(
            fn=lambda value: value,
            space=hp.uniform("value", 0, 1),
            trials=trials,
            verbose=verbose,
            **kwargs,
        )
        return trials.show_progressbar

    assert dispatch(verbose=False) is False
    assert dispatch(verbose=True) is True
    assert dispatch(verbose=False, show_progressbar=None) is False
    assert dispatch(verbose=False, show_progressbar=False) is False
    assert dispatch(verbose=False, show_progressbar=True) is True
    assert (
        dispatch(verbose=False, show_progressbar=custom_progress_callback)
        is custom_progress_callback
    )
