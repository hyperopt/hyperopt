"""
Progress is reported using context managers.

A progress context manager takes an `initial` and a `total` argument
and should yield an object with an `update(n)` method.
"""

import contextlib

from tqdm import tqdm
from .std_out_err_redirect_tqdm import std_out_err_redirect_tqdm


@contextlib.contextmanager
def tqdm_progress_callback(initial, total):
    with std_out_err_redirect_tqdm() as wrapped_stdout, tqdm(
        total=total,
        file=wrapped_stdout,
        postfix={"best loss": "?"},
        disable=False,
        dynamic_ncols=True,
        unit="trial",
        initial=initial,
    ) as pbar:
        yield pbar


@contextlib.contextmanager
def no_progress_callback(initial, total):
    class NoProgressContext:
        def update(self, n):
            pass

    yield NoProgressContext()


default_callback = tqdm_progress_callback
"""Use tqdm for progress by default"""
