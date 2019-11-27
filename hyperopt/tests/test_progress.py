from __future__ import print_function
from builtins import range
from unittest.mock import patch
import sys
import io

from hyperopt.progress import tqdm_progress_callback, no_progress_callback


def test_tqdm_progress_callback_restores_stdout():
    real_stdout = sys.stdout
    with tqdm_progress_callback(initial=0, total=100) as ctx:
        assert sys.stdout != real_stdout
        ctx.postfix = "best loss: 4711"
        ctx.update(42)
    assert sys.stdout == real_stdout


def test_no_progress_callback_no_output():
    output = io.StringIO()
    with patch("sys.stdout", new=output) as stdout, no_progress_callback(
        initial=0, total=100
    ) as ctx:
        ctx.postfix = "best loss: 4711"
        ctx.update(42)
    assert output.getvalue() == ""
