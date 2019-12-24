import sys

from hyperopt.progress import tqdm_progress_callback


def test_tqdm_progress_callback_restores_stdout():
    real_stdout = sys.stdout
    with tqdm_progress_callback(initial=0, total=100) as ctx:
        assert sys.stdout != real_stdout
        ctx.postfix = "best loss: 4711"
        ctx.update(42)
    assert sys.stdout == real_stdout
