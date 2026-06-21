from importlib.metadata import PackageNotFoundError, version

# -- syntactic sugar
# -- exceptions
# -- Import built-in optimization algorithms
from . import anneal, atpe, exceptions, hp, mix, rand, tpe
from .base import (
    JOB_STATE_DONE,
    JOB_STATE_ERROR,
    JOB_STATE_NEW,
    JOB_STATE_RUNNING,
    JOB_STATES,
    STATUS_FAIL,
    STATUS_NEW,
    STATUS_OK,
    STATUS_RUNNING,
    STATUS_STRINGS,
    STATUS_SUSPENDED,
    Ctrl,
    Domain,
    Trials,
    trials_from_docs,
)
from .fmin import FMinIter, fmin, fmin_pass_expr_memo_ctrl, partial, space_eval

# -- spark extension
from .spark import SparkTrials

try:
    __version__ = version("hyperopt")
except PackageNotFoundError:
    __version__ = "unknown"
