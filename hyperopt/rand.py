"""
Random search - presented as hyperopt.fmin_random
"""

from .base import BanditAlgo
from .base import Trials
from .fmin import FMinBase
from .fmin import _FBandit


class FMinRandom(FMinBase):
    def __init__(self, f, domain, trials, seed, max_evals):
        bandit = _FBandit(f, domain)
        algo = BanditAlgo(bandit, seed=seed)
        FMinBase.__init__(self, f, domain, trials, algo)
        self.max_evals = max_evals

    def next(self):
        self.trials.refresh()
        if len(self.trials) == self.max_evals:
            raise StopIteration()
        rval = FMinBase.next(self)


def fmin_random(f, domain, trials=None, seed=123, max_evals=100):
    """
    Minimize `f` over the given `domain` using random search.

    Parameters:
    -----------
    f - a callable taking a dictionary as an argument. It can return either a
        scalar loss value, or a result dictionary. The argument dictionary has
        keys for the hp_XXX nodes in the `domain` and a `ctrl` key.
        If returning a dictionary, `f`
        must return a 'loss' key, and may optionally return a 'status' key and
        certain other reserved keys to communicate with the Experiment and
        optimization algorithm [1, 2]. The entire dictionary will be stored to
        the trials object associated with the experiment.

    domain - a pyll graph involving hp_<xxx> nodes (see `pyll_utils`)

    [1] See keys used in `base.Experiment` and `base.Bandit`
    [2] Optimization algorithms may in some cases use or require auxiliary
        feedback.
    """
    if trials is None:
        trials = Trials()
    rval = FMinRandom(f, domain, trials, seed=seed, max_evals=max_evals)
    rval.exhaust()
    return rval
