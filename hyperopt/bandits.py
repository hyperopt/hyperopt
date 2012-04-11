"""
Sample problems on which to test algorithms.

XXX: get some standard optimization problems from literature

"""
import numpy as np

import base
from pyll import as_apply
from pyll import scope

from pyll_utils import hp_choice
from pyll_utils import hp_uniform, hp_loguniform, hp_quniform, hp_qloguniform
from pyll_utils import hp_normal, hp_lognormal, hp_qnormal, hp_qlognormal


@base.as_bandit(loss_target=0)
def quadratic1():
    """
    About the simplest problem you could ask for:
    optimize a one-variable quadratic function.
    """
    return {'loss': (hp_uniform('x', -5, 5) - 3) ** 2}


@base.as_bandit(loss_target=0)
def q1_choice():
    o_x = hp_choice('o_x', [
        (-3, hp_uniform('x_neg', -5, 5)),
        ( 3, hp_uniform('x_pos', -5, 5)),
        ])
    return {'loss': (o_x[0] - o_x[1])  ** 2}


@base.as_bandit(loss_target=0)
def q1_lognormal():
    """
    About the simplest problem you could ask for:
    optimize a one-variable quadratic function.
    """
    return {'loss': scope.max(-(hp_lognormal('x', 0, 2) - 3) ** 2, -100)}


@base.as_bandit(loss_target=-2, rseed=123)
def n_arms(N=2):
    """
    Each arm yields a reward from a different Gaussian.

    The correct arm is arm 0.

    """
    x = hp_choice('x', [0, 1])
    reward_mus = as_apply([-1] + [0] * (N - 1))
    reward_sigmas = as_apply([1] * N)
    return {'loss': scope.normal(reward_mus[x], reward_sigmas[x]),
            'loss_variance': 1.0}


@base.as_bandit(loss_target=-2)
def distractor():
    """
    This is a nasty function: it has a max in a spike near -10, and a long
    asymptote that is easy to find, but guides hill-climbing approaches away
    from the true max.

    The second peak is at x=-10.
    The prior mean is 0.
    """

    loss_target = -2
    x = hp_uniform('x', -15, 15)
    f1 = 1.0 / (1.0 + scope.exp(-x))    # climbs rightward from 0.0 to 1.0
    f2 = 2 * scope.exp(-(x + 10) ** 2)  # bump with height 2 at (x=-10)
    return {'loss': -f1 - f2}


@base.as_bandit(loss_target=-1)
def gauss_wave():
    """
    Essentially, this is a high-frequency sinusoidal function plus a broad quadratic.
    One variable controls the position along the curve.
    The binary variable determines whether the sinusoidal is shifted by pi.

    So there are actually two maxima in this problem, it's just one is more
    probable.  The tricky thing here is dealing with the fact that there are two
    variables and one is discrete.

    """

    x = hp_uniform('x', -20, 20)
    t = hp_choice('curve', [x, x + np.pi])
    f1 = scope.sin(t)
    f2 = 2 * scope.exp(-(t / 5.0) ** 2)
    return {'loss': - (f1 + f2)}


@base.as_bandit(loss_target=-2.5, rseed=123)
def gauss_wave2():
    """
    Variant of the GaussWave problem in which noise is added to the score
    function, and there is an option to either have no sinusoidal variation, or
    a negative cosine with variable amplitude.

    Immediate local max is to sample x from spec and turn off the neg cos.
    Better solution is to move x a bit to the side, turn on the neg cos and turn
    up the amp to 1.
    """

    var = .1
    x = hp_uniform('x', -20, 20)
    amp = hp_uniform('amp', 0, 1)
    t = (scope.normal(0, var) + 2 * scope.exp(-(x / 5.0) ** 2))
    return {'loss': - hp_choice('hf', [t, t + scope.sin(x) * amp]),
            'loss_variance': var}

