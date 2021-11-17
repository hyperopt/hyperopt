from past.utils import old_div
import unittest

import numpy as np

from hyperopt import Trials, Domain, fmin, hp, base
from hyperopt.rand import suggest
from hyperopt.pyll import as_apply
from hyperopt.pyll import scope


# -- define this bandit here too for completeness' sake
def domain_constructor(**b_kwargs):
    """
    Decorate a function that returns a pyll expressions so that
    it becomes a Domain instance instead of a function

    Example:

    @domain_constructor(loss_target=0)
    def f(low, high):
        return {'loss': hp.uniform('x', low, high) ** 2 }

    """

    def deco(f):
        def wrapper(*args, **kwargs):
            if "name" in b_kwargs:
                _b_kwargs = b_kwargs
            else:
                _b_kwargs = dict(b_kwargs, name=f.__name__)
            f_rval = f(*args, **kwargs)
            domain = Domain(lambda x: x, f_rval, **_b_kwargs)
            return domain

        wrapper.__name__ = f.__name__
        return wrapper

    return deco


@domain_constructor()
def coin_flip():
    """Possibly the simplest possible Bandit implementation"""
    return {"loss": hp.choice("flip", [0.0, 1.0]), "status": base.STATUS_OK}


@domain_constructor(loss_target=0)
def quadratic1():
    """
    About the simplest problem you could ask for:
    optimize a one-variable quadratic function.
    """
    return {"loss": (hp.uniform("x", -5, 5) - 3) ** 2, "status": base.STATUS_OK}


@domain_constructor(loss_target=0)
def q1_choice():
    o_x = hp.choice(
        "o_x", [(-3, hp.uniform("x_neg", -5, 5)), (3, hp.uniform("x_pos", -5, 5))]
    )
    return {"loss": (o_x[0] - o_x[1]) ** 2, "status": base.STATUS_OK}


@domain_constructor(loss_target=0)
def q1_lognormal():
    """
    About the simplest problem you could ask for:
    optimize a one-variable quadratic function.
    """
    return {
        "loss": scope.min(0.1 * (hp.lognormal("x", 0, 2) - 10) ** 2, 10),
        "status": base.STATUS_OK,
    }


@domain_constructor(loss_target=-2)
def n_arms(N=2):
    """
    Each arm yields a reward from a different Gaussian.

    The correct arm is arm 0.

    """
    rng = np.random.default_rng(123)
    x = hp.choice("x", [0, 1])
    reward_mus = as_apply([-1] + [0] * (N - 1))
    reward_sigmas = as_apply([1] * N)
    return {
        "loss": scope.normal(reward_mus[x], reward_sigmas[x], rng=rng),
        "loss_variance": 1.0,
        "status": base.STATUS_OK,
    }


@domain_constructor(loss_target=-2)
def distractor():
    """
    This is a nasty function: it has a max in a spike near -10, and a long
    asymptote that is easy to find, but guides hill-climbing approaches away
    from the true max.

    The second peak is at x=-10.
    The prior mean is 0.
    """

    x = hp.uniform("x", -15, 15)
    # climbs rightward from 0.0 to 1.0
    f1 = old_div(1.0, (1.0 + scope.exp(-x)))
    f2 = 2 * scope.exp(-((x + 10) ** 2))  # bump with height 2 at (x=-10)
    return {"loss": -f1 - f2, "status": base.STATUS_OK}


@domain_constructor(loss_target=-1)
def gauss_wave():
    """
    Essentially, this is a high-frequency sinusoidal function plus a broad quadratic.
    One variable controls the position along the curve.
    The binary variable determines whether the sinusoidal is shifted by pi.

    So there are actually two maxima in this problem, it's just one is more
    probable.  The tricky thing here is dealing with the fact that there are two
    variables and one is discrete.

    """

    x = hp.uniform("x", -20, 20)
    t = hp.choice("curve", [x, x + np.pi])
    f1 = scope.sin(t)
    f2 = 2 * scope.exp(-((old_div(t, 5.0)) ** 2))
    return {"loss": -(f1 + f2), "status": base.STATUS_OK}


@domain_constructor(loss_target=-2.5)
def gauss_wave2():
    """
    Variant of the GaussWave problem in which noise is added to the score
    function, and there is an option to either have no sinusoidal variation, or
    a negative cosine with variable amplitude.

    Immediate local max is to sample x from spec and turn off the neg cos.
    Better solution is to move x a bit to the side, turn on the neg cos and turn
    up the amp to 1.
    """

    rng = np.random.default_rng(123)
    var = 0.1
    x = hp.uniform("x", -20, 20)
    amp = hp.uniform("amp", 0, 1)
    t = scope.normal(0, var, rng=rng) + 2 * scope.exp(-((old_div(x, 5.0)) ** 2))
    return {
        "loss": -hp.choice("hf", [t, t + scope.sin(x) * amp]),
        "loss_variance": var,
        "status": base.STATUS_OK,
    }


@domain_constructor(loss_target=0)
def many_dists():
    a = hp.choice("a", [0, 1, 2])
    b = hp.randint("b", 10)
    bb = hp.randint("bb", 12, 25)
    c = hp.uniform("c", 4, 7)
    d = hp.loguniform("d", -2, 0)
    e = hp.quniform("e", 0, 10, 3)
    f = hp.qloguniform("f", 0, 3, 2)
    g = hp.normal("g", 4, 7)
    h = hp.lognormal("h", -2, 2)
    i = hp.qnormal("i", 0, 10, 2)
    j = hp.qlognormal("j", 0, 2, 1)
    k = hp.pchoice("k", [(0.1, 0), (0.9, 1)])
    z = a + b + bb + c + d + e + f + g + h + i + j + k
    return {"loss": scope.float(scope.log(1e-12 + z ** 2)), "status": base.STATUS_OK}


@domain_constructor(loss_target=0.398)
def branin():
    """
    The Branin, or Branin-Hoo, function has three global minima,
    and is roughly an angular trough across a 2D input space.

        f(x, y) = a (y - b x ** 2 + c x - r ) ** 2 + s (1 - t) cos(x) + s

    The recommended values of a, b, c, r, s and t are:
        a = 1
        b = 5.1 / (4 pi ** 2)
        c = 5 / pi
        r = 6
        s = 10
        t = 1 / (8 * pi)

    Global Minima:
      [(-pi, 12.275),
       (pi, 2.275),
       (9.42478, 2.475)]

    Source: http://www.sfu.ca/~ssurjano/branin.html
    """
    x = hp.uniform("x", -5.0, 10.0)
    y = hp.uniform("y", 0.0, 15.0)
    pi = float(np.pi)
    loss = (
        (y - (old_div(5.1, (4 * pi ** 2))) * x ** 2 + 5 * x / pi - 6) ** 2
        + 10 * (1 - old_div(1, (8 * pi))) * scope.cos(x)
        + 10
    )
    return {"loss": loss, "loss_variance": 0, "status": base.STATUS_OK}


class DomainExperimentMixin:
    def test_basic(self):
        domain = self._domain_cls()
        # print 'domain params', domain.params, domain
        # print 'algo params', algo.vh.params
        trials = Trials()
        fmin(
            lambda x: x,
            domain.expr,
            trials=trials,
            algo=suggest,
            rstate=np.random.default_rng(4),
            max_evals=self._n_steps,
        )
        assert trials.average_best_error(domain) - domain.loss_target < 0.2

    @classmethod
    def make(cls, domain_cls, n_steps=500):
        class Tester(unittest.TestCase, cls):
            def setUp(self):
                self._n_steps = n_steps
                self._domain_cls = domain_cls

        Tester.__name__ = domain_cls.__name__ + "Tester"
        return Tester


quadratic1Tester = DomainExperimentMixin.make(quadratic1)
q1_lognormalTester = DomainExperimentMixin.make(q1_lognormal)
q1_choiceTester = DomainExperimentMixin.make(q1_choice)
n_armsTester = DomainExperimentMixin.make(n_arms)
distractorTester = DomainExperimentMixin.make(distractor)
gauss_waveTester = DomainExperimentMixin.make(gauss_wave)
gauss_wave2Tester = DomainExperimentMixin.make(gauss_wave2, n_steps=5000)
many_distsTester = DomainExperimentMixin.make(many_dists)
braninTester = DomainExperimentMixin.make(branin)


class CasePerDomain:
    # -- this is a mixin
    # -- Override self.work to execute a test for each kind of self.bandit

    def test_quadratic1(self):
        self.bandit = quadratic1()
        self.work()

    def test_q1lognormal(self):
        self.bandit = q1_lognormal()
        self.work()

    def test_twoarms(self):
        self.bandit = n_arms()
        self.work()

    def test_distractor(self):
        self.bandit = distractor()
        self.work()

    def test_gausswave(self):
        self.bandit = gauss_wave()
        self.work()

    def test_gausswave2(self):
        self.bandit = gauss_wave2()
        self.work()

    def test_many_dists(self):
        self.bandit = many_dists()
        self.work()

    def test_branin(self):
        self.bandit = branin()
        self.work()


class NonCategoricalCasePerDomain:
    # -- this is a mixin
    # -- Override self.work to execute a test for each kind of self.bandit

    def test_quadratic1(self):
        self.bandit = quadratic1()
        self.work()

    def test_q1lognormal(self):
        self.bandit = q1_lognormal()
        self.work()

    def test_twoarms(self):
        self.bandit = n_arms()
        self.work()

    def test_distractor(self):
        self.bandit = distractor()
        self.work()

    def test_branin(self):
        self.bandit = branin()
        self.work()


# -- non-blank last line for flake8
