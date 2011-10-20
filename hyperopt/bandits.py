"""
Sample problems on which to test algorithms.

"""
import numpy
import theano
import genson

from theano import tensor

import base
from ht_dist2 import rSON2, rlist2, one_of, uniform, normal, lognormal, ceil_lognormal


class GensonBandit(base.Bandit):
    def __init__(self,genson_file):
        template = genson.load(open(genson_file))
        base.Bandit.__init__(self.template)
    

class Base(base.Bandit):
    def __init__(self, template):
        self.rng = numpy.random.RandomState(55)
        base.Bandit.__init__(self, template)

    def dryrun_argd(self):
        return self.template.render_sample(self.rng)

    def evaluate(self, argd, ctrl):
        return dict(
                loss = -self.score(argd),
                status = 'ok')


class Quadratic1(Base):
    """
    About the simplest problem you could ask for:
    optimize a one-variable quadratic function.
    """

    loss_target = 0

    def __init__(self):
        Base.__init__(self, uniform(-5, 5))

    def score(self, pt):
        return -(pt - 3)**2


class Q1Lognormal(Base):
    """
    About the simplest problem you could ask for:
    optimize a one-variable quadratic function.
    """

    loss_target = 0

    def __init__(self):
        Base.__init__(self, lognormal(0, 2))

    def score(self, pt):
        return max(-(pt - 3)**2, -100)


class TwoArms(Base):
    """
    Each arm yields a reward from a different Gaussian.

    How long does it take the algorithm to identify the best arm?
    """

    loss_target = -1

    def __init__(self):
        Base.__init__(self, one_of(0, 1))

    def score(self, pt):
        arms = 2
        reward_mus = [1] + [0] * (arms - 1)
        reward_sigmas = [1] * arms
        return numpy.random.normal(size=(),
                loc=reward_mus[pt],
                scale=reward_sigmas[pt])


class Distractor(Base):
    """
    This is a nasty function: it has a max in a spike near -10, and a long
    asymptote that is easy to find, but guides hill-climbing approaches away
    from the true max.
    """

    loss_target = -2

    def __init__(self, sigma=10):
        """
        The second peak is at x=-10.
        The prior mean is 0.
        """
        Base.__init__(self, normal(0, sigma))

    def score(self, pt):
        f1 = 1.0 / (1.0 + numpy.exp(-pt))  # climbs rightward from 0.0 to 1.0
        f2 = 2 * numpy.exp(-(pt + 10)**2)  # bump with height 2 at (x=-10)
        return f1 + f2


class EggCarton(Base):
    """
    Essentially, this is a high-frequency sinusoidal function plus a broad quadratic.
    One variable controls the position along the curve.
    The binary variable determines whether the sinusoidal is shifted by pi.

    So there are actually two maxima in this problem, it's just one is more
    probable.  The tricky thing here is dealing with the fact that there are two
    variables and one is discrete.

    """

    loss_target = -1

    def __init__(self):
        Base.__init__(self, rSON2(
            'curve', one_of(0, 1),
            'x', uniform(-20, 20)))

    def score(self, pt):
        if pt['curve']:
            x = pt['x']
        else:
            x = pt['x'] + numpy.pi

        f1 = numpy.sin(x)            # climbs rightward from 0.0 to 1.0
        f2 = 2 * numpy.exp(-(x/5.0)**2)  # bump with height 2 at (x=-10)
        return f1 + f2


class EggCarton2(Base):
    """
    Variant of the EggCarton problem in which noise is added to the score
    function, and there is an option to either have no sinusoidal variation, or
    a negative cosine with variable amplitude.

    Immediate local max is to sample x from spec and turn off the neg cos.
    Better solution is to move x a bit to the side, turn on the neg cos and turn
    up the amp to 1.
    """

    loss_target = -3

    def __init__(self):
        Base.__init__(self, rSON2(
            'x', uniform(-20, 20),
            'hf', one_of(
                rSON2(
                    'kind', 'raw'),
                rSON2(
                    'kind', 'negcos',
                    'amp', uniform(0, 1)))))

    def score(self, pt):
        r = numpy.random.randn() * .1
        x = pt['x']
        r += 2 * numpy.exp(-(x/5.0)**2) # up to 2
        if pt['hf']['kind'] == 'negcos':
            r -= numpy.cos(x) * pt['hf']['amp']

        return r

