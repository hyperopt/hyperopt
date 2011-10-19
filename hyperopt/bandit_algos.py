"""
Hyper-parameter optimization algorithms (BanditAlgo subclasses)

"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import sys
import logging
logger = logging.getLogger(__name__)

import numpy
import base

class Random(base.BanditAlgo):
    """Random search director
    """

    def __init__(self, *args, **kwargs):
        base.BanditAlgo.__init__(self, *args, **kwargs)
        self.rng = numpy.random.RandomState(self.seed)

    def suggest(self, X_list, Ys, Y_status, N):
        return [self.bandit.template.sample(self.rng)
                for n in range(N)]

try:
    # imports theano
    from theano_bandit_algos import TheanoRandom
except ImportError:
    pass

try:
    # imports theano, montetheano
    from theano_gm import AdaptiveParzenGM, GM_BanditAlgo
except ImportError:
    pass

try:
    # imports theano, montetheano
    from theano_gp import GaussianGP, GP_BanditAlgo
except ImportError:
    pass

