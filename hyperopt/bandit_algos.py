"""
XXX
"""
import sys
import logging
logger = logging.getLogger(__name__)

import numpy
import base

class Random(base.BanditAlgo):
    """Random search algorithm
    """

    def __init__(self, bandit):
        base.BanditAlgo.__init__(self, bandit)
        self.rng = numpy.random.RandomState(self.seed)

    def suggest(self, trials, results, N):
        return [self.bandit.template.sample(self.rng)
                for n in range(N)]

try:
    # imports theano, montetheano
    from theano_bandit_algos import TheanoRandom
except ImportError:
    pass

try:
    # imports theano, montetheano
    from theano_bandit_algos import AdaptiveParzenGM, GM_BanditAlgo
except ImportError:
    pass

try:
    # imports theano, montetheano
    from theano_bandit_algos import GaussianGP, GP_BanditAlgo
except ImportError:
    pass

