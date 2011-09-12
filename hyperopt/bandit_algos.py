"""
XXX
"""
import numpy
import theano

import base
import ht_dist2

class Random(base.BanditAlgo):
    """Random search director
    """

    def __init__(self, *args, **kwargs):
        base.BanditAlgo.__init__(self, *args, **kwargs)
        self.rng = numpy.random.RandomState(self.seed)

    def suggest(self, X_list, Ys, Y_status, N):
        return [self.bandit.template.sample(self.rng)
                for n in range(N)]


class TheanoRandom(base.TheanoBanditAlgo):
    """Random search director, but testing the machinery that translates
    doctree configurations into sparse matrix configurations.
    """
    def set_bandit(self, bandit):
        base.TheanoBanditAlgo.set_bandit(self, bandit)
        self._sampler = theano.function(
                [self.s_N],
                self.s_idxs + self.s_vals)

    def theano_suggest(self, X_idxs, X_vals, Y, Y_status, N):
        """Ignore X and Y, draw from prior"""
        rvals = self._sampler(N)
        return rvals[:len(rvals)/2], rvals[len(rvals)/2:]

class GM_BanditAlgo(base.BanditAlgo):
    pass


class GP_BanditAlgo(base.BanditAlgo):
    pass
