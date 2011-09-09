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
        return [self.bandit.template.render_sample(self.rng)
                for n in range(N)]


class TheanoRandom(base.TheanoBanditAlgo):
    pass

class GM_BanditAlgo(base.BanditAlgo):
    pass


class GP_BanditAlgo(base.BanditAlgo):
    pass
