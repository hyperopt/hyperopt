"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

from hyperopt.base import Bandit, BanditAlgo
from hyperopt.theano_gp import GP_BanditAlgo
from hyperopt.ht_dist2 import rSON2, normal
from hyperopt.experiments import SerialExperiment

def test_fit_normal():
    class B(Bandit):
        def __init__(self):
            Bandit.__init__(self, rSON2('x', normal(0, 1)))
        @classmethod
        def evaluate(cls, config, ctrl):
            return dict(loss=(config['x'] - 2)**2, status='ok')
    gp = GP_BanditAlgo(B())
    se = SerialExperiment(gp)
    se.run(gp.n_startup_jobs)

    assert len(se.trials) == len(se.results) == gp.n_startup_jobs

    # now trigger the use of the GP, EI, etc.
    se.run(1)

# test that it can fit a GP to each of the simple variable types:
#  - normal
#  - uniform
#  - lognormal
#  - quantized lognormal
#  - categorical


# for a Bandit of two variables, of which one doesn't do anything
# test that the learned length scales are appropriate


# for a Bandit with
#    template one_of({'a':normal, 'b':normal}, {'c':normal, 'd':normal})
# and an evaluate that depends only on a or d,
# show that the length scales of b and c go to inf.
