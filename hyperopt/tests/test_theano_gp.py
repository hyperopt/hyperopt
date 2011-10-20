"""
Tests of hyperopt.theano_gp
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import numpy

import matplotlib.pyplot as plt

from hyperopt.idxs_vals_rnd import IdxsValsList
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

    class A(GP_BanditAlgo):
        def theano_suggest_from_model(self, X_IVLs, Ys, N):
            x_all, y_all, y_mean = self.prepare_GP_training_data(X_IVLs, Ys)
            self.fit_GP(x_all, y_all, y_mean)

            plt.scatter(x_all[0].vals, y_all)
            plt.xlim([-5, 5])
            xmesh = numpy.arange(-5, 5, .1)
            gp_mean, gp_var = self.GP_mean_variance(
                    IdxsValsList.fromlists([numpy.arange(len(xmesh))], [xmesh]))
            plt.plot(xmesh, gp_mean)
            plt.plot(xmesh, gp_mean + numpy.sqrt(gp_var))
            plt.plot(xmesh, gp_mean - numpy.sqrt(gp_var))

            plt.show()

            return self.theano_suggest_from_prior(N)

    se = SerialExperiment(A(B()))
    se.run(A.n_startup_jobs)

    assert len(se.trials) == len(se.results) == A.n_startup_jobs

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
