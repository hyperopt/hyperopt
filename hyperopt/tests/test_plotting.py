"""
Verify that the plotting routines can at least run.

If environment variable HYPEROPT_SHOW is defined and true,
then the plots actually appear.

"""
import unittest
import os

try:
    import matplotlib
    matplotlib.use('svg')  # -- prevents trying to connect to X server
except ImportError:
    import nose
    raise nose.SkipTest()

from hyperopt import Trials
import hyperopt.bandits
import hyperopt.plotting
from hyperopt import rand, fmin

def get_do_show():
    rval = int(os.getenv('HYPEROPT_SHOW', '0'))
    print 'do_show =', rval
    return rval

class TestPlotting(unittest.TestCase):
    def setUp(self):
        bandit = self.bandit = hyperopt.bandits.many_dists()
        trials = self.trials = Trials()
        fmin(lambda x: x,
            space=bandit.expr,
            trials=trials,
            algo=rand.suggest,
            max_evals=200)

    def test_plot_history(self):
        hyperopt.plotting.main_plot_history(
                self.trials,
                do_show=get_do_show())

    def test_plot_histogram(self):
        hyperopt.plotting.main_plot_histogram(
                self.trials,
                do_show=get_do_show())

    def test_plot_vars(self):
        hyperopt.plotting.main_plot_vars(
                self.trials,
                self.bandit,
                do_show=get_do_show())

