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

from hyperopt import Experiment, Trials, TreeParzenEstimator
from .test_tpe import many_dists
import hyperopt.bandits
import hyperopt.plotting

def get_do_show():
    rval = int(os.getenv('HYPEROPT_SHOW', '0'))
    print 'do_show =', rval
    return rval

class TestPlotting(unittest.TestCase):
    def setUp(self):
        bandit = self.bandit = many_dists()
        algo = TreeParzenEstimator(bandit)
        trials = Trials()
        experiment = Experiment(trials, algo, async=False)
        experiment.max_queue_len = 1
        N=200
        if 0:
            import cProfile
            stats = cProfile.runctx('experiment.run(N)', globals={},
                    locals=locals(), filename='fooprof')
            import pstats
            p = pstats.Stats('fooprof')
            p.sort_stats('cumulative').print_stats(10)
            p.sort_stats('time').print_stats(10)
        else:
            experiment.run(N)
        self.trials = trials

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

