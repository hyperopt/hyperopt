"""
Verify that the plotting routines can at least run.

If environment variable HYPEROPT_SHOW is defined and true,
then the plots actually appear.

"""
import unittest
import os
import nose

import matplotlib
matplotlib.use('svg')  # -- prevents trying to connect to X server

from hyperopt import Random, Experiment, Trials
import hyperopt.bandits
import hyperopt.plotting

def get_do_show():
    rval = int(os.getenv('HYPEROPT_SHOW', '0'))
    print 'do_show =', rval
    return rval

class TestPlotting(unittest.TestCase):
    def setUp(self):
        bandit = hyperopt.bandits.GaussWave2()
        algo = Random(bandit)
        trials = Trials()
        experiment = Experiment(trials, algo, async=False)
        experiment.max_queue_len = 50
        experiment.run(500)
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
        raise nose.SkipTest()
        hyperopt.plotting.main_plot_vars(
                self.trials,
                do_show=get_do_show())

