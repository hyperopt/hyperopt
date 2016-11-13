from __future__ import print_function
from __future__ import division
from builtins import zip
from past.utils import old_div
import numpy as np
import hyperopt.criteria as crit


def test_ei():
    rng = np.random.RandomState(123)
    for mean, var in [(0, 1), (-4, 9)]:
        thresholds = np.arange(-5, 5, .25) * np.sqrt(var) + mean

        v_n = [crit.EI_gaussian_empirical(mean, var, thresh, rng, 10000)
               for thresh in thresholds]
        v_a = [crit.EI_gaussian(mean, var, thresh)
               for thresh in thresholds]

        # import matplotlib.pyplot as plt
        # plt.plot(thresholds, v_n)
        # plt.plot(thresholds, v_a)
        # plt.show()

        if not np.allclose(v_n, v_a, atol=0.03, rtol=0.03):
            for t, n, a in zip(thresholds, v_n, v_a):
                print((t, n, a, abs(n - a), old_div(abs(n - a), (abs(n) + abs(a)))))
            assert 0
            # mean, var, thresh, v_n, v_a)


def test_log_ei():
    for mean, var in [(0, 1), (-4, 9)]:
        thresholds = np.arange(-5, 30, .25) * np.sqrt(var) + mean

        ei = np.asarray(
            [crit.EI_gaussian(mean, var, thresh)
             for thresh in thresholds])
        nlei = np.asarray(
            [crit.logEI_gaussian(mean, var, thresh)
             for thresh in thresholds])
        naive = np.log(ei)
        # import matplotlib.pyplot as plt
        # plt.plot(thresholds, ei, label='ei')
        # plt.plot(thresholds, nlei, label='nlei')
        # plt.plot(thresholds, naive, label='naive')
        # plt.legend()
        # plt.show()

        # -- assert that they match when the threshold isn't too high
        assert np.allclose(nlei, naive)


def test_log_ei_range():
    assert np.all(
        np.isfinite(
            [crit.logEI_gaussian(0, 1, thresh)
             for thresh in [-500, 0, 50, 100, 500, 5000]]))


def test_ucb():
    assert np.allclose(crit.UCB(0, 1, 1), 1)
    assert np.allclose(crit.UCB(0, 1, 2), 2)
    assert np.allclose(crit.UCB(0, 4, 1), 2)
    assert np.allclose(crit.UCB(1, 4, 1), 3)

# -- flake8
