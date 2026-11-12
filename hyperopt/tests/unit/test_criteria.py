import numpy as np
import pytest

import hyperopt.criteria as crit


def test_ei():
    rng = np.random.default_rng(123)
    for mean, var in [(0, 1), (-4, 9)]:
        thresholds = np.arange(-5, 5, 0.25) * np.sqrt(var) + mean

        v_n = [
            crit.EI_gaussian_empirical(mean, var, thresh, rng, 10000)
            for thresh in thresholds
        ]
        v_a = [crit.EI_gaussian(mean, var, thresh) for thresh in thresholds]

        # import matplotlib.pyplot as plt
        # plt.plot(thresholds, v_n)
        # plt.plot(thresholds, v_a)
        # plt.show()

        if not np.allclose(v_n, v_a, atol=0.03, rtol=0.03):
            for t, n, a in zip(thresholds, v_n, v_a):
                print((t, n, a, abs(n - a), abs(n - a) / (abs(n) + abs(a))))
            assert 0
            # mean, var, thresh, v_n, v_a)


def test_log_ei():
    for mean, var in [(0, 1), (-4, 9)]:
        thresholds = np.arange(-5, 30, 0.25) * np.sqrt(var) + mean

        ei = np.asarray([crit.EI_gaussian(mean, var, thresh) for thresh in thresholds])
        nlei = np.asarray(
            [crit.logEI_gaussian(mean, var, thresh) for thresh in thresholds]
        )
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
            [
                crit.logEI_gaussian(0, 1, thresh)
                for thresh in [-500, 0, 50, 100, 500, 5000]
            ]
        )
    )


@pytest.mark.parametrize("mean,threshold", [(-2.0, 0.0), (0.0, 0.0), (3.0, 1.0)])
@pytest.mark.parametrize("function", [crit.EI_gaussian, crit.logEI_gaussian])
def test_expected_improvement_with_zero_variance(function, mean, threshold):
    improvement = max(mean - threshold, 0.0)
    if function is crit.logEI_gaussian:
        expected = np.log(improvement) if improvement else -np.inf
    else:
        expected = improvement

    with np.errstate(all="raise"):
        actual = function(mean, 0.0, threshold)

    assert actual == expected


@pytest.mark.parametrize("function", [crit.EI_gaussian, crit.logEI_gaussian])
def test_expected_improvement_with_mixed_variances(function):
    mean = np.array([-2.0, 0.0, 3.0, 0.5])
    var = np.array([0.0, 0.0, 0.0, 1.0])
    expected = np.array([0.0, 0.0, 2.0, crit.EI_gaussian(0.5, 1.0, 1.0)])
    if function is crit.logEI_gaussian:
        with np.errstate(divide="ignore"):
            expected = np.log(expected)

    with np.errstate(all="raise"):
        actual = function(mean, var, 1.0)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("function", [crit.EI_gaussian, crit.logEI_gaussian])
def test_mixed_variance_inputs_preserve_positive_variance_precision(function):
    mean = np.array([1.0, 0.12345678], dtype=np.float32)
    var = np.array([0.0, 1.2345678], dtype=np.float32)
    threshold = np.zeros(2, dtype=np.float32)
    expected = function(mean[1], var[1], threshold[1])

    actual = function(mean, var, threshold)

    assert actual.dtype == np.dtype(np.float64)
    assert actual[1] == expected


def test_ucb():
    assert np.allclose(crit.UCB(0, 1, 1), 1)
    assert np.allclose(crit.UCB(0, 1, 2), 2)
    assert np.allclose(crit.UCB(0, 4, 1), 2)
    assert np.allclose(crit.UCB(1, 4, 1), 3)


# -- flake8
