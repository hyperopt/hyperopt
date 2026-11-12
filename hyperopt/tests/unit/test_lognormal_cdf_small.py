from functools import partial

import numpy as np
import pytest
from scipy import stats

from hyperopt import Trials, fmin, hp, pyll, tpe
from hyperopt.tpe import LGMM1_lpdf, ap_qloguniform_sampler, lognormal_cdf


@pytest.mark.parametrize("mu", [0.0, -30.0, -100.0, -500.0])
def test_lognormal_cdf_preserves_small_positive_values_and_zero(mu):
    values = np.r_[0.0, np.exp(mu + np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))]
    expected = stats.lognorm.cdf(values, s=1.0, scale=np.exp(mu))

    actual = lognormal_cdf(values, mu, 1.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
    assert actual[0] == 0.0
    assert actual[3] == 0.5


@pytest.mark.parametrize("scale", [1.0, 1e-15, 1e-100])
@pytest.mark.parametrize("bounded", [False, True])
def test_quantized_lognormal_mixture_matches_probability_of_each_bin(scale, bounded):
    samples = np.array([0.0, 1.0, 2.0, 3.0]) * scale
    weights = np.array([0.3, 0.7])
    mus = np.log(scale) + np.array([-0.5, 0.5])
    sigmas = np.array([0.8, 1.2])
    low, high = np.log(0.1 * scale), np.log(4.0 * scale)
    lower = np.maximum(samples - scale / 2, np.exp(low) if bounded else 0.0)
    upper = (
        np.minimum(samples + scale / 2, np.exp(high))
        if bounded
        else samples + scale / 2
    )
    probability = np.sum(
        weights
        * (
            stats.lognorm.cdf(upper[:, None], s=sigmas, scale=np.exp(mus))
            - stats.lognorm.cdf(lower[:, None], s=sigmas, scale=np.exp(mus))
        ),
        axis=1,
    )
    if bounded:
        probability /= np.sum(
            weights
            * (stats.norm.cdf(high, mus, sigmas) - stats.norm.cdf(low, mus, sigmas))
        )

    actual = LGMM1_lpdf(
        samples,
        weights,
        mus,
        sigmas,
        q=scale,
        **({"low": low, "high": high} if bounded else {}),
    )

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.log(probability), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("scale", [1e-15, 1e-100])
def test_tpe_quantized_loguniform_search_is_invariant_to_units(scale):
    def search(multiplier):
        trials = Trials()
        fmin(
            lambda value: (value / multiplier - 2.5) ** 2,
            hp.qloguniform(
                "x", np.log(multiplier), np.log(10 * multiplier), multiplier / 10
            ),
            algo=partial(tpe.suggest, n_startup_jobs=3),
            max_evals=12,
            trials=trials,
            rstate=np.random.default_rng(41),
            show_progressbar=False,
            verbose=False,
        )
        return np.array(trials.vals["x"]) / multiplier

    np.testing.assert_allclose(search(scale), search(1.0), rtol=1e-12, atol=1e-13)


def test_lognormal_cdf_keeps_negative_input_validation_and_empty_output():
    with pytest.raises(ValueError, match="negative arg"):
        lognormal_cdf(np.array([-1.0]), 0.0, 1.0)
    assert lognormal_cdf(np.array([]), -30.0, 1.0).shape == (0,)


@pytest.mark.parametrize("low,high,q", [(-800.0, -700.0, 1e-310), (-4.0, 0.0, 0.1)])
def test_quantized_loguniform_zero_observations_keep_a_finite_log_floor(low, high, q):
    posterior = ap_qloguniform_sampler(
        np.array([0.0, 0.0]), 1.0, low, high, q, size=(10,)
    )

    with np.errstate(divide="raise", invalid="raise"):
        samples = pyll.stochastic.sample(posterior, rng=np.random.default_rng(13))

    assert samples.shape == (10,)
    assert np.isfinite(samples).all()
    assert np.all(samples >= 0)
    assert np.all(samples <= np.exp(high) + q / 2)
