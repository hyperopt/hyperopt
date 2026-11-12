from functools import partial

import numpy as np
import pytest
from scipy import special, stats

from hyperopt import Trials, fmin, hp, tpe
from hyperopt.tpe import GMM1_lpdf


@pytest.mark.parametrize("scale", [1.0, 1e154, 1e200])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("shape", [(), (2, 2)])
def test_gaussian_mixture_log_density_remains_finite(scale, bounded, shape):
    samples = np.linspace(-0.5, 0.5, int(np.prod(shape))).reshape(shape) * scale
    weights = np.array([0.3, 0.7])
    mus = np.array([-0.2, 0.5]) * scale
    sigmas = np.array([0.4, 1.2]) * scale
    bounds = {"low": -scale, "high": scale} if bounded else {}
    expected = special.logsumexp(
        stats.norm.logpdf(samples[..., None], loc=mus, scale=sigmas) + np.log(weights),
        axis=-1,
    )
    if bounded:
        acceptance = np.sum(
            weights
            * (stats.norm.cdf(scale, mus, sigmas) - stats.norm.cdf(-scale, mus, sigmas))
        )
        expected -= np.log(acceptance)

    actual = GMM1_lpdf(samples, weights, mus, sigmas, **bounds)

    assert actual.shape == shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("scale", [1e155, 1e200])
def test_tpe_uniform_search_preserves_the_trajectory_when_rescaled(scale):
    def search(multiplier):
        trials = Trials()
        fmin(
            lambda x: (x / multiplier - 0.2) ** 2,
            hp.uniform("x", 0.0, multiplier),
            algo=partial(tpe.suggest, n_startup_jobs=3),
            max_evals=12,
            trials=trials,
            rstate=np.random.default_rng(41),
            show_progressbar=False,
            verbose=False,
        )
        return np.array(trials.vals["x"]) / multiplier

    np.testing.assert_allclose(search(scale), search(1.0), rtol=1e-12, atol=1e-14)
