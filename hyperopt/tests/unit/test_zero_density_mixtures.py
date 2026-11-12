from functools import partial

import numpy as np
import pytest
from scipy import stats

from hyperopt import fmin, hp, tpe
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.tpe import GMM1_lpdf, logsum_rows


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_row_log_sum_preserves_zero_mass_and_infinite_mass(dtype):
    values = np.array(
        [
            [-np.inf, -np.inf],
            [-1000, -1001],
            [0, -np.inf],
            [np.inf, 3],
            [np.inf, np.inf],
        ],
        dtype=dtype,
    )
    expected = np.array([-np.inf, -1000 + np.log1p(np.exp(-1)), 0, np.inf, np.inf])

    with np.errstate(invalid="raise"):
        actual = logsum_rows(values)

    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    assert not np.isnan(actual).any()


@pytest.mark.parametrize(
    "weights,mus,sigmas", [([1.0], [0.0], [1.0]), ([0.3, 0.7], [-0.5, 0.5], [1.0, 2.0])]
)
def test_gaussian_mixture_far_tails_return_negative_infinity(weights, mus, sigmas):
    samples = np.array([0.0, 1.0, 1e155, -1e155])
    with np.errstate(over="ignore"):
        components = stats.norm.logpdf(
            samples[:, None], loc=mus, scale=sigmas
        ) + np.log(weights)
        expected = np.logaddexp.reduce(components, axis=1)
        with np.errstate(invalid="raise"):
            actual = GMM1_lpdf(samples, weights, mus, sigmas)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert np.isneginf(actual[2:]).all()


@pytest.mark.parametrize("point", [1e155, -1e155])
def test_tpe_accepts_finite_initial_points_in_zero_density_tails(point):
    trials = generate_trials_to_calculate([{"x": 0.0}, {"x": point}])

    with np.errstate(over="ignore", divide="ignore", invalid="raise"):
        fmin(
            lambda value: float(abs(value) < 1e100),
            hp.normal("x", 0.0, 1.0),
            algo=partial(tpe.suggest, n_startup_jobs=2),
            max_evals=6,
            trials=trials,
            rstate=np.random.default_rng(41),
            show_progressbar=False,
            verbose=False,
        )

    assert len(trials) == 8
    assert np.isfinite(trials.vals["x"]).all()
    assert min(trials.losses()) == 0.0


def test_row_log_sum_keeps_finite_results_and_empty_sample_batches():
    values = np.log([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]])

    np.testing.assert_allclose(logsum_rows(values), [0.0, 0.0], atol=1e-15)
    assert logsum_rows(np.empty((0, 2))).shape == (0,)
