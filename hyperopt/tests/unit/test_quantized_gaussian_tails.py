import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from hyperopt.tpe import GMM1_lpdf


def integrated_log_density(samples, weights, mus, sigmas, q, low=None, high=None):
    def density(x):
        return sum(
            w * norm.pdf(x, mu, sigma) for w, mu, sigma in zip(weights, mus, sigmas)
        )

    accept = 1.0 if low is None else quad(density, low, high, epsabs=0, epsrel=1e-11)[0]
    result = []
    for sample in np.asarray(samples).flat:
        if q is None:
            mass = density(sample)
        else:
            lower, upper = sample - q / 2, sample + q / 2
            if low is not None:
                lower, upper = max(lower, low), min(upper, high)
            mass = quad(density, lower, upper, epsabs=0, epsrel=1e-11)[0]
        result.append(np.log(mass) - np.log(accept))
    return np.asarray(result).reshape(np.shape(samples))


@pytest.mark.parametrize(
    "samples", [10.0, [-12.0, -10.0, 0.0, 10.0, 12.0], [[-9.0, 0.0], [9.0, 12.0]]]
)
@pytest.mark.parametrize("q", [0.1, 0.5])
def test_quantized_gaussian_tail_mass(samples, q):
    args = dict(weights=[1.0], mus=[0.0], sigmas=[1.0], q=q)
    expected = integrated_log_density(samples, **args)
    with np.errstate(divide="raise", invalid="raise"):
        actual = GMM1_lpdf(samples, **args)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("q", [0.1, 0.5])
def test_quantized_tail_mass_sums_mixture_components(q):
    samples = np.array([-15.0, -12.0, 0.0, 12.0, 15.0])
    args = dict(weights=[0.2, 0.8], mus=[-0.3, 0.5], sigmas=[0.7, 1.3], q=q)
    expected = integrated_log_density(samples, **args)
    with np.errstate(divide="raise", invalid="raise"):
        actual = GMM1_lpdf(samples, **args)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bounds", [(-11.0, -9.0), (9.0, 11.0), (-1.0, 1.0)])
@pytest.mark.parametrize("q", [None, 0.2])
def test_truncated_gaussian_tail_normalization(bounds, q):
    low, high = bounds
    samples = np.linspace(low, high, 5)
    args = dict(weights=[1.0], mus=[0.0], sigmas=[1.0], q=q, low=low, high=high)
    expected = integrated_log_density(samples, **args)
    with np.errstate(divide="raise", invalid="raise"):
        actual = GMM1_lpdf(samples, **args)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_candidate_likelihood_ratios_remain_finite():
    samples = np.array([-10.2, -10.0, -9.8])
    below = dict(weights=[0.5, 0.5], mus=[-10.0, 0.0], sigmas=[0.2, 1.0], q=0.1)
    above = dict(weights=[0.5, 0.5], mus=[10.0, 0.0], sigmas=[0.2, 1.0], q=0.1)
    expected = integrated_log_density(samples, **below) - integrated_log_density(
        samples, **above
    )
    with np.errstate(divide="raise", invalid="raise"):
        scores = GMM1_lpdf(samples, **below) - GMM1_lpdf(samples, **above)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
    assert np.argmax(scores) == np.argmax(expected)
