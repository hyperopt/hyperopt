import numpy as np
import pytest
from scipy import integrate, stats

from hyperopt.tpe import LGMM1_lpdf


@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
@pytest.mark.parametrize(
    "weights,mus,sigmas",
    [([1.0], [0.0], [1.0]), ([0.3, 0.7], [-0.5, 0.8], [0.4, 1.1])],
)
def test_bounded_lognormal_mixture_density_is_conditional(weights, mus, sigmas, shape):
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    size = int(np.prod(shape))
    samples = np.exp(np.linspace(-0.8, 0.8, size)).reshape(shape)
    low, high = -1.0, 1.0
    acceptance = np.sum(
        weights * (stats.norm.cdf(high, mus, sigmas) - stats.norm.cdf(low, mus, sigmas))
    )
    density = np.sum(
        weights * stats.lognorm.pdf(samples[..., None], s=sigmas, scale=np.exp(mus)),
        axis=-1,
    )

    actual = LGMM1_lpdf(samples, weights, mus, sigmas, low=low, high=high)

    assert actual.shape == shape
    np.testing.assert_allclose(actual, np.log(density / acceptance), rtol=1e-12)


def test_bounded_lognormal_density_integrates_to_one():
    low, high = -1.0, 1.0
    area, error = integrate.quad(
        lambda x: np.exp(LGMM1_lpdf(x, [1.0], [0.0], [1.0], low=low, high=high)),
        np.exp(low),
        np.exp(high),
    )

    assert area == pytest.approx(1.0, abs=1e-10)
    assert error < 1e-8


def test_unbounded_lognormal_density_keeps_its_normalization():
    samples = np.exp([-1.0, 0.0, 1.0])

    actual = LGMM1_lpdf(samples, [1.0], [0.2], [0.7])

    np.testing.assert_allclose(
        actual, stats.lognorm.logpdf(samples, s=0.7, scale=np.exp(0.2))
    )


def test_quantized_bounded_lognormal_mass_is_not_normalized_twice():
    samples = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    low, high, q = -1.0, 1.0, 0.5
    distribution = stats.lognorm(s=1.0)
    lower = np.maximum(samples - q / 2, np.exp(low))
    upper = np.minimum(samples + q / 2, np.exp(high))
    acceptance = distribution.cdf(np.exp(high)) - distribution.cdf(np.exp(low))
    expected = (distribution.cdf(upper) - distribution.cdf(lower)) / acceptance

    actual = LGMM1_lpdf(samples, [1.0], [0.0], [1.0], low=low, high=high, q=q)

    np.testing.assert_allclose(np.exp(actual), expected, rtol=1e-12)
