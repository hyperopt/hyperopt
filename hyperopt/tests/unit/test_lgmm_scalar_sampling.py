import numpy as np
import pytest

from hyperopt import tpe
from hyperopt.pyll import stochastic


@pytest.mark.parametrize("size", [(), 3, (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("q", [None, 0.25])
def test_lgmm_sample_shapes_and_random_state(size, q):
    weights = np.array([0.3, 0.7])
    mus = np.array([-1.0, 0.6])
    sigmas = np.array([0.2, 0.7])
    rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(42)
    count = int(np.prod(size))
    active = reference_rng.multinomial(1, weights, size=count).argmax(axis=1)
    expected = np.exp(reference_rng.normal(mus[active], sigmas[active])).reshape(size)
    if q is not None:
        expected = np.round(expected / q) * q

    actual = tpe.LGMM1(weights, mus, sigmas, q=q, rng=rng, size=size)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert rng.random() == reference_rng.random()


@pytest.mark.parametrize("quantized", [False, True])
def test_default_size_in_lognormal_posterior_graph(quantized):
    if quantized:
        expression = tpe.ap_qlognormal_sampler([0.5, 1.0, 2.0], 1.0, 0.0, 1.0, 0.25)
    else:
        expression = tpe.ap_loglognormal_sampler([0.5, 1.0, 2.0], 1.0, 0.0, 1.0)
    sample = stochastic.sample(expression, rng=np.random.default_rng(42))
    assert sample.shape == ()
    assert np.isfinite(sample)
    assert sample >= 0
    if quantized:
        assert sample / 0.25 == round(float(sample / 0.25))
