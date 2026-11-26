import numpy as np
import pytest

from hyperopt.rdists import loguniform_gen


@pytest.mark.parametrize("make_rng", [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize("size", [(), 5, (2, 3), (0,)])
def test_loguniform_uses_supplied_random_state(make_rng, size):
    rng = make_rng(17)
    reference = make_rng(17)
    expected = 3.0 * np.exp(reference.uniform(-2.0, 1.0, size)) - 4.0
    actual = loguniform_gen(-2.0, 1.0).rvs(
        size=size, random_state=rng, loc=-4.0, scale=3.0
    )
    np.testing.assert_array_equal(actual, expected)
    assert rng.uniform() == reference.uniform()


def test_loguniform_integer_seed_is_repeatable():
    dist = loguniform_gen(-2.0, 1.0)
    np.testing.assert_array_equal(
        dist.rvs(size=10, random_state=42), dist.rvs(size=10, random_state=42)
    )


def test_loguniform_explicit_rng_does_not_advance_global_state():
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected_next = np.random.uniform()
        np.random.seed(123)
        loguniform_gen().rvs(size=5, random_state=np.random.default_rng(42))
        assert np.random.uniform() == expected_next
    finally:
        np.random.set_state(state)


def test_loguniform_respects_distribution_random_state():
    dist = loguniform_gen(-1.0, 2.0)
    dist.random_state = np.random.default_rng(23)
    reference = np.random.default_rng(23)
    for size in [4, (2, 3)]:
        np.testing.assert_array_equal(
            dist.rvs(size=size), np.exp(reference.uniform(-1.0, 2.0, size))
        )
