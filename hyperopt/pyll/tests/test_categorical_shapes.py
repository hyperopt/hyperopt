import numpy as np
import pytest

from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import categorical, sample


@pytest.mark.parametrize(
    "size", [(2, 3), (2, 1, 3), (0, 3), (2, 0), (2, 0, 3), (6,), 6, (), (0,)]
)
def test_categorical_preserves_requested_shape_and_flat_draw_order(size):
    shape = (1,) if size == () else ((size,) if isinstance(size, int) else size)
    probabilities = np.array([0.2, 0.3, 0.5])
    before = probabilities.copy()
    reference_rng = np.random.default_rng(17)
    flat = reference_rng.multinomial(1, probabilities, int(np.prod(shape))).argmax(
        axis=-1
    )
    expected = flat.reshape(shape)
    rng = np.random.default_rng(17)

    actual = categorical(probabilities, rng=rng, size=size)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == shape
    assert rng.random() == reference_rng.random()
    np.testing.assert_array_equal(probabilities, before)


@pytest.mark.parametrize("category", [0, 1, 2])
def test_categorical_matrix_keeps_category_indices(category):
    probabilities = np.eye(3)[category]

    actual = categorical(probabilities, rng=np.random.default_rng(3), size=(2, 4))

    np.testing.assert_array_equal(actual, np.full((2, 4), category))


def test_categorical_matrix_is_available_through_stochastic_graph_sampling():
    expression = {
        "choices": scope.categorical([0.0, 1.0, 0.0], size=(2, 3)),
        "offset": scope.uniform(-1, 1),
    }

    actual = sample(expression, np.random.default_rng(12))

    np.testing.assert_array_equal(actual["choices"], np.ones((2, 3), dtype=int))
    assert -1 < actual["offset"] < 1


def test_categorical_keeps_per_row_probabilities():
    probabilities = np.eye(3)[[2, 0, 1]]

    actual = categorical(probabilities, rng=np.random.default_rng(8), size=3)

    np.testing.assert_array_equal(actual, [2, 0, 1])
