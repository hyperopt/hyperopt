import numpy as np
import pytest

from hyperopt import Trials, anneal, base, fmin, hp, rand, space_eval
from hyperopt.fmin import generate_trials_to_calculate


def evaluated_trials(space, points):
    trials = generate_trials_to_calculate(points)
    fmin(
        lambda value: 0.0,
        space,
        algo=rand.suggest,
        trials=trials,
        max_evals=len(points),
        rstate=np.random.default_rng(19),
        show_progressbar=False,
    )
    return trials


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("probabilities", [[1 / 3] * 3, [0.1, 0.2, 0.7]])
def test_annealing_batch_has_one_count_per_category(batch_size, probabilities):
    space = hp.pchoice("choice", list(zip(probabilities, ["a", "b", "c"])))
    trials = evaluated_trials(space, [{"choice": 2}] * 4)
    domain = base.Domain(lambda value: 0.0, space)
    ids = list(range(4, 4 + batch_size))
    seed = 37
    reference_rng = np.random.default_rng([seed, *ids])
    reference_rng.geometric(0.5, size=batch_size)
    # Every observed configuration chose c. Four observations give prior mass 1/1.4.
    prior_mass = 1 / 1.4
    expected_probabilities = prior_mass * np.array(probabilities)
    expected_probabilities[2] += 1 - prior_mass
    expected = [
        reference_rng.multinomial(1, expected_probabilities).argmax()
        for _ in range(batch_size)
    ]
    algorithm = anneal.AnnealingAlgo(domain, trials, seed)

    indices, values = algorithm.batch(ids)

    np.testing.assert_array_equal(indices["choice"], ids)
    np.testing.assert_array_equal(values["choice"], expected)
    assert algorithm.rng.random() == reference_rng.random()


def test_public_annealing_batch_handles_conditional_categorical_parameters():
    space = hp.choice(
        "branch",
        [
            {"left": hp.pchoice("inner", [(0.2, "a"), (0.3, "b"), (0.5, "c")])},
            {"right": "fixed"},
        ],
    )
    trials = evaluated_trials(space, [{"branch": 0, "inner": 2}] * 4)
    domain = base.Domain(lambda value: 0.0, space)
    ids = list(range(4, 12))

    indices, values = anneal.suggest_batch(ids, domain, trials, seed=9)

    np.testing.assert_array_equal(indices["branch"], ids)
    left_ids = [tid for tid, value in zip(ids, values["branch"]) if value == 0]
    assert len(left_ids) > 1
    np.testing.assert_array_equal(indices["inner"], left_ids)
    for tid in ids:
        point = {
            label: value[list(indices[label]).index(tid)]
            for label, value in values.items()
            if tid in indices[label]
        }
        result = space_eval(space, point)
        assert result in [
            {"left": "a"},
            {"left": "b"},
            {"left": "c"},
            {"right": "fixed"},
        ]


def test_annealing_batch_without_history_still_draws_from_the_prior():
    space = hp.pchoice("choice", [(0.0, "a"), (0.0, "b"), (1.0, "c")])
    domain = base.Domain(lambda value: 0.0, space)

    indices, values = anneal.suggest_batch([0, 1, 2], domain, Trials(), seed=10)

    np.testing.assert_array_equal(indices["choice"], [0, 1, 2])
    np.testing.assert_array_equal(values["choice"], [2, 2, 2])
