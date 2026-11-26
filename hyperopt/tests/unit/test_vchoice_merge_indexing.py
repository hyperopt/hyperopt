import numpy as np
import pytest

from hyperopt import hp, space_eval
from hyperopt.pyll import as_apply, rec_eval, scope
from hyperopt.pyll.stochastic import recursive_set_rng_kwarg
from hyperopt.vectorize import VectorizeHelper, vchoice_merge


class CountingIds(list):
    def __init__(self, values):
        super().__init__(values)
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


def test_each_selected_branch_is_indexed_once():
    first = CountingIds([7, 2, 9])
    second = CountingIds([3, 8])
    assert vchoice_merge(
        [9, 8, 2, 3, 7], [0, 1, 0, 1, 0], (first, [70, 20, 90]), (second, [30, 80])
    ) == [90, 80, 20, 30, 70]
    assert first.scans == second.scans == 1


def test_duplicate_branch_ids_keep_the_first_value():
    assert vchoice_merge([2, 1, 2], [0, 0, 0], ([1, 2, 2], [10, 20, 99])) == [
        20,
        10,
        20,
    ]


def test_string_ids_preserve_value_references_and_output_order():
    left, right = {"value": [1]}, {"value": [2]}
    result = vchoice_merge(
        ["b", "a"], [1, 0], (["a"], [left]), (["b"], [right]), ([], [])
    )
    assert result[0] is right
    assert result[1] is left


def test_empty_output_and_unused_branches():
    ids = CountingIds([1, 2])
    assert vchoice_merge([], [], (ids, [10, 20])) == []
    assert ids.scans == 0


def test_missing_branch_id_still_raises_value_error():
    with pytest.raises(ValueError):
        vchoice_merge([2], [0], ([1], [10]))


@pytest.mark.parametrize("seed", [5, 17, 42])
def test_nested_vectorized_choices_match_scalar_space_evaluation(seed):
    space = hp.choice(
        "branch",
        [
            hp.uniform("left", -2, -1),
            hp.choice("inner", [3.0, hp.uniform("right", 1, 2)]),
        ],
    )
    helper = VectorizeHelper(space, scope.range(32), build=True)
    graph = as_apply([helper.v_expr, helper.idxs_by_label(), helper.vals_by_label()])
    graph = recursive_set_rng_kwarg(graph, as_apply(np.random.default_rng(seed)))
    values, ids, parameters = rec_eval(graph)
    configurations = [dict() for _ in values]
    for label, trial_ids in ids.items():
        for tid, value in zip(trial_ids, parameters[label]):
            configurations[tid][label] = value
    expected = [space_eval(space, config) for config in configurations]
    np.testing.assert_array_equal(values, expected)
