from hyperopt import pyll_utils
from hyperopt.pyll_utils import EQ
from hyperopt.pyll_utils import expr_to_config
from hyperopt import hp
from hyperopt.pyll import as_apply
from hyperopt.pyll.stochastic import sample
import unittest
import numpy as np
import pytest


def test_expr_to_config():
    z = hp.randint("z", 10)
    a = hp.choice(
        "a",
        [
            hp.uniform("b", -1, 1) + z,
            {
                "c": 1,
                "d": hp.choice(
                    "d", [3 + hp.loguniform("c", 0, 1), 1 + hp.loguniform("e", 0, 1)]
                ),
            },
        ],
    )

    expr = as_apply((a, z))

    hps = {}
    expr_to_config(expr, (True,), hps)

    for label, dct in list(hps.items()):
        print(label)
        print(
            "  dist: %s(%s)"
            % (
                dct["node"].name,
                ", ".join(map(str, [ii.eval() for ii in dct["node"].inputs()])),
            )
        )
        if len(dct["conditions"]) > 1:
            print("  conditions (OR):")
            for condseq in dct["conditions"]:
                print("    ", " AND ".join(map(str, condseq)))
        elif dct["conditions"]:
            for condseq in dct["conditions"]:
                print("  conditions :", " AND ".join(map(str, condseq)))

    assert hps["a"]["node"].name == "randint"
    assert hps["b"]["node"].name == "uniform"
    assert hps["c"]["node"].name == "loguniform"
    assert hps["d"]["node"].name == "randint"
    assert hps["e"]["node"].name == "loguniform"
    assert hps["z"]["node"].name == "randint"

    assert {(True, EQ("a", 0))} == {(True, EQ("a", 0))}
    assert hps["a"]["conditions"] == {(True,)}
    assert hps["b"]["conditions"] == {(True, EQ("a", 0))}, hps["b"]["conditions"]
    assert hps["c"]["conditions"] == {(True, EQ("a", 1), EQ("d", 0))}
    assert hps["d"]["conditions"] == {(True, EQ("a", 1))}
    assert hps["e"]["conditions"] == {(True, EQ("a", 1), EQ("d", 1))}
    assert hps["z"]["conditions"] == {(True,), (True, EQ("a", 0))}


def test_remove_allpaths():
    z = hp.uniform("z", 0, 10)
    a = hp.choice("a", [z + 1, z - 1])
    hps = {}
    expr_to_config(a, (True,), hps)
    aconds = hps["a"]["conditions"]
    zconds = hps["z"]["conditions"]
    assert aconds == {(True,)}, aconds
    assert zconds == {(True,)}, zconds


def test_remove_allpaths_int():
    z = hp.uniformint("z", 0, 10)
    a = hp.choice("a", [z + 1, z - 1])
    hps = {}
    expr_to_config(a, (True,), hps)
    aconds = hps["a"]["conditions"]
    zconds = hps["z"]["conditions"]
    assert aconds == {(True,)}, aconds
    assert zconds == {(True,)}, zconds


@pyll_utils.validate_distribution_range
def stub_pyll_fn(label, low, high):
    """
    Stub function to test distribution range validation fn
    """
    pass


@pytest.mark.parametrize(
    "arguments", [["z", 0, 10], {"label": "z", "low": 0, "high": 10}]
)
def test_uniformint_arguments(arguments):
    """
    Test whether uniformint can accept both positional and keyword arguments.
    Related to PR #704.
    """
    if isinstance(arguments, list):
        space = hp.uniformint(*arguments)
    if isinstance(arguments, dict):
        space = hp.uniformint(**arguments)
    rng = np.random.default_rng(np.random.PCG64(123))
    values = [sample(space, rng=rng) for _ in range(10)]
    assert values == [7, 1, 2, 2, 2, 8, 9, 3, 8, 9]


class TestValidateDistributionRange(unittest.TestCase):
    """
    We can't test low being set via kwarg while high is set via arg because
    that's not a validate fn call
    """

    def test_raises_error_for_low_arg_high_arg(self):
        self.assertRaises(ValueError, stub_pyll_fn, "stub", 1, 1)

    def test_raises_error_for_low_arg_high_kwarg(self):
        self.assertRaises(ValueError, stub_pyll_fn, "stub", 1, high=1)

    def test_raises_error_for_low_kwarg_high_kwarg(self):
        self.assertRaises(ValueError, stub_pyll_fn, "stub", low=1, high=1)


class TestDistributionsWithRangeValidateBoundries(unittest.TestCase):
    def test_hp_uniform_raises_error_when_range_is_zero(self):
        self.assertRaises(ValueError, hp.uniform, "stub", 10, 10)

    def test_hp_quniform_raises_error_when_range_is_zero(self):
        self.assertRaises(ValueError, hp.quniform, "stub", 10, 10, 1)

    def test_hp_loguniform_raises_error_when_range_is_zero(self):
        self.assertRaises(ValueError, hp.loguniform, "stub", 10, 10, 1)

    def test_hp_qloguniform_raises_error_when_range_is_zero(self):
        self.assertRaises(ValueError, hp.qloguniform, "stub", 10, 10, 1)
