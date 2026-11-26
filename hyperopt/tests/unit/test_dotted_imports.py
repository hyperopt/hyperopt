import collections
import collections.abc
import importlib
import math

import pytest

from hyperopt import utils


@pytest.mark.parametrize("name", ["math", "collections.abc", "os.path"])
def test_import_tokens_consumes_complete_module_name(name):
    module, remaining = utils.import_tokens(name.split("."))
    assert module is importlib.import_module(name)
    assert remaining == []


@pytest.mark.parametrize(
    "tokens,module,remaining",
    [
        (["math", "sin"], math, ["sin"]),
        (
            ["collections", "abc", "Mapping", "register"],
            collections.abc,
            ["Mapping", "register"],
        ),
    ],
)
def test_import_tokens_retains_attribute_suffix(tokens, module, remaining):
    actual_module, actual_remaining = utils.import_tokens(tokens)
    assert actual_module is module
    assert actual_remaining == remaining


@pytest.mark.parametrize(
    "name,expected",
    [
        ("math.sin", math.sin),
        ("collections.Counter", collections.Counter),
        ("collections.Counter.most_common", collections.Counter.most_common),
        ("collections.abc", collections.abc),
    ],
)
def test_json_lookup_resolves_modules_functions_and_nested_attributes(name, expected):
    assert utils.json_lookup(name) is expected


def test_json_call_passes_constructor_arguments():
    result = utils.json_call("collections.Counter", args=("abb",), kwargs={"c": 4})
    assert result == collections.Counter(a=1, b=2, c=4)


def test_get_obj_passes_pickled_constructor_arguments():
    arguments = utils.pickler.dumps({"args": ("abb",), "kwargs": {"c": 4}})
    result = utils.get_obj("collections.Counter", argstr=arguments)
    assert result == collections.Counter(a=1, b=2, c=4)


def test_missing_attribute_still_raises():
    with pytest.raises(AttributeError):
        utils.json_lookup("math.hyperopt_missing_attribute")
