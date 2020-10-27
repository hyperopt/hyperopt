from hyperopt.pyll import base
from hyperopt.pyll.base import (
    Literal,
    as_apply,
    Apply,
    dfs,
    scope,
    rec_eval,
    p0,
    Lambda,
    clone_merge,
)

from nose import SkipTest
from nose.tools import assert_raises
import numpy as np


def test_literal_pprint():
    l = Literal(5)
    print(str(l))
    assert str(l) == "0 Literal{5}"


def test_literal_apply():
    l0 = Literal([1, 2, 3])
    print(str(l0))
    assert str(l0) == "0 Literal{[1, 2, 3]}"


def test_literal_unpacking():
    l0 = Literal([1, 2, 3])
    a, b, c = l0
    print(a)
    assert c.name == "getitem"
    assert c.pos_args[0] is l0
    assert isinstance(c.pos_args[1], Literal)
    assert c.pos_args[1]._obj == 2


def test_as_apply_passthrough():
    a4 = as_apply(4)
    assert a4 is as_apply(a4)


def test_as_apply_literal():
    assert isinstance(as_apply(7), Literal)


def test_as_apply_list_of_literals():
    l = [9, 3]
    al = as_apply(l)
    assert isinstance(al, Apply)
    assert al.name == "pos_args"
    assert isinstance(al.pos_args[0], Literal)
    assert isinstance(al.pos_args[1], Literal)
    al.pos_args[0]._obj == 9
    al.pos_args[1]._obj == 3


def test_as_apply_tuple_of_literals():
    l = (9, 3)
    al = as_apply(l)
    assert isinstance(al, Apply)
    assert al.name == "pos_args"
    assert isinstance(al.pos_args[0], Literal)
    assert isinstance(al.pos_args[1], Literal)
    al.pos_args[0]._obj == 9
    al.pos_args[1]._obj == 3
    assert len(al) == 2


def test_as_apply_list_of_applies():
    alist = [as_apply(i) for i in range(5)]

    al = as_apply(alist)
    assert isinstance(al, Apply)
    assert al.name == "pos_args"
    # -- have to come back to this if Literal copies args
    assert al.pos_args == alist


def test_as_apply_dict_of_literals():
    d = {"a": 9, "b": 10}
    ad = as_apply(d)
    assert isinstance(ad, Apply)
    assert ad.name == "dict"
    assert len(ad) == 2
    assert ad.named_args[0][0] == "a"
    assert ad.named_args[0][1]._obj == 9
    assert ad.named_args[1][0] == "b"
    assert ad.named_args[1][1]._obj == 10


def test_as_apply_dict_of_applies():
    d = {"a": as_apply(9), "b": as_apply(10)}
    ad = as_apply(d)
    assert isinstance(ad, Apply)
    assert ad.name == "dict"
    assert len(ad) == 2
    assert ad.named_args[0][0] == "a"
    assert ad.named_args[0][1]._obj == 9
    assert ad.named_args[1][0] == "b"
    assert ad.named_args[1][1]._obj == 10


def test_as_apply_nested_dict():
    d = {"a": 9, "b": {"c": 11, "d": 12}}
    ad = as_apply(d)
    assert isinstance(ad, Apply)
    assert ad.name == "dict"
    assert len(ad) == 2
    assert ad.named_args[0][0] == "a"
    assert ad.named_args[0][1]._obj == 9
    assert ad.named_args[1][0] == "b"
    assert ad.named_args[1][1].name == "dict"
    assert ad.named_args[1][1].named_args[0][0] == "c"
    assert ad.named_args[1][1].named_args[0][1]._obj == 11
    assert ad.named_args[1][1].named_args[1][0] == "d"
    assert ad.named_args[1][1].named_args[1][1]._obj == 12


def test_dfs():
    dd = as_apply({"c": 11, "d": 12})

    d = {"a": 9, "b": dd, "y": dd, "z": dd + 1}
    ad = as_apply(d)
    order = dfs(ad)
    print([str(o) for o in order])
    assert order[0]._obj == 9
    assert order[1]._obj == 11
    assert order[2]._obj == 12
    assert order[3].named_args[0][0] == "c"
    assert order[4]._obj == 1
    assert order[5].name == "add"
    assert order[6].named_args[0][0] == "a"
    assert len(order) == 7


@scope.define_info(o_len=2)
def _test_foo():
    return 1, 2


def test_o_len():
    obj = scope._test_foo()
    x, y = obj
    assert x.name == "getitem"
    assert x.pos_args[1]._obj == 0
    assert y.pos_args[1]._obj == 1


def test_eval_arithmetic():
    a, b, c = as_apply((2, 3, 4))

    assert (a + b).eval() == 5
    assert (a + b + c).eval() == 9
    assert (a + b + 1 + c).eval() == 10

    assert (a * b).eval() == 6
    assert (a * b * c * (-1)).eval() == -24

    assert (a - b).eval() == -1
    assert (a - b * c).eval() == -10

    assert (a // b).eval() == 0  # int div
    assert (b // a).eval() == 1  # int div
    assert (c / a).eval() == 2
    assert (4 / a).eval() == 2
    assert (a / 4.0).eval() == 0.5


def test_bincount():
    def test_f(f):
        r = np.arange(10)
        counts = f(r)
        assert isinstance(counts, np.ndarray)
        assert len(counts) == 10
        assert np.all(counts == 1)

        r = np.arange(10) + 3
        counts = f(r)
        assert isinstance(counts, np.ndarray)
        assert len(counts) == 13
        assert np.all(counts[3:] == 1)
        assert np.all(counts[:3] == 0)

        r = np.arange(10) + 3
        counts = f(r, minlength=5)  # -- ignore minlength
        assert isinstance(counts, np.ndarray)
        assert len(counts) == 13
        assert np.all(counts[3:] == 1)
        assert np.all(counts[:3] == 0)

        r = np.arange(10) + 3
        counts = f(r, minlength=15)  # -- pad to minlength
        assert isinstance(counts, np.ndarray)
        assert len(counts) == 15
        assert np.all(counts[:3] == 0)
        assert np.all(counts[3:13] == 1)
        assert np.all(counts[13:] == 0)

        r = np.arange(10) % 3 + 3
        counts = f(r, minlength=7)  # -- pad to minlength
        assert list(counts) == [0, 0, 0, 4, 3, 3, 0]

    try:
        test_f(base.bincount)
    except TypeError as e:
        if "function takes at most 2 arguments" in str(e):
            raise SkipTest()
        raise


def test_switch_and_Raise():
    i = Literal()
    ab = scope.switch(i, "a", "b", scope.Raise(Exception))
    assert rec_eval(ab, memo={i: 0}) == "a"
    assert rec_eval(ab, memo={i: 1}) == "b"
    assert_raises(Exception, rec_eval, ab, memo={i: 2})


def test_kwswitch():
    i = Literal()
    ab = scope.kwswitch(i, k1="a", k2="b", err=scope.Raise(Exception))
    assert rec_eval(ab, memo={i: "k1"}) == "a"
    assert rec_eval(ab, memo={i: "k2"}) == "b"
    assert_raises(Exception, rec_eval, ab, memo={i: "err"})


def test_recursion():
    scope.define(
        Lambda(
            "Fact",
            [("x", p0)],
            expr=scope.switch(p0 > 1, 1, p0 * base.apply("Fact", p0 - 1)),
        )
    )
    print(scope.Fact(3))
    assert rec_eval(scope.Fact(3)) == 6


def test_partial():
    add2 = scope.partial("add", 2)
    print(add2)
    assert len(str(add2).split("\n")) == 3

    # add2 evaluates to a scope method
    thing = rec_eval(add2)
    print(thing)
    assert "SymbolTableEntry" in str(thing)

    # add2() evaluates to a failure because it's only a partial application
    assert_raises(NotImplementedError, rec_eval, add2())

    # add2(3) evaluates to 5 because we've filled in all the blanks
    thing = rec_eval(add2(3))
    print(thing)
    assert thing == 5


def test_callpipe():

    # -- set up some 1-variable functions
    a2 = scope.partial("add", 2)
    a3 = scope.partial("add", 3)

    def s9(a):
        return scope.sub(a, 9)

    # x + 2 + 3 - 9 == x - 4
    r = scope.callpipe1([a2, a3, s9], 5)
    thing = rec_eval(r)
    assert thing == 1


def test_clone_merge():
    a, b, c = as_apply((2, 3, 2))
    d = (a + b) * (c + b)
    len_d = len(dfs(d))

    e = clone_merge(d, merge_literals=True)
    assert len_d == len(dfs(d))
    assert len_d > len(dfs(e))
    assert e.eval() == d.eval()


def test_clone_merge_no_merge_literals():
    a, b, c = as_apply((2, 3, 2))
    d = (a + b) * (c + b)
    len_d = len(dfs(d))
    e = clone_merge(d, merge_literals=False)
    assert len_d == len(dfs(d))
    assert len_d == len(dfs(e))
    assert e.eval() == d.eval()


def test_len():
    assert_raises(TypeError, len, scope.uniform(0, 1))
