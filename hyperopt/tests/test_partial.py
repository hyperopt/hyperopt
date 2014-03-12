#from functools import partial
from hyperopt.pyll.partial import partial, Literal
from hyperopt.pyll.partial import evaluate
from hyperopt.pyll.partial import depth_first_traversal
from hyperopt.pyll.partial import as_partialplus as as_pp


def test_arithmetic():
    def check(a, b):
        assert evaluate(as_pp(partial(int, a)) +
                        as_pp(partial(int, b))) == a + b
        assert evaluate(as_pp(partial(int, a)) -
                        as_pp(partial(int, b))) == a - b
        assert evaluate(as_pp(partial(int, a)) *
                        as_pp(partial(int, b))) == a * b
        assert evaluate(as_pp(partial(int, a)) /
                        as_pp(partial(int, b))) == a / b
        assert evaluate(as_pp(partial(int, a)) %
                        as_pp(partial(int, b))) == a % b
        assert evaluate(as_pp(partial(int, a)) |
                        as_pp(partial(int, b))) == a | b
        assert evaluate(as_pp(partial(int, a)) ^
                        as_pp(partial(int, b))) == a ^ b
        assert evaluate(as_pp(partial(int, a)) &
                        as_pp(partial(int, b))) == a & b
    yield check, 6, 5
    yield check, 4, 2
    yield check, 9, 11


def test_switch():
    """Test the "switch" program structure in partial.evaluate"""
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'
    # TODO: James: I opted for this behaviour rather than list(f, el1, el2...)
    # is there a compelling reason to do that? It kind of breaks with the
    # model.
    plist = as_pp([-1, partial(dont_eval)])
    assert -1 == evaluate(plist[0])


def test_switch_range():
    """Test that "switch" works on index ranges"""
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'
    plist = as_pp([-1, 0, 1, partial(dont_eval)])
    assert [-1, 0, 1] == evaluate(plist[:3])

    plist = as_pp((-1, 0, 1, partial(dont_eval)))
    assert (-1, 0, 1) == evaluate(plist[:3])


def test_arg():
    """Test basic partial.arg lookups"""
    def f(a, b=None):
        return -1

    assert partial(f, 0, 1).arg['a'] == Literal(0)
    assert partial(f, 0, 1).arg['b'] == Literal(1)

    assert partial(f, 0).arg['a'] == Literal(0)
    assert partial(f, 0).arg['b'] == Literal(None)

    assert partial(f, a=3).arg['a'] == Literal(3)
    assert partial(f, a=3).arg['b'] == Literal(None)

    assert partial(f, 2, b=5).arg['a'] == Literal(2)
    assert partial(f, 2, b=5).arg['b'] == Literal(5)

    assert partial(f, a=2, b=5).arg['a'] == Literal(2)
    assert partial(f, a=2, b=5).arg['b'] == Literal(5)


def test_star_args():
    """Test partial.arg lookups on *args"""
    def f(a, *b):
        return -1

    assert partial(f, 0, 1).arg['a'] == Literal(0)
    assert partial(f, 0, 1).arg['b'] == (Literal(1),)
    assert partial(f, 0, 1, 2, 3).arg['b'] == (Literal(1), Literal(2),
                                               Literal(3))


def test_kwargs():
    """Test partial.arg lookups on **kwargs"""
    def f(a, **b):
        return -1

    assert partial(f, 0, b=1).arg['a'] == Literal(0)
    assert partial(f, 0, b=1).arg['b'] == {'b': Literal(1)}
    assert partial(f, 0, foo=1, bar=2, baz=3).arg['b'] == {
        'foo': Literal(1),
        'bar': Literal(2),
        'baz': Literal(3),
    }


def test_star_kwargs():
    """Test partial.arg lookups on *args and **kwargs"""
    def f(a, *u, **b):
        return -1

    assert partial(f, 0, b=1).arg['a'] == Literal(0)
    assert partial(f, 0, b=1).arg['b'] == {'b': Literal(1)}

    assert partial(f, 0, 'q', 'uas', foo=1, bar=2).arg['a'] == Literal(0)
    assert partial(f, 0, 'q', 'uas', foo=1, bar=2).arg['u'] == (Literal('q'),
                                                                Literal('uas'))
    assert partial(f, 0, 'q', 'uas', foo=1, bar=2).arg['b'] == {
        'foo': Literal(1),
        'bar': Literal(2),
    }


def test_depth_first_traversal():
    def add(x, y):
        return x + y

    def index_of(l, e):
        """index() using is for comparison"""
        for i, v in enumerate(l):
            if v is e:
                return i

    p1 = partial(float, 5.0)
    p2 = partial(add, p1, 0.5)
    p3 = partial(add, p1, p2)
    p4 = partial(add, p2, p3)
    p5 = partial(int, p4)
    traversal = list(depth_first_traversal(p5))
    assert len(traversal) == 7
    assert index_of(traversal, p5) < index_of(traversal, p4)
    assert index_of(traversal, p4) < index_of(traversal, p3)
    assert index_of(traversal, p4) < index_of(traversal, p2)
    assert index_of(traversal, p2) < index_of(traversal, p1)


if __name__ == "__main__":
    test_switch()
    test_switch_range()
