from hyperopt.pyll.partial import PartialPlus, evaluate


def test_arithmetic():
    def check(a, b):
        assert evaluate(PartialPlus(int, a) + PartialPlus(int, b)) == a + b
        assert evaluate(PartialPlus(int, a) - PartialPlus(int, b)) == a - b
        assert evaluate(PartialPlus(int, a) * PartialPlus(int, b)) == a * b
        assert evaluate(PartialPlus(int, a) / PartialPlus(int, b)) == a / b
        assert evaluate(PartialPlus(int, a) % PartialPlus(int, b)) == a % b
        assert evaluate(PartialPlus(int, a) | PartialPlus(int, b)) == a | b
        assert evaluate(PartialPlus(int, a) ^ PartialPlus(int, b)) == a ^ b
        assert evaluate(PartialPlus(int, a) & PartialPlus(int, b)) == a & b
    yield check, 6, 5
    yield check, 4, 2
    yield check, 9, 11


def test_switch():
    """Test the "switch" program structure in partial.evaluate"""
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'
    plist = PartialPlus(list, -1, PartialPlus(dont_eval))
    assert -1 == evaluate(plist[0])


def test_switch_range():
    """Test that "switch" works on index ranges"""
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'
    plist = PartialPlus(list, -1, 0, 1, PartialPlus(dont_eval))
    assert [-1, 0, 1] == evaluate(plist[:3])

    plist = PartialPlus(tuple, -1, 0, 1, PartialPlus(dont_eval))
    assert (-1, 0, 1) == evaluate(plist[:3])


def test_arg():
    """Test basic PartialPlus.arg lookups"""
    def f(a, b=None):
        return -1

    assert PartialPlus(f, 0, 1).arg['a'] == 0
    assert PartialPlus(f, 0, 1).arg['b'] == 1

    assert PartialPlus(f, 0).arg['a'] == 0
    assert PartialPlus(f, 0).arg['b'] == None

    assert PartialPlus(f, a=3).arg['a'] == 3
    assert PartialPlus(f, a=3).arg['b'] == None

    assert PartialPlus(f, 2, b=5).arg['a'] == 2
    assert PartialPlus(f, 2, b=5).arg['b'] == 5

    assert PartialPlus(f, a=2, b=5).arg['a'] == 2
    assert PartialPlus(f, a=2, b=5).arg['b'] == 5


def test_star_args():
    """Test PartialPlus.arg lookups on *args"""
    def f(a, *b):
        return -1

    assert PartialPlus(f, 0, 1).arg['a'] == 0
    assert PartialPlus(f, 0, 1).arg['b'] == [1]
    assert PartialPlus(f, 0, 1, 2, 3).arg['b'] == [1, 2, 3]


def test_kwargs():
    """Test PartialPlus.arg lookups on **kwargs"""
    def f(a, **b):
        return -1

    assert PartialPlus(f, 0, b=1).arg['a'] == 0
    assert PartialPlus(f, 0, b=1).arg['b'] == {'b': 1}
    assert PartialPlus(f, 0, foo=1, bar=2, baz=3).arg['b'] == {
        'foo': 1,
        'bar': 2,
        'baz': 3,
    }

def test_star_kwargs():
    """Test PartialPlus.arg lookups on *args and **kwargs"""
    def f(a, *u, **b):
        return -1

    assert PartialPlus(f, 0, b=1).arg['a'] == 0
    assert PartialPlus(f, 0, b=1).arg['b'] == {'b': 1}

    assert PartialPlus(f, 0, 'q', 'uas', foo=1, bar=2).arg['a'] == 0
    assert PartialPlus(f, 0, 'q', 'uas', foo=1, bar=2).arg['u'] == [
        'q', 'uas']
    assert PartialPlus(f, 0, 'q', 'uas', foo=1, bar=2).arg['b'] == {
        'foo': 1,
        'bar': 2,
    }
