from functools import partial
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
