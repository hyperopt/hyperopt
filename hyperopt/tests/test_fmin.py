
from hyperopt import fmin, rand, tpe, hp, Trials


def test_quadratic1_rand():
    trials = Trials()

    argmin = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp.uniform('x', -5, 5),
            algo=rand.suggest,
            max_evals=500,
            trials=trials)

    assert len(trials) == 500
    assert abs(argmin['x'] - 3.0) < .25


def test_quadratic1_tpe():
    trials = Trials()

    argmin = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp.uniform('x', -5, 5),
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

    assert len(trials) == 50, len(trials)
    assert abs(argmin['x'] - 3.0) < .25, argmin
