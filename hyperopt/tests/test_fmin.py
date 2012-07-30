from hyperopt.pyll_utils import hp_uniform

from hyperopt import fmin, rand, tpe


def test_quadratic1_rand():

    report = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp_uniform('x', -5, 5),
            algo=rand.suggest,
            max_evals=500)

    assert len(report.trials) == 500
    assert abs(report.trials.argmin['x'] - 3.0) < .25


def test_quadratic1_tpe():

    report = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp_uniform('x', -5, 5),
            algo=tpe.suggest,
            max_evals=500)

    assert len(report.trials) == 500
    assert abs(report.trials.argmin['x'] - 3.0) < .25
