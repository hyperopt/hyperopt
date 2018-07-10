from hyperopt import fmin, tpe, hp
from hyperopt.tools import generate_trials_to_calculate
import unittest

class TestGenerateTrialsToCalculate(unittest.TestCase):
    def test_generate_trials_to_calculate(self):
        points = [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}]
        trials = generate_trials_to_calculate(points)
        best = fmin(fn=lambda space: space['x']**2 + space['y']**2,
                    space={'x': hp.uniform('x', -10, 10),
                           'y': hp.uniform('y', -10, 10)},
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials,
                    )
        assert best['x'] == 0.0 and best['y'] == 0.0
