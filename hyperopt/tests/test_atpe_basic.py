from __future__ import absolute_import
import unittest


class TestATPE(unittest.TestCase):
    def test_run_basic_search(self):
        def objective(args):
            case, val = args
            if case == "case 1":
                return val
            else:
                return val ** 2

        # define a search space
        from hyperopt import hp

        space = hp.choice(
            "a",
            [
                ("case 1", 1 + hp.lognormal("c1", 0, 1)),
                ("case 2", hp.uniform("c2", -10, 10)),
            ],
        )

        # minimize the objective over the space
        from hyperopt import fmin, atpe, space_eval

        best = fmin(objective, space, algo=atpe.suggest, max_evals=10)

        print(best)
        # -> {'a': 1, 'c2': 0.01420615366247227}
        print(space_eval(space, best))
        # -> ('case 2', 0.01420615366247227}
