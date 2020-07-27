from __future__ import absolute_import
from hyperopt import hp, fmin, atpe, space_eval


class TestATPE:
    def test_run_basic_search(self):
        def objective(args):
            case, val = args
            if case == "case 2":
                return val
            else:
                return val ** 2

        # define a search space
        space = hp.choice(
            "a",
            [
                ("case 1", 1 + hp.lognormal("c1", 0, 1)),
                ("case 2", hp.uniform("c2", -10, 10)),
            ],
        )

        # minimize the objective over the space
        # NOTE: Max evals should be greater than 10, as the first 10 runs are only the initialization rounds
        best = fmin(objective, space, algo=atpe.suggest, max_evals=20)

        # Assert that case 2 was the best choice
        assert best["a"] == 1
        assert space_eval(space, best)[0] == "case 2"
