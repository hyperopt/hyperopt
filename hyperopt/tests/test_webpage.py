from __future__ import print_function


def test_landing_screen():

    # define an objective function
    def objective(args):
        case, val = args
        if case == 'case 1':
            return val
        else:
            return val ** 2

    # define a search space
    from hyperopt import hp
    space = hp.choice('a',
                      [
                          ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                          ('case 2', hp.uniform('c2', -10, 10))
                      ])

    # minimize the objective over the space
    import hyperopt
    best = hyperopt.fmin(objective, space,
                         algo=hyperopt.tpe.suggest,
                         max_evals=100)

    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}

    print(hyperopt.space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}
