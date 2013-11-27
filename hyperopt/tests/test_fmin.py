import nose.tools

from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_FAIL, STATUS_OK


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


def test_quadratic1_anneal():
    trials = Trials()
    import hyperopt.anneal

    N = 20
    def fn(x):
        return (x - 3) ** 2

    argmin = fmin(
            fn=fn,
            space=hp.uniform('x', -5, 5),
            algo=hyperopt.anneal.suggest,
            max_evals=N,
            trials=trials)

    print argmin

    assert len(trials) == N
    assert abs(argmin['x'] - 3.0) < .25


@nose.tools.raises(exceptions.DuplicateLabel)
def test_duplicate_label_is_error():
    trials = Trials()

    def fn(xy):
        x, y = xy
        return x ** 2 + y ** 2

    fmin(fn=fn,
            space=[
                hp.uniform('x', -5, 5),
                hp.uniform('x', -5, 5),
                ],
            algo=rand.suggest,
            max_evals=500,
            trials=trials)


def test_space_eval():
    space = hp.choice('a',
        [
            ('case 1', 1 + hp.lognormal('c1', 0, 1)),
            ('case 2', hp.uniform('c2', -10, 10))
        ])

    assert space_eval(space, {'a': 0, 'c1': 1.0}) == ('case 1', 2.0)
    assert space_eval(space, {'a': 1, 'c2': 3.5}) == ('case 2', 3.5)


def test_status_fail_tpe():
    trials = Trials()

    argmin = fmin(
            fn=lambda x: ( {'loss': (x - 3) ** 2, 'status': STATUS_OK} if (x < 0) else
                           {'status': STATUS_FAIL}),
            space=hp.uniform('x', -5, 5),
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

    assert len(trials) == 50, len(trials)
    assert argmin['x'] < 0, argmin
    assert trials.best_trial['result'].has_key('loss'), trials.best_trial['result'].has_key('loss')
    assert trials.best_trial['result']['loss'] >= 9, trials.best_trial['result']['loss']
