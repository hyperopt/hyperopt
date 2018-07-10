from hyperopt.base import Trials, JOB_STATE_NEW


def _generate_trial(tid, space):
    variables = space.keys()
    idxs = {v: [tid] for v in variables}
    vals = {k: [v] for k, v in space.items()}
    return {'state': JOB_STATE_NEW,
            'tid': tid,
            'spec': None,
            'result': {'status': 'new'},
            'misc': {'tid': tid,
                     'cmd': ('domain_attachment',
                             'FMinIter_Domain'),
                     'workdir': None,
                     'idxs': idxs,
                     'vals': vals},
            'exp_key': None,
            'owner': None,
            'version': 0,
            'book_time': None,
            'refresh_time': None,
            }


def generate_trials_to_calculate(points):
    """
    :param points: List of points to be inserted in trials object in form of
                    dictionary with variable names as keys and variable values
                     as dict values. Example code:

    points = [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}]
    trials = generate_trials_to_calculate(points)
    best = fmin(fn=lambda space: space['x']**2 + space['y']**2,
                space={'x': hp.uniform('x', -10, 10),
                       'y': hp.uniform('y', -10, 10)},
                algo=tpe.suggest,
                max_evals=10,
                trials=trials,
                )
    :return: object of class base.Trials() with points which will be calculated
                before optimisation start if passed to fmin().
    """
    trials = Trials()
    new_trials = [_generate_trial(tid, x) for tid, x in enumerate(points)]
    trials.insert_trial_docs(new_trials)
    return trials
