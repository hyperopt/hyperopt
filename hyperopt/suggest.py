from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from builtins import object
import logging
import os

import numpy as np

from .utils import coarse_utcnow
from . import base
from . import tpe
from . import hp

from pandas import DataFrame
from datetime import datetime


class SuggestObject(object):
    """Object for conducting search experiments.
    """

    def __init__(self, algo, domain, trials, rstate):
        self.algo = algo
        self.domain = domain
        self.trials = trials
        self.rstate = rstate

    def serial_evaluate(self):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == base.JOB_STATE_NEW:
                result = dict(loss=None, status='ok')
                now = coarse_utcnow()
                trial['book_time'] = now
                trial['refresh_time'] = now
                trial['state'] = base.JOB_STATE_DONE
                trial['result'] = result
                trial['refresh_time'] = coarse_utcnow()
        self.trials.refresh()

    def run(self, num_trials):
        """
        :param num_trials: 
        :return: 
        """
        trials = self.trials
        algo = self.algo

        for j in range(num_trials):
            new_ids = trials.new_trial_ids(1)
            self.trials.refresh()
            new_trials = algo(new_ids, self.domain, trials,
                              self.rstate.randint(2 ** 31 - 1))
            self.trials.insert_trial_docs(new_trials)
            self.trials.refresh()

        self.serial_evaluate()


def dataframe_to_trials(data: DataFrame):
    trials = base.Trials()
    names = list(data)[:-1]
    for index, row in data.iterrows():
        misc = dict(tid=index, cmd=('domain_attachment', 'FMinIter_Domain'), workdir=None,
                    idxs=dict(zip(names, [[index]] * len(names))),
                    vals=dict(zip(names, [[each] for each in row.tolist()[:-1]])))  # last of row is loss
        trials._dynamic_trials.append(dict(
            state=2, tid=index, spec=None,
            result=dict(loss=row[-1], status='ok'),
            misc=misc, exp_key=None, owner=None, version=0,
            book_time=datetime.now(), refresh_time=datetime.now(),
        ))
    return trials


def trials_to_dataframe(trials: base.Trials):
    trials = trials._dynamic_trials
    if len(trials) == 0:
        return None
    names = list(trials[0]['misc']['vals'].keys())
    data = DataFrame(columns=names + ['loss'])
    for name in names:
        data[name] = [trial['misc']['vals'][name][0] for trial in trials]
    data['loss'] = [trial['result']['loss'] for trial in trials]
    return data


def suggest(data: DataFrame, space=None, num_trials=3):
    """
    :param data: dataframe, last column must be loss
    :param space: a list of range specification. example: [hp.uniform('x', -100, 100), hp.uniform('y', 0, 10)]
    :param num_trials:
    :return: a data frame with new suggested points at the end. losses are set to None for new points
    """
    algo = tpe.suggest

    if space is None:
        if len(data) > 0:
            # default range
            names = list(data)[:-1]
            space = [hp.uniform(name, -100, 100) for name in names]
        else:
            raise Exception("Space cannot be None if data is empty")

    trials = dataframe_to_trials(data)

    env_rseed = os.environ.get('HYPEROPT_FMIN_SEED', '')
    if env_rseed:
        rstate = np.random.RandomState(int(env_rseed))
    else:
        rstate = np.random.RandomState()

    domain = base.Domain(None, space, pass_expr_memo_ctrl=None)

    rval = SuggestObject(algo, domain, trials, rstate=rstate)
    rval.catch_eval_exceptions = True
    rval.run(num_trials)

    out_data = trials_to_dataframe(trials)

    return out_data
# -- flake8 doesn't like blank last line
