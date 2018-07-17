from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from builtins import str
from builtins import object

import functools
import logging
import os
import sys
import time

import numpy as np

from . import pyll
from .utils import coarse_utcnow
from . import base

standard_library.install_aliases()
logger = logging.getLogger(__name__)


try:
    import dill as pickler
except Exception as e:
    logger.info('Failed to load dill, try installing dill via "pip install dill" for enhanced pickling support.')
    import six.moves.cPickle as pickler


def generate_trial(tid, space):
    variables = space.keys()
    idxs = {v: [tid] for v in variables}
    vals = {k: [v] for k, v in space.items()}
    return {'state': base.JOB_STATE_NEW,
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
    Function that generates trials to be evaluated from list of points

    :param points: List of points to be inserted in trials object in form of
        dictionary with variable names as keys and variable values as dict
        values. Example value:
        [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}]

    :return: object of class base.Trials() with points which will be calculated
        before optimisation start if passed to fmin().
    """
    trials = base.Trials()
    new_trials = [generate_trial(tid, x) for tid, x in enumerate(points)]
    trials.insert_trial_docs(new_trials)
    return trials


def fmin_pass_expr_memo_ctrl(f):
    """
    Mark a function as expecting kwargs 'expr', 'memo' and 'ctrl' from
    hyperopt.fmin.

    expr - the pyll expression of the search space
    memo - a partially-filled memo dictionary such that
           `rec_eval(expr, memo=memo)` will build the proposed trial point.
    ctrl - the Experiment control object (see base.Ctrl)

    """
    f.fmin_pass_expr_memo_ctrl = True
    return f


def partial(fn, **kwargs):
    """functools.partial work-alike for functions decorated with
    fmin_pass_expr_memo_ctrl
    """
    rval = functools.partial(fn, **kwargs)
    if hasattr(fn, 'fmin_pass_expr_memo_ctrl'):
        rval.fmin_pass_expr_memo_ctrl = fn.fmin_pass_expr_memo_ctrl
    return rval


class FMinIter(object):
    """Object for conducting search experiments.
    """
    catch_eval_exceptions = False
    pickle_protocol = -1

    def __init__(self, algo, domain, trials, rstate, asynchronous=None,
                 max_queue_len=1,
                 poll_interval_secs=1.0,
                 max_evals=sys.maxsize,
                 verbose=0,
                 ):
        self.algo = algo
        self.domain = domain
        self.trials = trials
        if asynchronous is None:
            self.asynchronous = trials.asynchronous
        else:
            self.asynchronous = asynchronous
        self.poll_interval_secs = poll_interval_secs
        self.max_queue_len = max_queue_len
        self.max_evals = max_evals
        self.rstate = rstate

        if self.asynchronous:
            if 'FMinIter_Domain' in trials.attachments:
                logger.warn('over-writing old domain trials attachment')
            msg = pickler.dumps(domain)
            # -- sanity check for unpickling
            pickler.loads(msg)
            trials.attachments['FMinIter_Domain'] = msg

    def serial_evaluate(self, N=-1):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == base.JOB_STATE_NEW:
                trial['state'] == base.JOB_STATE_RUNNING
                now = coarse_utcnow()
                trial['book_time'] = now
                trial['refresh_time'] = now
                spec = base.spec_from_misc(trial['misc'])
                ctrl = base.Ctrl(self.trials, current_trial=trial)
                try:
                    result = self.domain.evaluate(spec, ctrl)
                except Exception as e:
                    logger.info('job exception: %s' % str(e))
                    trial['state'] = base.JOB_STATE_ERROR
                    trial['misc']['error'] = (str(type(e)), str(e))
                    trial['refresh_time'] = coarse_utcnow()
                    if not self.catch_eval_exceptions:
                        # -- JOB_STATE_ERROR means this trial
                        #    will be removed from self.trials.trials
                        #    by this refresh call.
                        self.trials.refresh()
                        raise
                else:
                    trial['state'] = base.JOB_STATE_DONE
                    trial['result'] = result
                    trial['refresh_time'] = coarse_utcnow()
                N -= 1
                if N == 0:
                    break
        self.trials.refresh()

    def block_until_done(self):
        already_printed = False
        if self.asynchronous:
            unfinished_states = [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]

            def get_queue_len():
                return self.trials.count_by_state_unsynced(unfinished_states)

            qlen = get_queue_len()
            while qlen > 0:
                if not already_printed:
                    logger.info('Waiting for %d jobs to finish ...' % qlen)
                    already_printed = True
                time.sleep(self.poll_interval_secs)
                qlen = get_queue_len()
            self.trials.refresh()
        else:
            self.serial_evaluate()

    def run(self, N, block_until_done=True):
        """
        block_until_done  means that the process blocks until ALL jobs in
        trials are not in running or new state

        """
        trials = self.trials
        algo = self.algo
        n_queued = 0

        def get_queue_len():
            return self.trials.count_by_state_unsynced(base.JOB_STATE_NEW)

        stopped = False
        while n_queued < N:
            qlen = get_queue_len()
            while qlen < self.max_queue_len and n_queued < N:
                n_to_enqueue = min(self.max_queue_len - qlen, N - n_queued)
                new_ids = trials.new_trial_ids(n_to_enqueue)
                self.trials.refresh()
                if 0:
                    for d in self.trials.trials:
                        print('trial %i %s %s' % (d['tid'], d['state'],
                                                  d['result'].get('status')))
                new_trials = algo(new_ids, self.domain, trials,
                                  self.rstate.randint(2 ** 31 - 1))
                assert len(new_ids) >= len(new_trials)
                if len(new_trials):
                    self.trials.insert_trial_docs(new_trials)
                    self.trials.refresh()
                    n_queued += len(new_trials)
                    qlen = get_queue_len()
                else:
                    stopped = True
                    break

            if self.asynchronous:
                # -- wait for workers to fill in the trials
                time.sleep(self.poll_interval_secs)
            else:
                # -- loop over trials and do the jobs directly
                self.serial_evaluate()

            if stopped:
                break

        if block_until_done:
            self.block_until_done()
            self.trials.refresh()
            logger.info('Queue empty, exiting run.')
        else:
            qlen = get_queue_len()
            if qlen:
                msg = 'Exiting run, not waiting for %d jobs.' % qlen
                logger.info(msg)

    def __iter__(self):
        return self

    def __next__(self):
        self.run(1, block_until_done=self.asynchronous)
        if len(self.trials) >= self.max_evals:
            raise StopIteration()
        return self.trials

    def exhaust(self):
        n_done = len(self.trials)
        self.run(self.max_evals - n_done, block_until_done=self.asynchronous)
        self.trials.refresh()
        return self


def fmin(fn, space, algo, max_evals, trials=None, rstate=None,
         allow_trials_fmin=True, pass_expr_memo_ctrl=None,
         catch_eval_exceptions=False,
         verbose=0,
         return_argmin=True,
         points_to_evaluate=None
         ):
    """Minimize a function over a hyperparameter space.

    More realistically: *explore* a function over a hyperparameter space
    according to a given algorithm, allowing up to a certain number of
    function evaluations.  As points are explored, they are accumulated in
    `trials`


    Parameters
    ----------

    fn : callable (trial point -> loss)
        This function will be called with a value generated from `space`
        as the first and possibly only argument.  It can return either
        a scalar-valued loss, or a dictionary.  A returned dictionary must
        contain a 'status' key with a value from `STATUS_STRINGS`, must
        contain a 'loss' key if the status is `STATUS_OK`. Particular
        optimization algorithms may look for other keys as well.  An
        optional sub-dictionary associated with an 'attachments' key will
        be removed by fmin its contents will be available via
        `trials.trial_attachments`. The rest (usually all) of the returned
        dictionary will be stored and available later as some 'result'
        sub-dictionary within `trials.trials`.

    space : hyperopt.pyll.Apply node
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).

    algo : search algorithm
        This object, such as `hyperopt.rand.suggest` and
        `hyperopt.tpe.suggest` provides logic for sequential search of the
        hyperparameter space.

    max_evals : int
        Allow up to this many function evaluations before returning.

    trials : None or base.Trials (or subclass)
        Storage for completed, ongoing, and scheduled evaluation points.  If
        None, then a temporary `base.Trials` instance will be created.  If
        a trials object, then that trials object will be affected by
        side-effect of this call.

    rstate : numpy.RandomState, default numpy.random or `$HYPEROPT_FMIN_SEED`
        Each call to `algo` requires a seed value, which should be different
        on each call. This object is used to draw these seeds via `randint`.
        The default rstate is
        `numpy.random.RandomState(int(env['HYPEROPT_FMIN_SEED']))`
        if the `HYPEROPT_FMIN_SEED` environment variable is set to a non-empty
        string, otherwise np.random is used in whatever state it is in.

    verbose : int
        Print out some information to stdout during search.

    allow_trials_fmin : bool, default True
        If the `trials` argument

    pass_expr_memo_ctrl : bool, default False
        If set to True, `fn` will be called in a different more low-level
        way: it will receive raw hyperparameters, a partially-populated
        `memo`, and a Ctrl object for communication with this Trials
        object.

    return_argmin : bool, default True
        If set to False, this function returns nothing, which can be useful
        for example if it is expected that `len(trials)` may be zero after
        fmin, and therefore `trials.argmin` would be undefined.

    points_to_evaluate : list, default None
        Only works if trials=None. If points_to_evaluate equals None then the
        trials are evaluated normally. If list of dicts is passed then
        given points are evaluated before optimisation starts, so the overall
        number of optimisation steps is len(points_to_evaluate) + max_evals.
        Elements of this list must be in a form of a dictionary with variable
        names as keys and variable values as dict values. Example
        points_to_evaluate value is [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 2.0}]

    Returns
    -------

    argmin : None or dictionary
        If `return_argmin` is False, this function returns nothing.
        Otherwise, it returns `trials.argmin`.  This argmin can be converted
        to a point in the configuration space by calling
        `hyperopt.space_eval(space, best_vals)`.


    """
    if rstate is None:
        env_rseed = os.environ.get('HYPEROPT_FMIN_SEED', '')
        if env_rseed:
            rstate = np.random.RandomState(int(env_rseed))
        else:
            rstate = np.random.RandomState()

    if allow_trials_fmin and hasattr(trials, 'fmin'):
        return trials.fmin(
            fn, space,
            algo=algo,
            max_evals=max_evals,
            rstate=rstate,
            pass_expr_memo_ctrl=pass_expr_memo_ctrl,
            verbose=verbose,
            catch_eval_exceptions=catch_eval_exceptions,
            return_argmin=return_argmin,
        )

    if trials is None:
        if points_to_evaluate is None:
            trials = base.Trials()
        else:
            assert type(points_to_evaluate) == list
            trials = generate_trials_to_calculate(points_to_evaluate)

    domain = base.Domain(fn, space,
                         pass_expr_memo_ctrl=pass_expr_memo_ctrl)

    rval = FMinIter(algo, domain, trials, max_evals=max_evals,
                    rstate=rstate,
                    verbose=verbose)
    rval.catch_eval_exceptions = catch_eval_exceptions
    rval.exhaust()
    if return_argmin:
        return trials.argmin


def space_eval(space, hp_assignment):
    """Compute a point in a search space from a hyperparameter assignment.

    Parameters:
    -----------
    space - a pyll graph involving hp nodes (see `pyll_utils`).

    hp_assignment - a dictionary mapping hp node labels to values.
    """
    space = pyll.as_apply(space)
    nodes = pyll.toposort(space)
    memo = {}
    for node in nodes:
        if node.name == 'hyperopt_param':
            label = node.arg['label'].eval()
            if label in hp_assignment:
                memo[node] = hp_assignment[label]
    rval = pyll.rec_eval(space, memo=memo)
    return rval

# -- flake8 doesn't like blank last line
