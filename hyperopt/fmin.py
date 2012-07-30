import logging
import sys

import numpy as np
import time

import pyll
from pyll.stochastic import recursive_set_rng_kwarg

from .vectorize import VectorizeHelper
import base

logger = logging.getLogger(__name__)


class Domain(base.Bandit):
    """
    Picklable representation of search space and evaluation function.
    """
    def __init__(self, fn, expr, init_pyll_memo=None,
            cmd=None,
            workdir=None,
            **bandit_kwargs):
        self.fn = fn
        self.expr = expr
        self.init_pyll_memo = init_pyll_memo
        base.Bandit.__init__(self, expr, do_checks=False, **bandit_kwargs)

        # -- This code was stolen from base.BanditAlgo, a class which may soon
        #    be gone
        self.cmd = cmd
        self.workdir = workdir
        self.s_new_ids = pyll.Literal('new_ids')  # -- list at eval-time
        before = pyll.dfs(self.expr)
        # -- raises exception if expr contains cycles
        pyll.toposort(self.expr)
        vh = self.vh = VectorizeHelper(self.expr, self.s_new_ids)
        # -- raises exception if v_expr contains cycles
        pyll.toposort(vh.v_expr)

        idxs_by_label = vh.idxs_by_label()
        vals_by_label = vh.vals_by_label()
        after = pyll.dfs(self.expr)
        # -- try to detect if VectorizeHelper screwed up anything inplace
        assert before == after
        assert set(idxs_by_label.keys()) == set(vals_by_label.keys())
        assert set(idxs_by_label.keys()) == set(self.params.keys())

        # -- make the graph runnable and SON-encodable
        # N.B. operates inplace
        self.s_idxs_vals = recursive_set_rng_kwarg(
                pyll.scope.pos_args(idxs_by_label, vals_by_label),
                pyll.as_apply(self.rng))

        # -- raises an exception if no topological ordering exists
        pyll.toposort(self.s_idxs_vals)

    def evaluate(self, config, ctrl):
        memo = self.memo_from_config(config)
        memo[self.pyll_ctrl] = ctrl
        if self.init_pyll_memo:
            memo = self.init_pyll_memo(memo, config=config, ctrl=ctrl)
        if self.rng is not None and not self.installed_rng:
            # -- N.B. this modifies the expr graph in-place
            #    XXX this feels wrong
            self.expr = recursive_set_rng_kwarg(self.expr,
                pyll.as_apply(self.rng))
            self.installed_rng = True
        try:
            # -- the "work" of evaluating `config` can be written
            #    either into the pyll part (self.expr)
            #    or the normal Python part (self.fn)
            pyll_rval = pyll.rec_eval(self.expr, memo=memo)
            rval = self.fn(pyll_rval)
        except Exception, e:
            n_match = 0
            for match, match_pair in self.exceptions:
                if match(e):
                    rval = match_pair(e)
                    logger.info('Caught fn exception %s' % str(rval))
                    n_match += 1
                    break
            if n_match == 0:
                raise

        if isinstance(rval, (float, int, np.number)):
            dict_rval = {'loss': rval}
        elif isinstance(rval, (dict,)):
            dict_rval = rval
            if 'loss' not in dict_rval:
                raise ValueError('dictionary must have "loss" key',
                        dict_rval.keys())
        else:
            raise TypeError('invalid return type (neither number nor dict)', rval)

        if dict_rval['loss'] is not None:
            # -- fail if cannot be cast to float
            dict_rval['loss'] = float(dict_rval['loss'])

        dict_rval.setdefault('status', base.STATUS_OK)
        if dict_rval['status'] not in base.STATUS_STRINGS:
            raise ValueError('invalid status string', dict_rval['status'])

        attachments = dict_rval.pop('attachments', {})
        for key, val in attachments.items():
            ctrl.attachments[key] = val

        return dict_rval

    def short_str(self):
        return 'Domain{%s}' % str(self.fn)


# TODO: deprecate base.Experiment
class FMinIter(object):
    """Object for conducting search experiments.
    """
    catch_bandit_exceptions = False

    def __init__(self, algo, domain, trials, async=None,
            max_queue_len=1,
            poll_interval_secs=1.0,
            max_evals=sys.maxint,
            ):
        self.algo = algo
        self.domain = domain
        self.trials = trials
        if async is None:
            self.async = trials.async
        else:
            self.async = async
        self.poll_interval_secs = poll_interval_secs
        self.max_queue_len = max_queue_len
        self.max_evals = max_evals

    def serial_evaluate(self, N=-1):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == base.JOB_STATE_NEW:
                spec = base.spec_from_misc(trial['misc'])
                ctrl = base.Ctrl(self.trials, current_trial=trial)
                try:
                    result = self.domain.evaluate(spec, ctrl)
                except Exception, e:
                    logger.info('job exception: %s' % str(e))
                    trial['state'] = base.JOB_STATE_ERROR
                    trial['misc']['error'] = (str(type(e)), str(e))
                    if not self.catch_bandit_exceptions:
                        raise
                else:
                    logger.info('job returned status: %s' % result['status'])
                    logger.info('job returned loss: %s' % result.get('loss' ))
                    trial['state'] = base.JOB_STATE_DONE
                    trial['result'] = result
                N -= 1
                if N == 0:
                    break
        self.trials.refresh()

    def block_until_done(self):
        if self.async:
            unfinished_states = [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]

            def get_queue_len():
                return self.trials.count_by_state_unsynced(unfinished_states)

            qlen = get_queue_len()
            while qlen > 0:
                logger.info('Waiting for %d jobs to finish ...' % qlen)
                time.sleep(self.poll_interval_secs)
                qlen = get_queue_len()
            self.trials.refresh()
        else:
            self.serial_evaluate()

    def run(self, N, block_until_done=True):
        """
        block_until_done  means that the process blocks until ALL jobs in
        trials are not in running or new state

        suggest() can pass instance of StopExperiment to break out of
        enqueuing loop
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
                new_trials = algo(new_ids, self.domain, trials)
                if new_trials is base.StopExperiment:
                    stopped = True
                    break
                else:
                    assert len(new_ids) >= len(new_trials)
                    if len(new_trials):
                        self.trials.insert_trial_docs(new_trials)
                        self.trials.refresh()
                        n_queued += len(new_trials)
                        qlen = get_queue_len()
                    else:
                        break

            if self.async:
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

    def next(self):
        self.run(1, block_until_done=self.async)
        if len(self.trials) >= self.max_evals:
            raise StopIteration()
        return self.trials._dynamic_trials[-1]

    def exhaust(self):
        for foo in self:
            pass
        self.trials.refresh()
        return self


def fmin(fn, space, algo, max_evals,
        trials=None,
        fn_exceptions=(),
        init_pyll_memo=None,
        rseed=123,
        ):
    """
    Minimize `f` over the given `space` using random search.

    Parameters:
    -----------
    f - a callable taking a dictionary as an argument. It can return either a
        scalar loss value, or a result dictionary. The argument dictionary has
        keys for the hp_XXX nodes in the `space` and a `ctrl` key.
        If returning a dictionary, `f`
        must return a 'loss' key, and may optionally return a 'status' key and
        certain other reserved keys to communicate with the Experiment and
        optimization algorithm [1, 2]. The entire dictionary will be stored to
        the trials object associated with the experiment.

    space - a pyll graph involving hp_<xxx> nodes (see `pyll_utils`)

    algo - a minimization algorithm presented as an oracle
        `algo(new_ids, domain, trials)` returns a list of new trials to
        evaluate, one for each element of `new_ids`

    [1] See keys used in `base.Experiment` and `base.Bandit`
    [2] Optimization algorithms may in some cases use or require auxiliary
        feedback.
    """

    if trials is None:
        trials = base.Trials()

    domain = Domain(fn, space,
            exceptions=fn_exceptions,
            init_pyll_memo=init_pyll_memo,
            rseed=rseed)

    rval = FMinIter(algo, domain, trials, max_evals=max_evals)
    rval.exhaust()
    return rval

