"""Utilities for Parallel Model Selection with IPython

Author: James Bergstra <james.bergstra@gmail.com>
Licensed: MIT
"""
from time import sleep, time

import numpy as np
from IPython.parallel import interactive
#from IPython.parallel import TaskAborted
#from IPython.display import clear_output

from .base import Trials
from .base import Domain
from .base import JOB_STATE_NEW
from .base import JOB_STATE_RUNNING
from .base import JOB_STATE_DONE
from .base import JOB_STATE_ERROR
from .base import StopExperiment
from .base import spec_from_misc
from .utils import coarse_utcnow

import sys
print >> sys.stderr, "WARNING: IPythonTrials is not as complete, stable"
print >> sys.stderr, "         or well tested as Trials or MongoTrials."


class LostEngineError(RuntimeError):
    """An IPEngine disappeared during computation, and a job with it."""


class IPythonTrials(Trials):

    def __init__(self, client,
            job_error_reaction='raise',
            save_ipy_metadata=True):
        self._client = client
        self.job_map = {}
        self.job_error_reaction = job_error_reaction
        self.save_ipy_metadata = save_ipy_metadata
        Trials.__init__(self)
        self._testing_fmin_was_called = False

    def _insert_trial_docs(self, docs):
        rval = [doc['tid'] for doc in docs]
        self._dynamic_trials.extend(docs)
        return rval

    def refresh(self):
        job_map = {}

        # -- carry over state for active engines
        for eid in self._client.ids:
            job_map[eid] = self.job_map.pop(eid, (None, None))

        # -- deal with lost engines, abandoned promises
        for eid, (p, tt) in self.job_map.items():
            if self.job_error_reaction == 'raise':
                raise LostEngineError(p)
            elif self.job_error_reaction == 'log':
                tt['error'] = 'LostEngineError (%s)' % str(p)
                tt['state'] = JOB_STATE_ERROR
            else:
                raise ValueError(self.job_error_reaction)

        # -- remove completed jobs from job_map
        for eid, (p, tt) in job_map.items():
            if p is None:
                continue
            #print p
            #assert eid == p.engine_id
            if p.ready():
                try:
                    tt['result'] = p.get()
                    tt['state'] = JOB_STATE_DONE
                except Exception, e:
                    if self.job_error_reaction == 'raise':
                        raise
                    elif self.job_error_reaction == 'log':
                        tt['error'] = str(e)
                        tt['state'] = JOB_STATE_ERROR
                    else:
                        raise ValueError(self.job_error_reaction)
                if self.save_ipy_metadata:
                    tt['ipy_metadata'] = p.metadata
                tt['refresh_time'] = coarse_utcnow()
                job_map[eid] = (None, None)

        self.job_map = job_map

        Trials.refresh(self)

    def fmin(self, fn, space, algo, max_evals,
        rstate=None,
        verbose=0,
        wait=True,
        pass_expr_memo_ctrl=None,
        ):

        if rstate is None:
            rstate = np.random

        # -- used in test_ipy
        self._testing_fmin_was_called = True

        if pass_expr_memo_ctrl is None:
            try:
                pass_expr_memo_ctrl = fn.pass_expr_memo_ctrl
            except AttributeError:
                pass_expr_memo_ctrl = False

        domain = Domain(fn, space, rseed=rstate.randint(2 ** 31 - 1),
                pass_expr_memo_ctrl=pass_expr_memo_ctrl)

        last_print_time = 0

        while len(self._dynamic_trials) < max_evals:
            self.refresh()

            if verbose and last_print_time + 1 < time():
                print 'fmin: %4i/%4i/%4i/%4i  %f' % (
                    self.count_by_state_unsynced(JOB_STATE_NEW),
                    self.count_by_state_unsynced(JOB_STATE_RUNNING),
                    self.count_by_state_unsynced(JOB_STATE_DONE),
                    self.count_by_state_unsynced(JOB_STATE_ERROR),
                    min([float('inf')] + [l for l in self.losses() if l is not None])
                    )
                last_print_time = time()

            idles = [eid for (eid, (p, tt)) in self.job_map.items() if p is None]

            if idles:
                new_ids = self.new_trial_ids(len(idles))
                new_trials = algo(new_ids, domain, self)
                if new_trials is StopExperiment:
                    stopped = True # -- XXX: this should be returned somehow
                    break
                elif len(new_trials) == 0:
                    break
                else:
                    assert len(idles) == len(new_trials)
                    for eid, new_trial in zip(idles, new_trials):
                        now = coarse_utcnow()
                        new_trial['book_time'] = now
                        new_trial['refresh_time'] = now
                        promise = self._client[eid].apply_async(
                            call_domain,
                            domain,
                            config=spec_from_misc(new_trial['misc']),
                            )

                        # -- XXX bypassing checks because 'ar'
                        # is not ok for SONify... but should check
                        # for all else being SONify
                        tid, = self.insert_trial_docs([new_trial])
                        tt = self._dynamic_trials[-1]
                        assert tt['tid'] == tid
                        self.job_map[eid] = (promise, tt)
                        tt['state'] = JOB_STATE_RUNNING

        if wait:
            if verbose:
                print 'fmin: Waiting on remaining jobs...'
            self.wait(verbose=verbose)

        return self.argmin

    def wait(self, verbose=False, verbose_print_interval=1.0):
        last_print_time = 0
        while True:
            self.refresh()
            if verbose and last_print_time + verbose_print_interval < time():
                print 'fmin: %4i/%4i/%4i/%4i  %f' % (
                    self.count_by_state_unsynced(JOB_STATE_NEW),
                    self.count_by_state_unsynced(JOB_STATE_RUNNING),
                    self.count_by_state_unsynced(JOB_STATE_DONE),
                    self.count_by_state_unsynced(JOB_STATE_ERROR),
                    min([float('inf')]
                        + [l for l in self.losses() if l is not None])
                    )
                last_print_time = time()
            if self.count_by_state_unsynced(JOB_STATE_NEW):
                sleep(1e-1)
                continue
            if self.count_by_state_unsynced(JOB_STATE_RUNNING):
                sleep(1e-1)
                continue
            break

    def __getstate__(self):
        rval = dict(self.__dict__)
        del rval['_client']
        del rval['_trials']
        del rval['job_map']
        #print rval.keys()
        return rval

    def __setstate__(self, dct):
        self.__dict__ = dct
        self.job_map = {}
        Trials.refresh(self)


@interactive
def call_domain(domain, config):
    ctrl = None # -- not implemented yet
    return domain.evaluate(
            config=config,
            ctrl=ctrl,
            attach_attachments=False, # -- Not implemented yet
            )
