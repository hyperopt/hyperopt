"""Base classes / Design

The design is that there are three components fitting together in this project:

- Trials - a list of documents including at least sub-documents:
    ['spec'] - the specification of hyper-parameters for a job
    ['result'] - the result of Bandit.evaluate(). Typically includes:
        ['status'] - one of the STATUS_STRINGS
        ['loss'] - real-valued scalar that hyperopt is trying to minimize
    ['idxs'] - compressed representation of spec
    ['vals'] - compressed representation of spec
    ['tid'] - trial id (unique in Trials list)

- Bandit - specifies a search problem

- BanditAlgo - an algorithm for solving a Bandit search problem

- Experiment - uses a Bandit and a BanditAlgo to carry out a search by
         interacting with a Trials object.

- Ctrl - a channel for two-way communication
         between an Experiment and Bandit.evaluate.
         Experiment subclasses may subclass Ctrl to match. For example, if an
         experiment is going to dispatch jobs in other threads, then an
         appropriate thread-aware Ctrl subclass should go with it.

"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging
import datetime
import sys

import numpy as np

import bson  # -- comes with pymongo
from bson.objectid import ObjectId

import pyll
#from pyll import scope  # looks unused but
from pyll.stochastic import recursive_set_rng_kwarg

from .exceptions import DuplicateLabel
from .exceptions import InvalidTrial
from .pyll_utils import hp_choice
from .utils import pmin_sampled
from .utils import use_obj_for_literal_in_memo
from .vectorize import VectorizeHelper

logger = logging.getLogger(__name__)


# -- STATUS values
# These are used to store job status in a backend-agnostic way, for the
# purpose of communicating between Bandit, BanditAlgo, and any
# visualization/monitoring code.

# -- named constants for status possibilities
STATUS_NEW = 'new'
STATUS_RUNNING = 'running'
STATUS_SUSPENDED = 'suspended'
STATUS_OK = 'ok'
STATUS_FAIL = 'fail'
STATUS_STRINGS = (
    'new',        # computations have not started
    'running',    # computations are in prog
    'suspended',  # computations have been suspended, job is not finished
    'ok',         # computations are finished, terminated normally
    'fail')       # computations are finished, terminated with error
                  #   - result['status_fail'] should contain more info


# -- JOBSTATE values
# These are used to store job states for the purpose of scheduling and running
# Bandit.evaluate.  These values are used to communicate between an Experiment
# and a worker process.

# -- named constants for job execution pipeline
JOB_STATE_NEW = 0
JOB_STATE_RUNNING = 1
JOB_STATE_DONE = 2
JOB_STATE_ERROR = 3
JOB_STATES = [
    JOB_STATE_NEW,
    JOB_STATE_RUNNING,
    JOB_STATE_DONE,
    JOB_STATE_ERROR]


TRIAL_KEYS = [
    'tid',
    'spec',
    'result',
    'misc',
    'state',
    'owner',
    'book_time',
    'refresh_time',
    'exp_key']

TRIAL_MISC_KEYS = [
    'tid',
    'cmd',
    'idxs',
    'vals']


def _all_same(*args):
    return 1 == len(set(args))


def StopExperiment(*args, **kwargs):
    """ Return StopExperiment

    StopExperiment is a symbol used as a special return value from
    BanditAlgo.suggest. With this implementation both `return StopExperiment`
    and `return StopExperiment()` have the same effect.
    """
    return StopExperiment


def SONify(arg, memo=None):
    add_arg_to_raise = True
    try:
        if memo is None:
            memo = {}
        if id(arg) in memo:
            rval = memo[id(arg)]
        if isinstance(arg, ObjectId):
            rval = arg
        elif isinstance(arg, datetime.datetime):
            rval = arg
        elif isinstance(arg, np.floating):
            rval = float(arg)
        elif isinstance(arg, np.integer):
            rval = int(arg)
        elif isinstance(arg, (list, tuple)):
            rval = type(arg)([SONify(ai, memo) for ai in arg])
        elif isinstance(arg, dict):
            rval = dict(
                [(SONify(k, memo), SONify(v, memo)) for k, v in arg.items()])
        elif isinstance(arg, (basestring, float, int, long, type(None))):
            rval = arg
        elif isinstance(arg, np.ndarray):
            if arg.ndim == 0:
                rval = SONify(arg.sum())
            else:
                rval = map(SONify, arg)  # N.B. memo None
        # -- put this after ndarray because ndarray not hashable
        elif arg in (True, False):
            rval = int(arg)
        else:
            add_arg_to_raise = False
            raise TypeError('SONify', arg)
    except Exception, e:
        if add_arg_to_raise:
            e.args = e.args + (arg,)
        raise
    memo[id(rval)] = rval
    return rval


def miscs_update_idxs_vals(miscs, idxs, vals,
                           assert_all_vals_used=True,
                           idxs_map=None):
    """
    Unpack the idxs-vals format into the list of dictionaries that is
    `misc`.

    idxs_map: a dictionary of id->id mappings so that the misc['idxs'] can
        contain different numbers than the idxs argument. XXX CLARIFY
    """
    if idxs_map is None:
        idxs_map = {}

    def imap(i):
        return idxs_map.get(i, i)

    assert set(idxs.keys()) == set(vals.keys())

    misc_by_id = dict([(m['tid'], m) for m in miscs])

    if idxs and assert_all_vals_used:
        # -- Assert that every val will be used to update some doc.
        all_ids = set()
        for idxlist in idxs.values():
            all_ids.update(map(imap, idxlist))
        assert all_ids == set(misc_by_id.keys())

    for tid, misc_tid in misc_by_id.items():
        misc_tid['idxs'] = {}
        misc_tid['vals'] = {}
        for node_id in idxs:
            node_idxs = map(imap, idxs[node_id])
            node_vals = vals[node_id]
            if tid in node_idxs:
                pos = node_idxs.index(tid)
                misc_tid['idxs'][node_id] = [tid]
                misc_tid['vals'][node_id] = [node_vals[pos]]

                # -- assert that tid occurs only once
                assert tid not in node_idxs[pos+1:]
            else:
                misc_tid['idxs'][node_id] = []
                misc_tid['vals'][node_id] = []
    return miscs


def miscs_to_idxs_vals(miscs, keys=None):
    if keys is None:
        if len(miscs) == 0:
            raise ValueError('cannot infer keys from empty miscs')
        keys = miscs[0]['idxs'].keys()
    idxs = dict([(k, []) for k in keys])
    vals = dict([(k, []) for k in keys])
    for misc in miscs:
        for node_id in idxs:
            t_idxs = misc['idxs'][node_id]
            t_vals = misc['vals'][node_id]
            assert len(t_idxs) == len(t_vals)
            assert t_idxs == [] or t_idxs == [misc['tid']]
            idxs[node_id].extend(t_idxs)
            vals[node_id].extend(t_vals)
    return idxs, vals


def spec_from_misc(misc):
    spec = {}
    for k, v in misc['vals'].items():
        if len(v) == 0:
            pass
        elif len(v) == 1:
            spec[k] = v[0]
        else:
            raise NotImplementedError('multiple values', (k, v))
    return spec


class Trials(object):
    """Database interface supporting data-driven model-based optimization.

    The model-based optimization algorithms used by hyperopt's fmin function
    work by analyzing samples of a response surface--a history of what points
    in the search space were tested, and what was discovered by those tests.
    A Trials instance stores that history and makes it available to fmin and
    to the various optimization algorithms.

    This class (`base.Trials`) is a pure-Python implementation of the database
    in terms of lists of dictionaries.  Subclass `mongoexp.MongoTrials`
    implements the same API in terms of a mongodb database running in another
    process. Other subclasses may be implemented in future.

    The elements of `self.trials` represent all of the completed, in-progress,
    and scheduled evaluation points from an e.g. `fmin` call.

    Each element of `self.trials` is a dictionary with *at least* the following
    keys:

    * **tid**: a unique trial identification object within this Trials instance
      usually it is an integer, but it isn't obvious that other sortable,
      hashable objects couldn't be used at some point.

    * **result**: a sub-dictionary representing what was returned by the fmin
      evaluation function. This sub-dictionary has a key 'status' with a value
      from `STATUS_STRINGS` and if the status is 'ok', then there should be
      a 'loss' key as well with a floating-point value.  Other special keys in
      this sub-dictionary may be used by optimization algorithms  (see them
      for details). Other keys in this sub-dictionary can be used by the
      evaluation function to store miscelaneous diagnostics and debugging
      information.

    * **misc**: despite generic name, this is currently where the trial's
      hyperparameter assigments are stored. This sub-dictionary has two
      elements: `'idxs'` and `'vals'`. The `vals` dictionary is
      a sub-sub-dictionary mapping each hyperparameter to either `[]` (if the
      hyperparameter is inactive in this trial), or `[<val>]` (if the
      hyperparameter is active). The `idxs` dictionary is technically
      redundant -- it is the same as `vals` but it maps hyperparameter names
      to either `[]` or `[<tid>]`.

    """

    async = False

    def __init__(self, exp_key=None, refresh=True):
        self._ids = set()
        self._dynamic_trials = []
        self._exp_key = exp_key
        self.attachments = {}
        if refresh:
            self.refresh()

    def view(self, exp_key=None, refresh=True):
        rval = object.__new__(self.__class__)
        rval._exp_key = exp_key
        rval._ids = self._ids
        rval._dynamic_trials = self._dynamic_trials
        rval.attachments = self.attachments
        if refresh:
            rval.refresh()
        return rval

    def aname(self, trial, name):
        return 'ATTACH::%s::%s' % (trial['tid'], name)

    def trial_attachments(self, trial):
        """
        Support syntax for load:  self.trial_attachments(doc)[name]
        # -- does this work syntactically?
        #    (In any event a 2-stage store will work)
        Support syntax for store: self.trial_attachments(doc)[name] = value
        """

        # don't offer more here than in MongoCtrl
        class Attachments(object):
            def __contains__(_self, name):
                return self.aname(trial, name) in self.attachments

            def __getitem__(_self, name):
                return self.attachments[self.aname(trial, name)]

            def __setitem__(_self, name, value):
                self.attachments[self.aname(trial, name)] = value

            def __delitem__(_self, name):
                del self.attachments[self.aname(trial, name)]

        return Attachments()

    def __iter__(self):
        try:
            return iter(self._trials)
        except AttributeError:
            print >> sys.stderr, "You have to refresh before you iterate"
            raise

    def __len__(self):
        try:
            return len(self._trials)
        except AttributeError:
            print >> sys.stderr, "You have to refresh before you compute len"
            raise

    def __getitem__(self, item):
        # -- how to make it obvious whether indexing is by _trials position
        #    or by tid if both are integers?
        raise NotImplementedError('')

    def refresh(self):
        # In MongoTrials, this method fetches from database
        if self._exp_key is None:
            self._trials = [
                tt for tt in self._dynamic_trials
                if tt['state'] != JOB_STATE_ERROR]
        else:
            self._trials = [tt
                            for tt in self._dynamic_trials
                            if (tt['state'] != JOB_STATE_ERROR
                                and tt['exp_key'] == self._exp_key)]
        self._ids.update([tt['tid'] for tt in self._trials])

    @property
    def trials(self):
        return self._trials

    @property
    def tids(self):
        return [tt['tid'] for tt in self._trials]

    @property
    def specs(self):
        return [tt['spec'] for tt in self._trials]

    @property
    def results(self):
        return [tt['result'] for tt in self._trials]

    @property
    def miscs(self):
        return [tt['misc'] for tt in self._trials]

    @property
    def idxs(self):
        return miscs_to_idxs_vals(self.miscs)[0]

    @property
    def vals(self):
        return miscs_to_idxs_vals(self.miscs)[1]

    def assert_valid_trial(self, trial):
        if not (hasattr(trial, 'keys') and hasattr(trial, 'values')):
            raise InvalidTrial('trial should be dict-like', trial)
        for key in TRIAL_KEYS:
            if key not in trial:
                raise InvalidTrial('trial missing key', key)
        for key in TRIAL_MISC_KEYS:
            if key not in trial['misc']:
                raise InvalidTrial('trial["misc"] missing key', key)
        if trial['tid'] != trial['misc']['tid']:
            raise InvalidTrial(
                'tid mismatch between root and misc',
                trial)
        # -- check for SON-encodable
        try:
            bson.BSON.encode(trial)
        except:
            # TODO: save the trial object somewhere to inspect, fix, re-insert
            #       so that precious data is not simply deallocated and lost.
            print '-' * 80
            print "CANT ENCODE"
            print '-' * 80
            raise
        if trial['exp_key'] != self._exp_key:
            raise InvalidTrial('wrong exp_key',
                               (trial['exp_key'], self._exp_key))
        # XXX how to assert that tids are unique?
        return trial

    def _insert_trial_docs(self, docs):
        """insert with no error checking
        """
        rval = [doc['tid'] for doc in docs]
        self._dynamic_trials.extend(docs)
        return rval

    def insert_trial_doc(self, doc):
        """insert trial after error checking

        Does not refresh. Call self.refresh() for the trial to appear in
        self.specs, self.results, etc.
        """
        doc = self.assert_valid_trial(SONify(doc))
        return self._insert_trial_docs([doc])[0]
        # refreshing could be done fast in this base implementation, but with
        # a real DB the steps should be separated.

    def insert_trial_docs(self, docs):
        """ trials - something like is returned by self.new_trial_docs()
        """
        docs = [self.assert_valid_trial(SONify(doc))
                for doc in docs]
        return self._insert_trial_docs(docs)

    def new_trial_ids(self, N):
        aa = len(self._ids)
        rval = range(aa, aa + N)
        self._ids.update(rval)
        return rval

    def new_trial_docs(self, tids, specs, results, miscs):
        assert len(tids) == len(specs) == len(results) == len(miscs)
        rval = []
        for tid, spec, result, misc in zip(tids, specs, results, miscs):
            doc = dict(
                state=JOB_STATE_NEW,
                tid=tid,
                spec=spec,
                result=result,
                misc=misc)
            doc['exp_key'] = self._exp_key
            doc['owner'] = None
            doc['version'] = 0
            doc['book_time'] = None
            doc['refresh_time'] = None
            rval.append(doc)
        return rval

    def source_trial_docs(self, tids, specs, results, miscs, sources):
        assert _all_same(map(len, [tids, specs, results, miscs, sources]))
        rval = []
        for tid, spec, result, misc, source in zip(tids, specs, results, miscs,
                                                   sources):
            doc = dict(
                version=0,
                tid=tid,
                spec=spec,
                result=result,
                misc=misc,
                state=source['state'],
                exp_key=source['exp_key'],
                owner=source['owner'],
                book_time=source['book_time'],
                refresh_time=source['refresh_time'],
                )
            # -- ensure that misc has the following fields,
            #    some of which may already by set correctly.
            assign = ('tid', tid), ('cmd', None), ('from_tid', source['tid'])
            for k, v in assign:
                assert doc['misc'].setdefault(k, v) == v
            rval.append(doc)
        return rval

    def delete_all(self):
        self._dynamic_trials = []
        self.attachments = {}
        self.refresh()

    def count_by_state_synced(self, arg, trials=None):
        """
        Return trial counts by looking at self._trials
        """
        if trials is None:
            trials = self._trials
        if arg in JOB_STATES:
            queue = [doc for doc in trials if doc['state'] == arg]
        elif hasattr(arg, '__iter__'):
            states = set(arg)
            assert all([x in JOB_STATES for x in states])
            queue = [doc for doc in trials if doc['state'] in states]
        else:
            raise TypeError(arg)
        rval = len(queue)
        return rval

    def count_by_state_unsynced(self, arg):
        """
        Return trial counts that count_by_state_synced would return if we
        called refresh() first.
        """
        if self._exp_key is not None:
            exp_trials = [tt
                          for tt in self._dynamic_trials
                          if tt['exp_key'] == self._exp_key]
        else:
            exp_trials = self._dynamic_trials
        return self.count_by_state_synced(arg, trials=exp_trials)

    def losses(self, bandit=None):
        if bandit is None:
            return [r.get('loss') for r in self.results]
        else:
            return map(bandit.loss, self.results, self.specs)

    def statuses(self, bandit=None):
        if bandit is None:
            return [r.get('status') for r in self.results]
        else:
            return map(bandit.status, self.results, self.specs)

    def average_best_error(self, bandit=None):
        """Return the average best error of the experiment

        Average best error is defined as the average of bandit.true_loss,
        weighted by the probability that the corresponding bandit.loss is best.

        For bandits with loss measurement variance of 0, this function simply
        returns the true_loss corresponding to the result with the lowest loss.
        """

        if bandit is None:
            results = self.results
            loss = [r['loss']
                    for r in results if r['status'] == STATUS_OK]
            loss_v = [r.get('loss_variance', 0)
                      for r in results if r['status'] == STATUS_OK]
            true_loss = [r.get('true_loss', r['loss'])
                         for r in results if r['status'] == STATUS_OK]
        else:
            def fmap(f):
                rval = np.asarray([
                    f(r, s)
                    for (r, s) in zip(self.results, self.specs)
                    if bandit.status(r) == STATUS_OK]).astype('float')
                if not np.all(np.isfinite(rval)):
                    raise ValueError()
                return rval
            loss = fmap(bandit.loss)
            loss_v = fmap(bandit.loss_variance)
            true_loss = fmap(bandit.true_loss)
        loss3 = zip(loss, loss_v, true_loss)
        if not loss3:
            raise ValueError('Empty loss vector')
        loss3.sort()
        loss3 = np.asarray(loss3)
        if np.all(loss3[:, 1] == 0):
            best_idx = np.argmin(loss3[:, 0])
            return loss3[best_idx, 2]
        else:
            cutoff = 0
            sigma = np.sqrt(loss3[0][1])
            while (cutoff < len(loss3)
                    and loss3[cutoff][0] < loss3[0][0] + 3 * sigma):
                cutoff += 1
            pmin = pmin_sampled(loss3[:cutoff, 0], loss3[:cutoff, 1])
            #print pmin
            #print loss3[:cutoff, 0]
            #print loss3[:cutoff, 1]
            #print loss3[:cutoff, 2]
            avg_true_loss = (pmin * loss3[:cutoff, 2]).sum()
            return avg_true_loss

    @property
    def best_trial(self):
        results = self.results
        best = np.argmin([r.get('loss', float('inf')) for r in results])
        return self.trials[best]

    @property
    def argmin(self):
        best_trial = self.best_trial
        vals = best_trial['misc']['vals']
        # unpack the one-element lists to values
        # and skip over the 0-element lists
        rval = {}
        for k, v in vals.items():
            if v:
                rval[k] = v[0]
        return rval

    def fmin(self, fn, space, algo, max_evals,
             rstate=None,
             verbose=0,
             pass_expr_memo_ctrl=None,
             catch_eval_exceptions=False,
             return_argmin=True,
             ):
        """Minimize a function over a hyperparameter space.

        For most parameters, see `hyperopt.fmin.fmin`.

        Parameters
        ----------

        catch_eval_exceptions : bool, default False
            If set to True, exceptions raised by either the evaluation of the
            configuration space from hyperparameters or the execution of `fn`
            , will be caught by fmin, and recorded in self._dynamic_trials as
            error jobs (JOB_STATE_ERROR).  If set to False, such exceptions
            will not be caught, and so they will propagate to calling code.


        """
        # -- Stop-gap implementation!
        #    fmin should have been a Trials method in the first place
        #    but for now it's still sitting in another file.
        import fmin as fmin_module
        return fmin_module.fmin(
            fn, space, algo, max_evals,
            trials=self,
            rstate=rstate,
            verbose=verbose,
            allow_trials_fmin=False,  # -- prevent recursion
            pass_expr_memo_ctrl=pass_expr_memo_ctrl,
            catch_eval_exceptions=catch_eval_exceptions,
            return_argmin=return_argmin)


def trials_from_docs(docs, validate=True, **kwargs):
    """Construct a Trials base class instance from a list of trials documents
    """
    rval = Trials(**kwargs)
    if validate:
        rval.insert_trial_docs(docs)
    else:
        rval._insert_trial_docs(docs)
    rval.refresh()
    return rval


class Ctrl(object):
    """Control object for interruptible, checkpoint-able evaluation
    """
    info = logger.info
    warn = logger.warn
    error = logger.error
    debug = logger.debug

    def __init__(self, trials, current_trial=None):
        # -- attachments should be used like
        #      attachments[key]
        #      attachments[key] = value
        #    where key and value are strings. Client code should not
        #    expect any dictionary-like behaviour beyond that (no update)
        if trials is None:
            self.trials = Trials()
        else:
            self.trials = trials
        self.current_trial = current_trial

    def checkpoint(self, r=None):
        assert self.current_trial in self.trials._trials
        if r is not None:
            self.current_trial['result'] = r

    @property
    def attachments(self):
        """
        Support syntax for load:  self.attachments[name]
        Support syntax for store: self.attachments[name] = value
        """
        return self.trials.trial_attachments(trial=self.current_trial)

    def inject_results(self, specs, results, miscs, new_tids=None):
        """Inject new results into self.trials

        Returns ??? XXX

        new_tids can be None, in which case new tids will be generated
        automatically

        """
        trial = self.current_trial
        assert trial is not None
        num_news = len(specs)
        assert len(specs) == len(results) == len(miscs)
        if new_tids is None:
            new_tids = self.trials.new_trial_ids(num_news)
        new_trials = self.trials.source_trial_docs(tids=new_tids,
                                                   specs=specs,
                                                   results=results,
                                                   miscs=miscs,
                                                   sources=[trial])
        for t in new_trials:
            t['state'] = JOB_STATE_DONE
        return self.trials.insert_trial_docs(new_trials)


class Bandit(object):
    """Specification of bandit problem.

    template - pyll specification of search domain

    evaluate - interruptible/checkpt calling convention for evaluation routine

    """
    # -- the Ctrl object is not used directly, but rather
    #    a live Ctrl instance is inserted for the pyll_ctrl
    #    in self.evaluate so that it can be accessed from within
    #    the pyll graph describing the search space.
    pyll_ctrl = pyll.as_apply(Ctrl)

    exceptions = []

    def __init__(self, expr,
                 name=None,
                 rseed=None,
                 loss_target=None,
                 exceptions=None,
                 do_checks=True,
                 ):
        if do_checks:
            if isinstance(expr, pyll.Apply):
                self.expr = expr
                # XXX: verify that expr is a dictionary with the right keys,
                #      then refactor the code below
            elif isinstance(expr, dict):
                if 'loss' not in expr:
                    raise ValueError('expr must define a loss')
                if 'status' not in expr:
                    expr['status'] = STATUS_OK
                self.expr = pyll.as_apply(expr)
            else:
                raise TypeError('expr must be a dictionary')
        else:
            self.expr = pyll.as_apply(expr)

        self.params = {}
        for node in pyll.dfs(self.expr):
            if node.name == 'hyperopt_param':
                label = node.arg['label'].obj
                if label in self.params:
                    raise DuplicateLabel(label)
                self.params[label] = node.arg['obj']

        if exceptions is not None:
            self.exceptions = exceptions
        self.loss_target = loss_target
        self.installed_rng = False
        if rseed is None:
            self.rng = None
        else:
            self.rng = np.random.RandomState(rseed)

        self.name = name

    def memo_from_config(self, config):
        memo = {}
        for node in pyll.dfs(self.expr):
            if node.name == 'hyperopt_param':
                label = node.arg['label'].obj
                # -- hack because it's not really garbagecollected
                #    this does have the desired effect of crashing the
                #    function if rec_eval actually needs a value that
                #    the the optimization algorithm thought to be unnecessary
                memo[node] = config.get(label, pyll.base.GarbageCollected)
        return memo

    def short_str(self):
        return self.__class__.__name__

    def use_obj_for_literal_in_memo(self, obj, lit, memo):
        return use_obj_for_literal_in_memo(self.expr, obj, lit, memo)

    def evaluate(self, config, ctrl):
        """Return a result document
        """
        memo = self.memo_from_config(config)
        self.use_obj_for_literal_in_memo(ctrl, Ctrl, memo)
        if self.rng is not None and not self.installed_rng:
            # -- N.B. this modifies the expr graph in-place
            #    XXX this feels wrong
            self.expr = recursive_set_rng_kwarg(self.expr,
                                                pyll.as_apply(self.rng))
            self.installed_rng = True
        try:
            r_dct = pyll.rec_eval(self.expr, memo=memo)
        except Exception, e:
            n_match = 0
            for match, match_pair in self.exceptions:
                if match(e):
                    r_dct = match_pair(e)
                    n_match += 1
                    break
            if n_match == 0:
                raise
        assert 'loss' in r_dct
        if r_dct['loss'] is not None:
            # -- assert that it can at least be cast to float
            float(r_dct['loss'])
        if r_dct['status'] not in STATUS_STRINGS:
            raise ValueError('invalid status string', r_dct['status'])
        return r_dct

    def loss(self, result, config=None):
        """Extract the scalar-valued loss from a result document
        """
        return result.get('loss', None)

    def loss_variance(self, result, config=None):
        """Return the variance in the estimate of the loss"""
        return result.get('loss_variance', 0.0)

    def true_loss(self, result, config=None):
        """Return a true loss, in the case that the `loss` is a surrogate"""
        # N.B. don't use get() here, it evaluates self.loss un-necessarily
        try:
            return result['true_loss']
        except KeyError:
            return self.loss(result, config=config)

    def true_loss_variance(self, config=None):
        """Return the variance in  true loss,
        in the case that the `loss` is a surrogate.
        """
        raise NotImplementedError()

    def status(self, result, config=None):
        """Extract the job status from a result document
        """
        return result['status']

    def new_result(self):
        """Return a JSON-encodable object
        to serve as the 'result' for new jobs.
        """
        return {'status': STATUS_NEW}


class Domain(Bandit):
    """
    Picklable representation of search space and evaluation function.
    """
    # TODO: Merge with Bandit -- two classes are not necessary
    rec_eval_print_node_on_error = False

    def __init__(self, fn, expr,
                 workdir=None,
                 pass_expr_memo_ctrl=None,
                 **bandit_kwargs):
        self.cmd = ('domain_attachment', 'FMinIter_Domain')
        self.fn = fn
        if pass_expr_memo_ctrl is None:
            self.pass_expr_memo_ctrl = getattr(fn,
                                               'fmin_pass_expr_memo_ctrl',
                                               False)
        else:
            self.pass_expr_memo_ctrl = pass_expr_memo_ctrl
        Bandit.__init__(self, expr, do_checks=False, **bandit_kwargs)

        # -- This code was stolen from base.BanditAlgo, a class which may soon
        #    be gone
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

    def install_rng_in_s_idxs_vals(self, rng):
        self.s_idxs_vals = recursive_set_rng_kwarg(self.s_idxs_vals,
                                                   pyll.as_apply(rng))

    def install_rng_in_expr(self, rng):
        assert rng is not None
        self.expr = recursive_set_rng_kwarg(self.expr,
                                            pyll.as_apply(rng))
        self.installed_rng = (rng is self.rng)

    def evaluate(self, config, ctrl, attach_attachments=True):
        memo = self.memo_from_config(config)
        self.use_obj_for_literal_in_memo(ctrl, Ctrl, memo)
        if self.rng is not None and not self.installed_rng:
            # -- N.B. this modifies the expr graph in-place
            #    XXX this feels wrong
            self.install_rng_in_expr(self.rng)
        if self.pass_expr_memo_ctrl:
            rval = self.fn(expr=self.expr, memo=memo, ctrl=ctrl)
        else:
            # -- the "work" of evaluating `config` can be written
            #    either into the pyll part (self.expr)
            #    or the normal Python part (self.fn)
            pyll_rval = pyll.rec_eval(
                self.expr,
                memo=memo,
                print_node_on_error=self.rec_eval_print_node_on_error)
            rval = self.fn(pyll_rval)

        # XXX: THIS LOGIC IS KIND OF WRONG
        #      If the returned thing is a dictionary
        #      then it *must* have a status key, and then
        #      depending on what that is, it might need a loss, an error
        #      etc.
        if isinstance(rval, (float, int, np.number)):
            dict_rval = {'loss': rval}
        elif isinstance(rval, (dict,)):
            dict_rval = rval
            if 'loss' not in dict_rval:
                raise ValueError(
                    'dictionary must have "loss" key',
                    dict_rval.keys())
        else:
            raise TypeError(
                'invalid return type (neither number nor dict)',
                rval)

        if dict_rval['loss'] is not None:
            # -- fail if cannot be cast to float
            dict_rval['loss'] = float(dict_rval['loss'])

        dict_rval.setdefault('status', STATUS_OK)
        if dict_rval['status'] not in STATUS_STRINGS:
            raise ValueError('invalid status string', dict_rval['status'])

        if attach_attachments:
            attachments = dict_rval.pop('attachments', {})
            for key, val in attachments.items():
                ctrl.attachments[key] = val

        # -- don't do this here because SON-compatibility is only a requirement
        #    for trials destined for a mongodb. In-memory rvals can contain
        #    anything.
        #return base.SONify(dict_rval)
        return dict_rval

    def short_str(self):
        return 'Domain{%s}' % str(self.fn)


def as_bandit(**b_kwargs):
    """
    Decorate a function that returns a pyll expressions so that
    it becomes a Bandit instance instead of a function

    Example:

    @as_bandit(loss_target=0)
    def f(low, high):
        return {'loss': hp_uniform('x', low, high) ** 2 }

    """
    def deco(f):
        def wrapper(*args, **kwargs):
            if 'name' in b_kwargs:
                _b_kwargs = b_kwargs
            else:
                _b_kwargs = dict(b_kwargs, name=f.__name__)
            f_rval = f(*args, **kwargs)
            domain = Domain(lambda x: x, f_rval, **_b_kwargs)
            return domain
        wrapper.__name__ = f.__name__
        return wrapper
    return deco


@as_bandit()
def coin_flip():
    """ Possibly the simplest possible Bandit implementation
    """
    return {'loss': hp_choice('flip', [0.0, 1.0])}

# -- flake8 doesn't like blank last line
