"""Base classes / Design

The design is that there are three components fitting together in this project:

- Trials - a list of documents including at least sub-documents:
    ['spec'] - the specification of hyper-parameters for a job
    ['result'] - the result of Domain.evaluate(). Typically includes:
        ['status'] - one of the STATUS_STRINGS
        ['loss'] - real-valued scalar that hyperopt is trying to minimize
    ['idxs'] - compressed representation of spec
    ['vals'] - compressed representation of spec
    ['tid'] - trial id (unique in Trials list)

- Domain - specifies a search problem

- Ctrl - a channel for two-way communication
         between an Experiment and Domain.evaluate.
         Experiment subclasses may subclass Ctrl to match. For example, if an
         experiment is going to dispatch jobs in other threads, then an
         appropriate thread-aware Ctrl subclass should go with it.

"""

import numbers
from past.builtins import basestring
import logging
import datetime
import sys

import numpy as np

try:
    import bson  # -- comes with pymongo
    from bson.objectid import ObjectId

    have_bson = True
except ImportError:
    have_bson = False

from . import pyll
from .pyll.stochastic import recursive_set_rng_kwarg

from .exceptions import (
    DuplicateLabel,
    InvalidTrial,
    InvalidResultStatus,
    InvalidLoss,
    AllTrialsFailed,
)
from .utils import pmin_sampled
from .utils import use_obj_for_literal_in_memo
from .vectorize import VectorizeHelper

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

logger = logging.getLogger(__name__)


# -- STATUS values
#    An eval_fn returning a dictionary must have a status key with
#    one of these values. They are used by optimization routines
#    and plotting functions.

STATUS_NEW = "new"
STATUS_RUNNING = "running"
STATUS_SUSPENDED = "suspended"
STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_STRINGS = (
    "new",  # computations have not started
    "running",  # computations are in prog
    "suspended",  # computations have been suspended, job is not finished
    "ok",  # computations are finished, terminated normally
    "fail",
)  # computations are finished, terminated with error
#   - result['status_fail'] should contain more info


# -- JOBSTATE values
# These are used internally by the scheduler.
# These values are used to communicate between an Experiment
# and a worker process. Consider moving them to mongoexp.

# -- named constants for job execution pipeline
JOB_STATE_NEW = 0
JOB_STATE_RUNNING = 1
JOB_STATE_DONE = 2
JOB_STATE_ERROR = 3
JOB_STATE_CANCEL = 4
JOB_STATES = [
    JOB_STATE_NEW,
    JOB_STATE_RUNNING,
    JOB_STATE_DONE,
    JOB_STATE_ERROR,
    JOB_STATE_CANCEL,
]
JOB_VALID_STATES = {JOB_STATE_NEW, JOB_STATE_RUNNING, JOB_STATE_DONE}


TRIAL_KEYS = [
    "tid",
    "spec",
    "result",
    "misc",
    "state",
    "owner",
    "book_time",
    "refresh_time",
    "exp_key",
]

TRIAL_MISC_KEYS = ["tid", "cmd", "idxs", "vals"]


def _all_same(*args):
    return 1 == len(set(args))


def SONify(arg, memo=None):
    if not have_bson:
        return arg
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
            rval = {SONify(k, memo): SONify(v, memo) for k, v in list(arg.items())}
        elif isinstance(arg, (basestring, float, int, type(None))):
            rval = arg
        elif isinstance(arg, np.ndarray):
            if arg.ndim == 0:
                rval = SONify(arg.sum())
            else:
                rval = list(map(SONify, arg))  # N.B. memo None
        # -- put this after ndarray because ndarray not hashable
        elif isinstance(arg, bool):
            rval = int(arg)
        else:
            add_arg_to_raise = False
            raise TypeError("SONify", arg)
    except Exception as e:
        if add_arg_to_raise:
            e.args = e.args + (arg,)
        raise
    memo[id(rval)] = rval
    return rval


def miscs_update_idxs_vals(miscs, idxs, vals, assert_all_vals_used=True, idxs_map=None):
    """
    Unpack the idxs-vals format into the list of dictionaries that is `misc`.

    idxs_map: a dictionary of id->id mappings so that the misc['idxs'] can
        contain different numbers than the idxs argument. XXX CLARIFY
    """
    if idxs_map is None:
        idxs_map = {}

    assert set(idxs.keys()) == set(vals.keys())

    misc_by_id = {m["tid"]: m for m in miscs}
    for m in miscs:
        m["idxs"] = {key: [] for key in idxs}
        m["vals"] = {key: [] for key in idxs}

    for key in idxs:
        assert len(idxs[key]) == len(vals[key])
        for tid, val in zip(idxs[key], vals[key]):
            tid = idxs_map.get(tid, tid)
            if assert_all_vals_used or tid in misc_by_id:
                misc_by_id[tid]["idxs"][key] = [tid]
                misc_by_id[tid]["vals"][key] = [val]

    return miscs


def miscs_to_idxs_vals(miscs, keys=None):
    if keys is None:
        if len(miscs) == 0:
            raise ValueError("cannot infer keys from empty miscs")
        keys = list(miscs[0]["idxs"].keys())
    idxs, vals = {k: [] for k in keys}, {k: [] for k in keys}
    for misc in miscs:
        for node_id in idxs:
            t_idxs = misc["idxs"][node_id]
            t_vals = misc["vals"][node_id]
            assert len(t_idxs) == len(t_vals)
            assert t_idxs == [] or t_idxs == [misc["tid"]]
            idxs[node_id].extend(t_idxs)
            vals[node_id].extend(t_vals)
    return idxs, vals


def spec_from_misc(misc):
    spec = {}
    for k, v in list(misc["vals"].items()):
        if len(v) == 0:
            pass
        elif len(v) == 1:
            spec[k] = v[0]
        else:
            raise NotImplementedError("multiple values", (k, v))
    return spec


def validate_timeout(timeout):
    if timeout is not None and (
        not isinstance(timeout, numbers.Number)
        or timeout <= 0
        or isinstance(timeout, bool)
    ):
        raise Exception(
            "The timeout argument should be None or a positive value. "
            f"Given value: {timeout}"
        )


def validate_loss_threshold(loss_threshold):
    if loss_threshold is not None and (
        not isinstance(loss_threshold, numbers.Number)
        or isinstance(loss_threshold, bool)
    ):
        raise Exception(
            "The loss_threshold argument should be None or a numeric value. "
            f"Given value: {loss_threshold}"
        )


class Trials:
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
      from `STATUS_STRINGS` and the status is `STATUS_OK`, then there should be
      a 'loss' key as well with a floating-point value.  Other special keys in
      this sub-dictionary may be used by optimization algorithms  (see them
      for details). Other keys in this sub-dictionary can be used by the
      evaluation function to store miscellaneous diagnostics and debugging
      information.

    * **misc**: despite generic name, this is currently where the trial's
      hyperparameter assignments are stored. This sub-dictionary has two
      elements: `'idxs'` and `'vals'`. The `vals` dictionary is
      a sub-sub-dictionary mapping each hyperparameter to either `[]` (if the
      hyperparameter is inactive in this trial), or `[<val>]` (if the
      hyperparameter is active). The `idxs` dictionary is technically
      redundant -- it is the same as `vals` but it maps hyperparameter names
      to either `[]` or `[<tid>]`.

    """

    asynchronous = False

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
        return "ATTACH::{}::{}".format(trial["tid"], name)

    def trial_attachments(self, trial):
        """
        Support syntax for load:  self.trial_attachments(doc)[name]
        # -- does this work syntactically?
        #    (In any event a 2-stage store will work)
        Support syntax for store: self.trial_attachments(doc)[name] = value
        """

        # don't offer more here than in MongoCtrl
        class Attachments:
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
            print("You have to refresh before you iterate", file=sys.stderr)
            raise

    def __len__(self):
        try:
            return len(self._trials)
        except AttributeError:
            print("You have to refresh before you compute len", file=sys.stderr)
            raise

    def __getitem__(self, item):
        # -- how to make it obvious whether indexing is by _trials position
        #    or by tid if both are integers?
        raise NotImplementedError("")

    def refresh(self):
        # In MongoTrials, this method fetches from database
        if self._exp_key is None:
            self._trials = [
                tt for tt in self._dynamic_trials if tt["state"] in JOB_VALID_STATES
            ]
        else:
            self._trials = [
                tt
                for tt in self._dynamic_trials
                if (tt["state"] in JOB_VALID_STATES and tt["exp_key"] == self._exp_key)
            ]
        self._ids.update([tt["tid"] for tt in self._trials])

    @property
    def trials(self):
        return self._trials

    @property
    def tids(self):
        return [tt["tid"] for tt in self._trials]

    @property
    def specs(self):
        return [tt["spec"] for tt in self._trials]

    @property
    def results(self):
        return [tt["result"] for tt in self._trials]

    @property
    def miscs(self):
        return [tt["misc"] for tt in self._trials]

    @property
    def idxs_vals(self):
        return miscs_to_idxs_vals(self.miscs)

    @property
    def idxs(self):
        return self.idxs_vals[0]

    @property
    def vals(self):
        return self.idxs_vals[1]

    def assert_valid_trial(self, trial):
        if not (hasattr(trial, "keys") and hasattr(trial, "values")):
            raise InvalidTrial("trial should be dict-like", trial)
        for key in TRIAL_KEYS:
            if key not in trial:
                raise InvalidTrial("trial missing key %s", key)
        for key in TRIAL_MISC_KEYS:
            if key not in trial["misc"]:
                raise InvalidTrial('trial["misc"] missing key', key)
        if trial["tid"] != trial["misc"]["tid"]:
            raise InvalidTrial("tid mismatch between root and misc", trial)
        # -- check for SON-encodable
        if have_bson:
            try:
                bson.BSON.encode(trial)
            except:
                # TODO: save the trial object somewhere to inspect, fix, re-insert
                #       so that precious data is not simply deallocated and lost.
                print("-" * 80)
                print("CAN'T ENCODE")
                print("-" * 80)
                raise
        if trial["exp_key"] != self._exp_key:
            raise InvalidTrial("wrong exp_key", (trial["exp_key"], self._exp_key))
        # XXX how to assert that tids are unique?
        return trial

    def _insert_trial_docs(self, docs):
        """insert with no error checking"""
        rval = [doc["tid"] for doc in docs]
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
        """trials - something like is returned by self.new_trial_docs()"""
        docs = [self.assert_valid_trial(SONify(doc)) for doc in docs]
        return self._insert_trial_docs(docs)

    def new_trial_ids(self, n):
        aa = len(self._ids)
        rval = list(range(aa, aa + n))
        self._ids.update(rval)
        return rval

    def new_trial_docs(self, tids, specs, results, miscs):
        assert len(tids) == len(specs) == len(results) == len(miscs)
        trials_docs = []
        for tid, spec, result, misc in zip(tids, specs, results, miscs):
            doc = {
                "state": JOB_STATE_NEW,
                "tid": tid,
                "spec": spec,
                "result": result,
                "misc": misc,
                "exp_key": self._exp_key,
                "owner": None,
                "version": 0,
                "book_time": None,
                "refresh_time": None,
            }
            trials_docs.append(doc)
        return trials_docs

    def source_trial_docs(self, tids, specs, results, miscs, sources):
        assert len(tids) == len(specs) == len(results) == len(miscs) == len(sources)
        rval = []
        for tid, spec, result, misc, source in zip(
            tids, specs, results, miscs, sources
        ):
            doc = dict(
                version=0,
                tid=tid,
                spec=spec,
                result=result,
                misc=misc,
                state=source["state"],
                exp_key=source["exp_key"],
                owner=source["owner"],
                book_time=source["book_time"],
                refresh_time=source["refresh_time"],
            )
            # -- ensure that misc has the following fields,
            #    some of which may already by set correctly.
            assign = ("tid", tid), ("cmd", None), ("from_tid", source["tid"])
            for k, v in assign:
                assert doc["misc"].setdefault(k, v) == v
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
            queue = [doc for doc in trials if doc["state"] == arg]
        elif hasattr(arg, "__iter__"):
            states = set(arg)
            assert all([x in JOB_STATES for x in states])
            queue = [doc for doc in trials if doc["state"] in states]
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
            exp_trials = [
                tt for tt in self._dynamic_trials if tt["exp_key"] == self._exp_key
            ]
        else:
            exp_trials = self._dynamic_trials
        return self.count_by_state_synced(arg, trials=exp_trials)

    def losses(self, bandit=None):
        if bandit is None:
            return [r.get("loss") for r in self.results]
        return list(map(bandit.loss, self.results, self.specs))

    def statuses(self, bandit=None):
        if bandit is None:
            return [r.get("status") for r in self.results]
        return list(map(bandit.status, self.results, self.specs))

    def average_best_error(self, bandit=None):
        """Return the average best error of the experiment

        Average best error is defined as the average of bandit.true_loss,
        weighted by the probability that the corresponding bandit.loss is best.

        For domains with loss measurement variance of 0, this function simply
        returns the true_loss corresponding to the result with the lowest loss.
        """

        if bandit is None:
            results = self.results
            loss = [r["loss"] for r in results if r["status"] == STATUS_OK]
            loss_v = [
                r.get("loss_variance", 0) for r in results if r["status"] == STATUS_OK
            ]
            true_loss = [
                r.get("true_loss", r["loss"])
                for r in results
                if r["status"] == STATUS_OK
            ]
        else:

            def fmap(f):
                rval = np.asarray(
                    [
                        f(r, s)
                        for (r, s) in zip(self.results, self.specs)
                        if bandit.status(r) == STATUS_OK
                    ]
                ).astype("float")
                if not np.all(np.isfinite(rval)):
                    raise ValueError()
                return rval

            loss = fmap(bandit.loss)
            loss_v = fmap(bandit.loss_variance)
            true_loss = fmap(bandit.true_loss)
        loss3 = list(zip(loss, loss_v, true_loss))
        if not loss3:
            raise ValueError("Empty loss vector")
        loss3.sort()
        loss3 = np.asarray(loss3)
        if np.all(loss3[:, 1] == 0):
            best_idx = np.argmin(loss3[:, 0])
            return loss3[best_idx, 2]
        else:
            cutoff = 0
            sigma = np.sqrt(loss3[0][1])
            while cutoff < len(loss3) and loss3[cutoff][0] < loss3[0][0] + 3 * sigma:
                cutoff += 1
            pmin = pmin_sampled(loss3[:cutoff, 0], loss3[:cutoff, 1])
            avg_true_loss = (pmin * loss3[:cutoff, 2]).sum()
            return avg_true_loss

    @property
    def best_trial(self):
        """
        Trial with lowest non-NaN loss and status=STATUS_OK.
        If no such trial exists, returns None.
        """
        candidates = [
            t
            for t in self.trials
            if t["result"]["status"] == STATUS_OK and not np.isnan(t["result"]["loss"])
        ]
        if not candidates:
            raise AllTrialsFailed
        losses = [float(t["result"]["loss"]) for t in candidates]
        if len(losses) == 0:
            return None
        best = np.nanargmin(losses)
        return candidates[best]

    @property
    def argmin(self):
        best_trial = self.best_trial
        vals = best_trial["misc"]["vals"]
        # unpack the one-element lists to values
        # and skip over the 0-element lists
        rval = {}
        for k, v in list(vals.items()):
            if v:
                rval[k] = v[0]
        return rval

    def fmin(
        self,
        fn,
        space,
        algo=None,
        max_evals=None,
        timeout=None,
        loss_threshold=None,
        max_queue_len=1,
        rstate=None,
        verbose=False,
        pass_expr_memo_ctrl=None,
        catch_eval_exceptions=False,
        return_argmin=True,
        show_progressbar=True,
        early_stop_fn=None,
        trials_save_file="",
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

        show_progressbar : bool or context manager, default True.
            Show a progressbar. See `hyperopt.progress` for customizing progress reporting.

        """
        # -- Stop-gap implementation!
        #    fmin should have been a Trials method in the first place
        #    but for now it's still sitting in another file.
        from .fmin import fmin

        return fmin(
            fn,
            space,
            algo=algo,
            max_evals=max_evals,
            timeout=timeout,
            loss_threshold=loss_threshold,
            trials=self,
            rstate=rstate,
            verbose=verbose,
            max_queue_len=max_queue_len,
            allow_trials_fmin=False,  # -- prevent recursion
            pass_expr_memo_ctrl=pass_expr_memo_ctrl,
            catch_eval_exceptions=catch_eval_exceptions,
            return_argmin=return_argmin,
            show_progressbar=show_progressbar,
            early_stop_fn=early_stop_fn,
            trials_save_file=trials_save_file,
        )


def trials_from_docs(docs, validate=True, **kwargs):
    """Construct a Trials base class instance from a list of trials documents"""
    rval = Trials(**kwargs)
    if validate:
        rval.insert_trial_docs(docs)
    else:
        rval._insert_trial_docs(docs)
    rval.refresh()
    return rval


class Ctrl:
    """Control object for interruptible, checkpoint-able evaluation"""

    info = logger.info
    warn = logger.warning
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
            self.current_trial["result"] = r

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
        new_trials = self.trials.source_trial_docs(
            tids=new_tids, specs=specs, results=results, miscs=miscs, sources=[trial]
        )
        for t in new_trials:
            t["state"] = JOB_STATE_DONE
        return self.trials.insert_trial_docs(new_trials)


class Domain:
    """Picklable representation of search space and evaluation function."""

    rec_eval_print_node_on_error = False

    # -- the Ctrl object is not used directly, but rather
    #    a live Ctrl instance is inserted for the pyll_ctrl
    #    in self.evaluate so that it can be accessed from within
    #    the pyll graph describing the search space.
    pyll_ctrl = pyll.as_apply(Ctrl)

    def __init__(
        self,
        fn,
        expr,
        workdir=None,
        pass_expr_memo_ctrl=None,
        name=None,
        loss_target=None,
    ):
        """
        Parameters
        ----------

        fn : callable
            This stores the `fn` argument to `fmin`. (See `hyperopt.fmin.fmin`)

        expr : hyperopt.pyll.Apply
            This is the `space` argument to `fmin`. (See `hyperopt.fmin.fmin`)

        workdir : string (or None)
            If non-None, the current working directory will be `workdir`while
            `expr` and `fn` are evaluated. (XXX Currently only respected by
            jobs run via MongoWorker)

        pass_expr_memo_ctrl : bool
            If True, `fn` will be called like this:
            `fn(self.expr, memo, ctrl)`,
            where `memo` is a dictionary mapping `Apply` nodes to their
            computed values, and `ctrl` is a `Ctrl` instance for communicating
            with a Trials database.  This lower-level calling convention is
            useful if you want to call e.g. `hyperopt.pyll.rec_eval` yourself
            in some customized way.

        name : string (or None)
            Label, used for pretty-printing.

        loss_target : float (or None)
            The actual or estimated minimum of `fn`.
            Some optimization algorithms may behave differently if their first
            objective is to find an input that achieves a certain value,
            rather than the more open-ended objective of pure minimization.
            XXX: Move this from Domain to be an fmin arg.

        """
        self.fn = fn
        if pass_expr_memo_ctrl is None:
            self.pass_expr_memo_ctrl = getattr(fn, "fmin_pass_expr_memo_ctrl", False)
        else:
            self.pass_expr_memo_ctrl = pass_expr_memo_ctrl

        self.expr = pyll.as_apply(expr)

        self.params = {}
        for node in pyll.dfs(self.expr):
            if node.name == "hyperopt_param":
                label = node.arg["label"].obj
                if label in self.params:
                    raise DuplicateLabel(label)
                self.params[label] = node.arg["obj"]

        self.loss_target = loss_target
        self.name = name

        self.workdir = workdir
        self.s_new_ids = pyll.Literal("new_ids")  # -- list at eval-time
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

        self.s_rng = pyll.Literal("rng-placeholder")
        # -- N.B. operates inplace:
        self.s_idxs_vals = recursive_set_rng_kwarg(
            pyll.scope.pos_args(idxs_by_label, vals_by_label), self.s_rng
        )

        # -- raises an exception if no topological ordering exists
        pyll.toposort(self.s_idxs_vals)

        # -- Protocol for serialization.
        #    self.cmd indicates to e.g. MongoWorker how this domain
        #    should be [un]serialized.
        #    XXX This mechanism deserves review as support for ipython
        #        workers improves.
        self.cmd = ("domain_attachment", "FMinIter_Domain")

    def memo_from_config(self, config):
        memo = {}
        for node in pyll.dfs(self.expr):
            if node.name == "hyperopt_param":
                label = node.arg["label"].obj
                # -- hack because it's not really garbagecollected
                #    this does have the desired effect of crashing the
                #    function if rec_eval actually needs a value that
                #    the the optimization algorithm thought to be unnecessary
                memo[node] = config.get(label, pyll.base.GarbageCollected)
        return memo

    def evaluate(self, config, ctrl, attach_attachments=True):
        memo = self.memo_from_config(config)
        use_obj_for_literal_in_memo(self.expr, ctrl, Ctrl, memo)
        if self.pass_expr_memo_ctrl:
            rval = self.fn(expr=self.expr, memo=memo, ctrl=ctrl)
        else:
            # -- the "work" of evaluating `config` can be written
            #    either into the pyll part (self.expr)
            #    or the normal Python part (self.fn)
            pyll_rval = pyll.rec_eval(
                self.expr,
                memo=memo,
                print_node_on_error=self.rec_eval_print_node_on_error,
            )
            rval = self.fn(pyll_rval)

        if isinstance(rval, (float, int, np.number)):
            dict_rval = {"loss": float(rval), "status": STATUS_OK}
        else:
            dict_rval = dict(rval)
            status = dict_rval["status"]
            if status not in STATUS_STRINGS:
                raise InvalidResultStatus(dict_rval)

            if status == STATUS_OK:
                # -- make sure that the loss is present and valid
                try:
                    dict_rval["loss"] = float(dict_rval["loss"])
                except (TypeError, KeyError):
                    raise InvalidLoss(dict_rval)

        if attach_attachments:
            attachments = dict_rval.pop("attachments", {})
            for key, val in list(attachments.items()):
                ctrl.attachments[key] = val

        # -- don't do this here because SON-compatibility is only a requirement
        #    for trials destined for a mongodb. In-memory rvals can contain
        #    anything.
        return dict_rval

    def evaluate_async(self, config, ctrl, attach_attachments=True):
        """
        this is the first part of async evaluation for ipython parallel engines (see ipy.py)

        This breaks evaluate into two parts to allow for the apply_async call
        to only pass the objective function and arguments.
        """
        memo = self.memo_from_config(config)
        use_obj_for_literal_in_memo(self.expr, ctrl, Ctrl, memo)
        if self.pass_expr_memo_ctrl:
            pyll_rval = self.fn(expr=self.expr, memo=memo, ctrl=ctrl)
        else:
            # -- the "work" of evaluating `config` can be written
            #    either into the pyll part (self.expr)
            #    or the normal Python part (self.fn)
            pyll_rval = pyll.rec_eval(
                self.expr,
                memo=memo,
                print_node_on_error=self.rec_eval_print_node_on_error,
            )
            return (self.fn, pyll_rval)

    def evaluate_async2(self, rval, ctrl, attach_attachments=True):
        """
        this is the second part of async evaluation for ipython parallel engines (see ipy.py)
        """
        if isinstance(rval, (float, int, np.number)):
            dict_rval = {"loss": float(rval), "status": STATUS_OK}
        else:
            dict_rval = dict(rval)
            status = dict_rval["status"]
            if status not in STATUS_STRINGS:
                raise InvalidResultStatus(dict_rval)

            if status == STATUS_OK:
                # -- make sure that the loss is present and valid
                try:
                    dict_rval["loss"] = float(dict_rval["loss"])
                except (TypeError, KeyError):
                    raise InvalidLoss(dict_rval)

        if attach_attachments:
            attachments = dict_rval.pop("attachments", {})
            for key, val in list(attachments.items()):
                ctrl.attachments[key] = val

        # -- don't do this here because SON-compatibility is only a requirement
        #    for trials destined for a mongodb. In-memory rvals can contain
        #    anything.
        return dict_rval

    def short_str(self):
        return "Domain{%s}" % str(self.fn)

    def loss(self, result, config=None):
        """Extract the scalar-valued loss from a result document"""
        return result.get("loss", None)

    def loss_variance(self, result, config=None):
        """Return the variance in the estimate of the loss"""
        return result.get("loss_variance", 0.0)

    def true_loss(self, result, config=None):
        """Return a true loss, in the case that the `loss` is a surrogate"""
        # N.B. don't use get() here, it evaluates self.loss un-necessarily
        try:
            return result["true_loss"]
        except KeyError:
            return self.loss(result, config=config)

    def true_loss_variance(self, config=None):
        """Return the variance in  true loss,
        in the case that the `loss` is a surrogate.
        """
        raise NotImplementedError()

    def status(self, result, config=None):
        """Extract the job status from a result document"""
        return result["status"]

    def new_result(self):
        """Return a JSON-encodable object
        to serve as the 'result' for new jobs.
        """
        return {"status": STATUS_NEW}


# -- flake8 doesn't like blank last line
