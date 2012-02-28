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

__authors__   = "James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import copy
import hashlib
import logging
import time
import datetime
import sys

import numpy as np

import bson # -- comes with pymongo
from bson.objectid import ObjectId

import pyll
from pyll import scope
from pyll.stochastic import recursive_set_rng_kwarg

from .utils import pmin_sampled
from .vectorize import VectorizeHelper
from .vectorize import pretty_names
from .vectorize import replace_repeat_stochastic

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
JOB_STATES= [JOB_STATE_NEW,
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
        'vals',
        ]


def StopExperiment(*args, **kwargs):
    """ Return StopExperiment

    StopExperiment is a symbol used as a special return value from
    BanditAlgo.suggest. With this implementation both `return StopExperiment`
    and `return StopExperiment()` have the same effect.
    """
    return StopExperiment


def SONify(arg, memo=None):
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
        rval = dict([(SONify(k, memo), SONify(v, memo))
            for k, v in arg.items()])
    elif isinstance(arg, (basestring, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = SONify(arg.sum())
        else:
            rval = map(SONify, arg) # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    else:
        raise TypeError('SONify', arg)
    memo[id(rval)] = rval
    return rval


def miscs_update_idxs_vals(miscs, idxs, vals, assert_all_vals_used=True,
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


class InvalidTrial(Exception):
    pass


class Trials(object):
    """
    Trials are documents (dict-like) with *at least* the following keys:
        - spec: an instantiation of a Bandit template
        - tid: a unique trial identification integer within `self.trials`
        - result: sub-document returned by Bandit.evaluate
        - idxs:  sub-document mapping stochastic node names
                    to either [] or [tid]
        - vals:  sub-document mapping stochastic node names
                    to either [] or [<val>]
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
        Support syntax for load:  self.attachments[name]
        Support syntax for store: self.attachments[name] = value
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
        raise NotImplementedError('how to make it obvious whether'
                ' indexing is by _trials position or by tid?')

    def refresh(self):
        # In MongoTrials, this method fetches from database
        if self._exp_key is None:
            self._trials = [tt for tt in self._dynamic_trials
                if tt['state'] != JOB_STATE_ERROR]
        else:
            self._trials = [tt for tt in self._dynamic_trials
                if (tt['state'] != JOB_STATE_ERROR
                    and tt['exp_key'] == self._exp_key
                    )]
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
            raise InvalidTrial('tid mismatch between root and misc',
                    (trial['tid'], trial['misc']['tid']))
        # -- check for SON-encodable
        try:
            bson.BSON.encode(trial)
        except:
            # TODO: save the trial object somewhere to inspect, fix, re-insert, etc.
            print '-' * 80
            print "CANT ENCODE"
            print '-' * 80
            raise
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
        assert len(tids) == len(specs) == len(results) == len(miscs) == len(sources)
        rval = []
        for tid, spec, result, misc, source in zip(tids, specs, results, miscs, sources):
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
            exp_trials = [tt for tt in self._dynamic_trials
                if tt['exp_key'] == self._exp_key]
        else:
            exp_trials = self._dynamic_trials
        return self.count_by_state_synced(arg, trials=exp_trials)

    def losses(self, bandit=None):
        if bandit is None:
            bandit = Bandit(None)
        return map(bandit.loss, self.results, self.specs)

    def statuses(self, bandit=None):
        if bandit is None:
            bandit = Bandit(None)
        return map(bandit.status, self.results, self.specs)

    def average_best_error(self, bandit=None):
        """Return the average best error of the experiment

        Average best error is defined as the average of bandit.true_loss,
        weighted by the probability that the corresponding bandit.loss is best.

        For bandits with loss measurement variance of 0, this function simply
        returns the true_loss corresponding to the result with the lowest loss.
        """
        if bandit is None:
            bandit = Bandit(None)

        def fmap(f):
            rval = np.asarray([f(r, s)
                    for (r, s) in zip(self.results, self.specs)
                    if bandit.status(r) == STATUS_OK]).astype('float')
            if not np.all(np.isfinite(rval)):
                raise ValueError()
            return rval
        loss = fmap(bandit.loss)
        loss_v = fmap(bandit.loss_variance)
        if bandit.true_loss is not Bandit.true_loss:
            true_loss = fmap(bandit.true_loss)
            loss3 = zip(loss, loss_v, true_loss)
        else:
            loss3 = zip(loss, loss_v, loss)
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
        pass

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

        new_tids can be None, in which case new tids will be generated automatically

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

    def __init__(self, template):
        self.template = pyll.as_apply(template)

    def short_str(self):
        return self.__class__.__name__

    def dryrun_config(self):
        """Return a point that could have been drawn from the template
        that is useful for small trial debugging.
        """
        rng = np.random.RandomState(1)
        return pyll.stochastic.sample(self.template, rng)

    def evaluate(self, config, ctrl):
        """Return a result document
        """
        raise NotImplementedError('override me')

    def loss(self, result, config=None):
        """Extract the scalar-valued loss from a result document
        """
        try:
            return result['loss']
        except KeyError:
            return None

    def loss_variance(self, result, config=None):
        """Return the variance in the estimate of the loss"""
        return 0

    def true_loss(self, result, config=None):
        """Return a true loss, in the case that the `loss` is a surrogate"""
        return self.loss(result, config=config)

    def true_loss_variance(self, config=None):
        """Return the variance in  true loss,
        in the case that the `loss` is a surrogate.
        """
        return 0

    def loss_target(self):
        raise NotImplementedError('override-me')

    def status(self, result, config=None):
        """Extract the job status from a result document
        """
        return result['status']

    def new_result(self):
        """Return a JSON-encodable object
        to serve as the 'result' for new jobs.
        """
        return {'status': STATUS_NEW}

    @classmethod
    def main_dryrun(cls):
        self = cls()
        ctrl = Ctrl(Trials())
        config = self.dryrun_config()
        return self.evaluate(config, ctrl)


class CoinFlip(Bandit):
    """ Possibly the simplest possible Bandit implementation
    """

    def __init__(self):
        Bandit.__init__(self, dict(flip=scope.one_of('heads', 'tails')))

    def evaluate(self, config, ctrl):
        scores = dict(heads=1.0, tails=0.0)
        return dict(loss=scores[config['flip']], status=STATUS_OK)


class CoinFlipInjector(Bandit):

    def __init__(self):
        Bandit.__init__(self, dict(flip=scope.one_of('heads', 'tails')))

    def evaluate(self, config, ctrl):
        scores = dict(heads=1.0, tails=0.0)
        
        reverse = lambda x : 'tails' if x == 'heads' else 'heads'
        
        other_spec = dict(flip=reverse(config['flip']))
        other_result = dict(status=STATUS_OK,
                            loss=scores[other_spec['flip']])
        other_misc = {'idxs':None, 'vals':None}
        ctrl.inject_results([other_spec], [other_result], [other_misc])
        
        return dict(loss=scores[config['flip']], status=STATUS_OK)


class BanditAlgo(object):
    """
    Algorithm for solving Config-armed bandit (arms are from tree domain)

    X-armed bandit problems, and N-armed bandit problems are special cases.

    :type bandit: Bandit
    :param bandit: the bandit problem this algorithm should solve

    :param cmd: a pair used by MongoWorker to know how to evaluate suggestions
    :param workdir: optional hint to MongoWorker where to store temp files.

    """
    seed = 123

    def __init__(self, bandit, seed=seed, cmd=None, workdir=None):
        self.bandit = bandit
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.cmd = cmd
        self.workdir = workdir
        self.new_ids = ['dummy_id']
        # -- N.B. not necessarily actually a range
        self.s_new_ids = pyll.Literal(self.new_ids)
        self.template_clone_memo = {}
        template = pyll.clone(self.bandit.template, self.template_clone_memo)
        vh = self.vh = VectorizeHelper(template, self.s_new_ids)
        vh.build_idxs()
        vh.build_vals()
        # the keys (nid) here are strings like 'node_5'
        idxs_by_nid = vh.idxs_by_id()
        vals_by_nid = vh.vals_by_id()
        name_by_nid = vh.name_by_id()
        assert set(idxs_by_nid.keys()) == set(vals_by_nid.keys())
        assert set(name_by_nid.keys()) == set(vals_by_nid.keys())

        # -- replace repeat(dist(...)) with vectorized versions
        t_i_v = replace_repeat_stochastic(
                pyll.as_apply([
                    vh.vals_memo[template], idxs_by_nid, vals_by_nid]))
        assert t_i_v.name == 'pos_args'
        template, s_idxs_by_nid, s_vals_by_nid = t_i_v.pos_args
        # -- fetch the dictionaries off the top of the cloned graph
        idxs_by_nid = dict(s_idxs_by_nid.named_args)
        vals_by_nid = dict(s_vals_by_nid.named_args)

        # -- remove non-stochastic nodes from the idxs and vals
        #    because
        #    (a) they should be irrelevant for BanditAlgo operation,
        #    (b) they can be reconstructed from the template and the
        #    stochastic choices, and
        #    (c) they are often annoying when printing / saving.
        for node_id, name in name_by_nid.items():
            if name not in pyll.stochastic.implicit_stochastic_symbols:
                del name_by_nid[node_id]
                del vals_by_nid[node_id]
                del idxs_by_nid[node_id]
            elif name == 'one_of':
                # -- one_of nodes too, because they are duplicates of randint
                del name_by_nid[node_id]
                del vals_by_nid[node_id]
                del idxs_by_nid[node_id]

        # -- make the graph runnable and SON-encodable
        # N.B. operates inplace
        self.s_specs_idxs_vals = recursive_set_rng_kwarg(
                scope.pos_args(template, idxs_by_nid, vals_by_nid),
                pyll.as_apply(self.rng))

        self.vtemplate = template
        self.idxs_by_nid = idxs_by_nid
        self.vals_by_nid = vals_by_nid
        self.name_by_nid = name_by_nid

        # -- compute some document coordinate strings for the node_ids
        pnames = pretty_names(bandit.template, prefix=None)
        doc_coords = self.doc_coords = {}
        for node, pname in pnames.items():
            cnode = self.template_clone_memo[node]
            if cnode.name == 'one_of':
                choice_node = vh.choice_memo[cnode]
                assert choice_node.name == 'randint'
                doc_coords[vh.node_id[choice_node]] = pname + '.randint'
            if cnode in vh.node_id and vh.node_id[cnode] in name_by_nid:
                doc_coords[vh.node_id[cnode]] = pname
            else:
                #print 'DROPPING', node
                pass
        #print 'DOC_COORDS'
        #print doc_coords

    def short_str(self):
        return self.__class__.__name__

    def suggest(self, new_ids, trials):
        """
        new_ids - a list of unique identifiers (not necessarily ints!)
                  for the suggestions that this function should return.

        All lists have the same length.
        """
        # -- install new_ids as program arguments
        rval = []
        for new_id in new_ids:
            self.new_ids[:] = [new_id]

            sh1 = hashlib.sha1()
            sh1.update(str(new_id))
            self.rng.seed(int(int(sh1.hexdigest(), base=16) % (2**31)))

            # -- sample new specs, idxs, vals
            new_specs, idxs, vals = pyll.rec_eval(self.s_specs_idxs_vals)
            new_result = self.bandit.new_result()
            new_misc = dict(tid=new_id, cmd=self.cmd, workdir=self.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)
            rval.extend(trials.new_trial_docs([new_id],
                    new_specs, [new_result], [new_misc]))
        return rval


class Random(BanditAlgo):
    """Random search algorithm

    The base implementation of BanditAlgo actually does random sampling,
    This class is defined so that hyperopt.Random can be used to mean random
    sampling.
    """


class RandomStop(Random):
    """Run random search for up to `ntrials` iterations, and then stop.
    """
    def __init__(self, ntrials, *args, **kwargs):
        Random.__init__(self, *args, **kwargs)
        self.ntrials = ntrials

    def suggest(self, new_ids, trials):
        if len(trials) >= self.ntrials:
            return StopExperiment()
        else:
            return Random.suggest(self, new_ids, trials)


class Experiment(object):
    """Object for conducting search experiments.
    """
    catch_bandit_exceptions = True

    def __init__(self, trials, bandit_algo, async=None,
            max_queue_len=1,
            poll_interval_secs=1.0,
            ):
        self.trials = trials
        self.bandit_algo = bandit_algo
        self.bandit = bandit_algo.bandit
        if async is None:
            self.async = trials.async
        else:
            self.async = async
        self.poll_interval_secs = poll_interval_secs
        self.max_queue_len = max_queue_len

    def serial_evaluate(self, N=-1):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == JOB_STATE_NEW:
                spec = copy.deepcopy(trial['spec'])
                ctrl = Ctrl(self.trials, current_trial=trial)
                try:
                    result = self.bandit.evaluate(spec, ctrl)
                except Exception, e:
                    logger.info('job exception: %s' % str(e))
                    trial['state'] = JOB_STATE_ERROR
                    trial['misc']['error'] = (str(type(e)), str(e))
                    if not self.catch_bandit_exceptions:
                        raise
                else:
                    #logger.debug('job returned: %s' % str(result))
                    trial['state'] = JOB_STATE_DONE
                    trial['result'] = result

                N -= 1
                if N == 0:
                    break
        self.trials.refresh()

    def block_until_done(self):
        if self.async:
            unfinished_states = [JOB_STATE_NEW, JOB_STATE_RUNNING]

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

        bandit_algo can pass instance of StopExperiment to break out of
        enqueuing loop
        """
        trials = self.trials
        algo = self.bandit_algo
        bandit = algo.bandit
        n_queued = 0

        def get_queue_len():
            return self.trials.count_by_state_unsynced(JOB_STATE_NEW)

        stopped = False
        while n_queued < N:
            qlen = get_queue_len()
            while qlen < self.max_queue_len and n_queued < N:
                n_to_enqueue = min(self.max_queue_len - qlen, N - n_queued)
                new_ids = trials.new_trial_ids(n_to_enqueue)
                self.trials.refresh()
                new_trials = algo.suggest(new_ids, trials)
                if new_trials is StopExperiment:
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
