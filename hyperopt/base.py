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
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import copy
from itertools import izip
import logging
import time

import numpy as np

import bson # -- comes with pymongo

import pyll
from pyll import scope
from pyll.stochastic import replace_repeat_stochastic
from pyll.stochastic import replace_implicit_stochastic_nodes

from .utils import pmin_sampled
from .vectorize import VectorizeHelper, pretty_names

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


def SONify(arg, memo=None):
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]
    if isinstance(arg, np.floating):
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
        if x.ndim == 0:
            rval = SONify(x.sum())
        elif x.ndim == 1:
            rval = map(np_to_py_number, x) # N.B. memo None
        else:
            raise NotImplementedError()
    else:
        raise TypeError('SONify', arg)
    memo[id(rval)] = rval
    return rval


def miscs_update_idxs_vals(miscs, idxs, vals):
    """
    Unpack the idxs-vals format into the list of dictionaries that is
    `misc`.
    """
    assert set(idxs.keys()) == set(vals.keys())
    misc_by_id = dict([(m['tid'], m) for m in miscs])

    # -- assert that the idxs and vals correspond to the misc docs
    all_ids = set()
    for idxlist in idxs.values():
        all_ids.update(idxlist)
    assert all_ids == set(misc_by_id.keys())

    for tid, misc_tid in misc_by_id.items():
        misc_tid['idxs'] = {}
        misc_tid['vals'] = {}
        for node_id in idxs:
            node_idxs = list(idxs[node_id])
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


def miscs_to_idxs_vals(miscs):
    idxs = copy.deepcopy(miscs[0]['idxs'])
    vals = copy.deepcopy(miscs[0]['vals'])
    for misc in miscs[1:]:
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

    def __init__(self, exp_key=None):
        self._ids = set()
        self._dynamic_trials = []
        self._exp_key = exp_key
        self.attachments = {}
        self.refresh()

    def __iter__(self):
        return iter(self._trials)

    def __len__(self):
        return len(self._trials)

    def refresh(self):
        # any syncing to persistent storage would happen here
        self._trials = list(self._dynamic_trials)
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
            print '-' * 80
            print trial
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
            states = set(states)
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
        return self.count_by_state_synced(arg, trials=self._dynamic_trials)

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


class Ctrl(object):
    """Control object for interruptible, checkpoint-able evaluation
    """
    info = logger.info
    warn = logger.warn
    error = logger.error
    debug = logger.debug

    def __init__(self, trials):
        # -- attachments should be used like
        #      attachments[key]
        #      attachments[key] = value
        #    where key and value are strings. Client code should not
        #    expect any dictionary-like behaviour beyond that (no update)
        self.trials = trials
        self.attachments = {}

    def checkpoint(self, r=None):
        pass


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
        template = pyll.clone(self.template)
        runnable, lrng = pyll.stochastic.replace_implicit_stochastic_nodes(
                template, pyll.as_apply(rng))
        rval = pyll.rec_eval(runnable)
        return rval

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


class BanditAlgo(object):
    """
    Algorithm for solving Config-armed bandit (arms are from tree domain)

    X-armed bandit problems, and N-armed bandit problems are special cases.

    :type bandit: Bandit
    :param bandit: the bandit problem this algorithm should solve

    """
    seed = 123

    def __init__(self, bandit):
        self.bandit = bandit
        self.rng = np.random.RandomState(self.seed)
        self.new_ids = ['dummy_id']
        # -- N.B. not necessarily actually a range
        idx_range = pyll.Literal(self.new_ids)
        self.template_clone_memo = {}
        template = pyll.clone(self.bandit.template, self.template_clone_memo)
        vh = self.vh = VectorizeHelper(template, idx_range)
        vh.build_idxs()
        vh.build_vals()
        # the keys (nid) here are strings like 'node_5'
        idxs_by_nid = self.idxs_by_nid = vh.idxs_by_id()
        vals_by_nid = self.vals_by_nid = vh.vals_by_id()
        name_by_nid = self.name_by_nid = vh.name_by_id()
        assert set(idxs_by_nid.keys()) == set(vals_by_nid.keys())
        assert set(name_by_nid.keys()) == set(vals_by_nid.keys())

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
        specs_idxs_vals_0 = pyll.as_apply([
            vh.vals_memo[template], idxs_by_nid, vals_by_nid])
        specs_idxs_vals_1 = replace_repeat_stochastic(specs_idxs_vals_0)
        specs_idxs_vals_2, lrng = replace_implicit_stochastic_nodes(
                specs_idxs_vals_1,
                pyll.as_apply(self.rng))

        # -- represents symbolic (specs, idxs, vals)
        self.s_specs_idxs_vals = specs_idxs_vals_2

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

    def suggest(self, new_ids, specs, results, miscs):
        """
        specs is list of all specification documents from current Trial
        results is a list of result documents returned by Bandit.evaluate
        miscs is a list of documents with other information about each job.

        All lists have the same length.
        """
        # -- install new_ids as program arguments
        self.new_ids[:] = new_ids

        # XXX: use the ids to seed the random number generator
        #      to avoid suggesting duplicates without having to resort to
        #      rejection sampling.

        # -- sample new specs, idxs, vals
        new_specs, idxs, vals = pyll.rec_eval(self.s_specs_idxs_vals)
        new_results = [self.bandit.new_result() for ii in new_ids]
        new_miscs = [dict(tid=ii) for ii in new_ids]
        miscs_update_idxs_vals(new_miscs, idxs, vals)
        return new_specs, new_results, new_miscs


class Random(BanditAlgo):
    """Random search algorithm

    The base implementation of BanditAlgo actually does random sampling,
    This class is defined so that hyperopt.Random can be used to mean random
    sampling.
    """


class Experiment(object):
    """Object for conducting search experiments.
    """
    def __init__(self, trials, bandit_algo, async=None, cmd=None,
            max_queue_len=1,
            poll_interval_secs=1.0,
            workdir=None,
            ):
        self.trials = trials
        self.bandit_algo = bandit_algo
        self.bandit = bandit_algo.bandit
        if async is None:
            self.async = trials.async
        else:
            self.async = async
        self.cmd = cmd
        self.workdir = workdir
        self.poll_interval_secs = poll_interval_secs
        self.max_queue_len = max_queue_len

    def serial_evaluate(self):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == JOB_STATE_NEW:
                spec = copy.deepcopy(trial['spec'])
                ctrl = Ctrl(self.trials)
                try:
                    result = self.bandit.evaluate(spec, ctrl)
                except Exception, e:
                    logger.info('job exception: %s' % str(e))
                    trial['state'] = JOB_STATE_ERROR
                    trial['misc']['error'] = (str(type(e)), str(e))
                else:
                    logger.debug('job returned: %s' % str(result))
                    trial['state'] = JOB_STATE_DONE
                    trial['result'] = result
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
        trials = self.trials
        algo = self.bandit_algo
        bandit = algo.bandit
        n_queued = 0

        def get_queue_len():
            return self.trials.count_by_state_unsynced(JOB_STATE_NEW)

        while n_queued < N:
            qlen = get_queue_len()
            while qlen < self.max_queue_len and n_queued < N:
                n_to_enqueue = min(self.max_queue_len - qlen, N - n_queued)
                new_ids = trials.new_trial_ids(n_to_enqueue)
                self.trials.refresh()
                new_specs, new_results, new_miscs = algo.suggest(
                        new_ids,
                        trials.specs, trials.results,
                        trials.miscs)
                new_trials = trials.new_trial_docs(new_ids,
                        new_specs, new_results, new_miscs)
                for doc in new_trials:
                    assert 'cmd' not in doc['misc']
                    doc['misc']['cmd'] = self.cmd
                    if self.workdir:
                        assert 'workdir' not in doc['misc']
                        doc['misc']['workdir'] = self.workdir
                self.trials.insert_trial_docs(new_trials)
                n_queued += len(new_ids)
                qlen = get_queue_len()

            if self.async:
                # -- wait for workers to fill in the trials
                time.sleep(self.poll_interval_secs)
            else:
                # -- loop over trials and do the jobs directly
                self.serial_evaluate()

        if block_until_done:
            self.block_until_done()
            self.trials.refresh()
            logger.info('Queue empty, exiting run.')
        else:
            qlen = get_queue_len()
            msg = 'Exiting run, not waiting for %d jobs.' % qlen
            logger.info(msg)
            print msg

