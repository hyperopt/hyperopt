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
import logging

import numpy as np

from bson import SON  # -- from pymongo

import pyll
from pyll.stochastic import replace_repeat_stochastic
from pyll.stochastic import replace_implicit_stochastic_nodes

from .utils import pmin_sampled
from .vectorize import VectorizeHelper, pretty_names

logger = logging.getLogger(__name__)


# -- STATUS values
# These are used to store job status in a backend-agnostic way, for the
# purpose of communicating between Bandit, BanditAlgo, and any
# visualization/monitoring code.

STATUS_STRINGS = (
    'new',        # computations have not started
    'running',    # computations are in prog
    'suspended',  # computations have been suspended, job is not finished
    'ok',         # computations are finished, terminated normally
    'fail')       # computations are finished, terminated with error
                  #   - result['status_fail'] should contain more info

# -- named constants for status possibilities
STATUS_NEW = 'new'
STATUS_RUNNING = 'running'
STATUS_SUSPENDED = 'suspended'
STATUS_OK = 'ok'
STATUS_FAIL = 'fail'


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
                misc_tid['idxs'][node_id] = [tid]
                pos = node_idxs.index(tid)
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

    def __init__(self):
        self._trials = []
        self.refresh()

    def __iter__(self):
        return iter(self._trials)

    def __len__(self):
        return len(self._trials)

    def refresh(self):
        # any syncing to persistent storage would happen here
        self._specs = [tt['spec'] for tt in self._trials]
        self._results = [tt['result'] for tt in self._trials]
        self._miscs = [tt['misc'] for tt in self._trials]

    @property
    def specs(self):
        return self._specs

    @property
    def results(self):
        return self._results

    @property
    def miscs(self):
        return self._miscs

    @property
    def idxs(self):
        return miscs_to_idxs_vals(self._miscs)[0]

    @property
    def vals(self):
        return miscs_to_idxs_vals(self._miscs)[1]

    def assert_valid_trial(self, trial):
        if not (hasattr(trial, 'keys') and hasattr(trial, 'values')):
            raise InvalidTrial('trial should be dict-like', trial)
        for key in 'tid', 'spec', 'result', 'misc':
            if key not in trial:
                raise InvalidTrial('trial missing key', key)
        for key in 'tid', 'idxs', 'vals':
            if key not in trial['misc']:
                raise InvalidTrial('trial["misc"] missing key', key)
        if trial['tid'] != trial['misc']['tid']:
            raise InvalidTrial('tid mismatch between root and misc',
                    (trial['tid'], trial['misc']['tid']))
        # XXX check for SON-encodable
        # XXX how to assert that tids are unique?

    def _insert_trial_docs(self, docs):
        """insert with no error checking
        """
        rval = [doc['tid'] for doc in docs]
        self._trials.extend(docs)
        return rval

    def insert_trial_doc(self, doc):
        """insert trial after error checking

        Does not refresh. Call self.refresh() for the trial to appear in
        self.specs, self.results, etc.
        """
        self.assert_valid_trial(doc)
        return self._insert_trial_docs([doc])[0]
        # refreshing could be done fast in this base implementation, but with
        # a real DB the steps should be separated.

    def insert_trial_docs(self, docs):
        """ trials - something like is returned by self.new_trials()
        """
        for doc in docs:
            self.assert_valid_trial(doc)
        return self._insert_trial_docs(docs)

    def new_trial_ids(self, N):
        return range(
                len(self._trials),
                len(self._trials) + N)

    def new_trials(self, tids, specs, results, miscs):
        assert len(tids) == len(specs) == len(results) == len(miscs)
        rval = []
        for tid, spec, result, misc in zip(tids, specs, results, miscs):
            trial = dict(tid=tid, spec=spec, result=result, misc=misc)
            rval.append(trial)
        return rval

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

    def __init__(self):
        # -- attachments should be used like
        #      attachments[key]
        #      attachments[key] = value
        #    where key and value are strings. Client code should not
        #    expect any dictionary-like behaviour beyond that (no update)
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
        ctrl = Ctrl()
        config = self.dryrun_config()
        return self.evaluate(config, ctrl)


class CoinFlip(Bandit):
    """ Possibly the simplest possible Bandit implementation
    """

    def __init__(self):
        Bandit.__init__(self, dict(flip=pyll.scope.one_of('heads', 'tails')))

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
        idxs_by_id = vh.idxs_by_id()
        vals_by_id = vh.vals_by_id()
        name_by_id = vh.name_by_id()
        assert set(idxs_by_id.keys()) == set(vals_by_id.keys())
        assert set(name_by_id.keys()) == set(vals_by_id.keys())

        # -- remove non-stochastic nodes from the idxs and vals
        #    because (a) they should be irrelevant for BanditAlgo operation
        #    and (b) they can be reconstructed from the template and the
        #    stochastic choices.
        for node_id, name in name_by_id.items():
            if name not in pyll.stochastic.implicit_stochastic_symbols:
                del name_by_id[node_id]
                del vals_by_id[node_id]
                del idxs_by_id[node_id]
            if name == 'one_of':
                # -- one_of nodes too, because they are duplicates of randint
                del name_by_id[node_id]
                del vals_by_id[node_id]
                del idxs_by_id[node_id]

        # -- make the graph runnable
        specs_idxs_vals_0 = pyll.as_apply([
            vh.vals_memo[template], idxs_by_id, vals_by_id])
        specs_idxs_vals_1 = replace_repeat_stochastic(specs_idxs_vals_0)
        specs_idxs_vals_2, lrng = replace_implicit_stochastic_nodes(
                specs_idxs_vals_1,
                pyll.as_apply(self.rng))

        # -- represents symbolic specs/idxs/vals
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
            if cnode in vh.node_id and vh.node_id[cnode] in name_by_id:
                doc_coords[vh.node_id[cnode]] = pname
            else:
                #print 'DROPPING', node
                pass
        #print 'DOC_COORDS'
        #print doc_coords

    def short_str(self):
        return self.__class__.__name__

    def suggest(self,
            new_ids,
            specs,
            results,
            misc):
        """
        specs is list of all specification documents from current Trial
        results is a list of result documents returned by Bandit.evaluate
        misc is a list of documents with other information about each job.

        All lists have the same length.
        """
        raise NotImplementedError('override me')


class Random(BanditAlgo):
    """Random search algorithm
    """

    def suggest(self,
            new_ids,
            specs,
            results,
            misc):
        # -- install new_ids as program arguments
        self.new_ids[:] = new_ids
        # -- sample new specs, idxs, vals
        new_specs, idxs, vals = pyll.rec_eval(self.s_specs_idxs_vals)
        # -- normally the caller would do this checking
        #    but just so the reader knows what this function is supposed
        #    to return...
        new_results = [self.bandit.new_result() for ii in new_ids]
        new_miscs = [dict(tid=ii) for ii in new_ids]
        miscs_update_idxs_vals(new_miscs, idxs, vals)
        return new_specs, new_results, new_miscs


class Experiment(object):
    """Object for conducting search experiments.
    """
    max_queue_len = 1
    poll_interval_secs = 0.5

    def __init__(self, trials, bandit_algo, async=False):
        self.trials = trials
        self.bandit_algo = bandit_algo
        self.bandit = bandit_algo.bandit
        self.async = async

    def queue_len(self):
        if self.async:
            raise NotImplementedError('override-me')
        else:
            return len([tt for tt in self.trials._trials
                if tt['serial_status'] == 'TODO'])

    # -- override this method in async. experiment to be no-op
    def serial_evaluate(self):
        if self.async:
            time.sleep(self.poll_interval_secs)
        else:
            for trial in self.trials:
                if trial['serial_status'] == 'TODO':
                    spec = trial['spec']
                    ctrl = Ctrl() # TODO - give access to self.trials
                    result = self.bandit.evaluate(spec, ctrl)
                    # XXX verify result is SON-encodable
                    trial['result'] = result
                    trial['serial_status'] = 'DONE'
            self.trials.refresh()


    def block_until_done(self):
        if self.async:
            raise NotImplementedError()
        else:
            self.serial_evaluate()

    def enqueue(self, new_trials):
        for trial in new_trials:
            assert 'serial_status' not in trial
            trial['serial_status'] = 'TODO'
            self.trials.insert_trial_doc(trial)

    def run(self, N, block_until_done=True):
        trials = self.trials
        algo = self.bandit_algo
        bandit = algo.bandit
        n_queued = 0

        self.trials.refresh()
        while n_queued < N:
            while self.queue_len() < self.max_queue_len:
                n_to_enqueue = self.max_queue_len - self.queue_len()
                new_ids = trials.new_trial_ids(n_to_enqueue)
                new_specs, new_results, new_miscs = algo.suggest(
                        new_ids,
                        trials.specs, trials.results,
                        trials.miscs)
                new_trials = trials.new_trials(new_ids,
                        new_specs, new_results, new_miscs)
                self.enqueue(new_trials)
                self.trials.refresh()
                n_queued += len(new_ids)
            self.serial_evaluate()

        if block_until_done:
            self.block_until_done()
            self.trials.refresh()
            logger.info('Queue empty, exiting run.')
        else:
            msg = 'Exiting run, not waiting for %d jobs.' % self.queue_len()
            logger.info(msg)

