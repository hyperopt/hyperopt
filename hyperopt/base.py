"""Base classes / Design

The design is that there are three components fitting together in this project:

- Bandit - specifies a search problem

- BanditAlgo - an algorithm for solving a Bandit search problem

- Experiment - uses a Bandit and a BanditAlgo to carry out a search on some
               number of computers. (Includes CLI)

- Ctrl - a channel for two-way communication
         between an Experiment and Bandit.evaluate.
         Experiment subclasses may subclass Ctrl to match. For example, if an
         experiment is going to dispatch jobs in other threads, then an
         appropriate thread-aware Ctrl subclass should go with it.

- Template - an rSON hierarchy (see ht_dist2.py)

- TrialSpec - a JSON-encodable document used to specify the computation of a
  Trial.

- Result - a JSON-encodable document describing the results of a Trial.
    'status' - a string describing what happened to this trial
                (BanditAlgo-dependent, see e.g.
                theano_bandit_algos.STATUS_STRINGS)
    'loss' - a scalar saying how bad this trial was, or None if unknown / NA.

The modules communicate with trials in nested dictionary form.
TheanoBanditAlgo translates nested dictionary form into idxs, vals form.

"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import logging

import numpy as np

import pyll
from pyll.stochastic import replace_repeat_stochastic
from pyll.stochastic import replace_implicit_stochastic_nodes

from .utils import pmin_sampled
from .vectorize import VectorizeHelper

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

    def get_trials(self):
        # TODO
        raise NotImplementedError()

    def insert_trials(self):
        # TODO
        raise NotImplementedError()


class Bandit(object):
    """Specification of bandit problem.

    template - htdist2 specification of search domain

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

    @classmethod
    def evaluate(cls, config, ctrl):
        """Return a result document
        """
        raise NotImplementedError('override me')

    @classmethod
    def loss(cls, result, config=None):
        """Extract the scalar-valued loss from a result document
        """
        try:
            return result['loss']
        except KeyError:
            return None

    @classmethod
    def loss_variance(cls, result, config=None):
        """Return the variance in the estimate of the loss"""
        return 0

    @classmethod
    def true_loss(cls, result, config=None):
        """Return a true loss, in the case that the `loss` is a surrogate"""
        return cls.loss(result, config=config)

    @classmethod
    def true_loss_variance(cls, result, config=None):
        """Return the variance in  true loss,
        in the case that the `loss` is a surrogate.
        """
        return 0

    @classmethod
    def loss_target(cls):
        raise NotImplementedError('override-me')

    @classmethod
    def status(cls, result, config=None):
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
        self.idx_range = [0, 1]
        N0 = pyll.Literal(self.idx_range)
        idx_range = pyll.scope.range(N0[0], N0[1])
        template = pyll.clone(self.bandit.template)
        vh = VectorizeHelper(template, idx_range)
        vh.build_idxs()
        vh.build_vals()
        idxs_by_id = vh.idxs_by_id()
        vals_by_id = vh.vals_by_id()
        name_by_id = vh.name_by_id()
        for node_id, name in name_by_id.items():
            if name not in pyll.stochastic.implicit_stochastic_symbols:
                del name_by_id[node_id]
                del vals_by_id[node_id]
                del idxs_by_id[node_id]

        docs_idxs_vals_0 = pyll.as_apply([
            vh.vals_memo[template], idxs_by_id, vals_by_id])
        docs_idxs_vals_1 = replace_repeat_stochastic(docs_idxs_vals_0)
        docs_idxs_vals_2, lrng = replace_implicit_stochastic_nodes(
                docs_idxs_vals_1,
                pyll.as_apply(self.rng))
        # -- symbolic docs/idxs/vals
        self.s_docs_idxs_vals = docs_idxs_vals_2

    def short_str(self):
        return self.__class__.__name__

    def suggest_docs_idxs_vals(self, trials, results,
            stochastic_idxs, stochastic_vals, N):
        raise NotImplementedError('override me')


class Random(BanditAlgo):
    """Random search algorithm
    """
    def suggest_docs_idxs_vals(self, trials, results,
            stochastic_idxs,
            stochastic_vals,
            N):
        self.idx_range[1] = N
        docs, idxs, vals = pyll.rec_eval(self.s_docs_idxs_vals)
        return docs, idxs, vals


class Experiment(object):
    """Object for conducting search experiments.
    """
    def __init__(self, bandit_algo):
        self.bandit_algo = bandit_algo
        self.bandit = bandit_algo.bandit
        self.trials = []
        self.results = []

    def run(self, N):
        raise NotImplementedError('override-me')

    def losses(self):
        return map(self.bandit_algo.bandit.loss, self.results, self.trials)

    def statuses(self):
        return map(self.bandit_algo.bandit.status, self.results, self.trials)

    def average_best_error(self):
        """Return the average best error of the experiment

        Average best error is defined as the average of bandit.true_loss,
        weighted by the probability that the corresponding bandit.loss is best.

        For bandits with loss measurement variance of 0, this function simply
        returns the true_loss corresponding to the result with the lowest loss.
        """
        bandit = self.bandit_algo.bandit

        def fmap(f):
            rval = np.asarray([f(r, s)
                    for (r, s) in zip(self.results, self.trials)
                    if bandit.status(r) == 'ok']).astype('float')
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


class SerialExperiment(Experiment):
    """
    """

    def run(self, N):
        algo = self.bandit_algo
        bandit = algo.bandit

        for n in xrange(N):
            trial = algo.suggest(self.trials, self.results, 1)[0]
            result = bandit.evaluate(trial, base.Ctrl())
            if not isinstance(result, (dict, base.SON)):
                raise TypeError('result should be dict-like', result)
            logger.debug('trial: %s' % str(trial))
            logger.debug('result: %s' % str(result))
            self.trials.append(trial)
            self.results.append(result)

