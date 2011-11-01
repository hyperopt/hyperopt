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

import cPickle
import logging
import sys

import numpy
try:
    from bson import SON
except ImportError:
    SON = dict

import ht_dist2
import utils
import idxs_vals_rnd

logger = logging.getLogger(__name__)


class Ctrl(object):
    """Control object for interruptible, checkpoint-able evaluation
    """
    info = logger.info
    warn = logger.warn
    error = logger.error
    debug = logger.debug

    def checkpoint(self, r=None):
        pass


class Bandit(object):
    """Specification of bandit problem.

    template - htdist2 specification of search domain

    evaluate - interruptible/checkpt calling convention for evaluation routine

    """
    def __init__(self, template):
        self.template = template

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        # recursively change type in-place
        ht_dist2.bless(self.template)


    def short_str(self):
        return self.__class__.__name__

    def dryrun_config(self):
        """Return a point that could have been drawn from the template
        that is useful for small trial debugging.
        """
        raise NotImplementedError('override me')

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
    def status(cls, result):
        """Extract the job status from a result document
        """
        return result['status']

    def new_result(self):
        """Return a JSON-encodable object
        to serve as the 'result' for new jobs.
        """
        return {'status': 'new'}

    @classmethod
    def main_dryrun(cls):
        self = cls()
        ctrl = Ctrl()
        config = self.dryrun_config()
        self.evaluate(config, ctrl)


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

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        # recursively change type in-place
        if 'template' in dct:
            ht_dist2.bless(self.template)

    def short_str(self):
        return self.__class__.__name__

    def suggest(self, trials, results, N):
        raise NotImplementedError('override me')


class Experiment(object):
    """Object for conducting search experiments.
    """
    def __init__(self, bandit_algo):
        self.bandit_algo = bandit_algo
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
        def fmap(f):
            rval = numpy.asarray([f(r, s)
                    for (r, s) in zip(self.results, self.trials)
                    if self.bandit.status(r) == 'ok']).astype('float')
            if not numpy.all(numpy.isfinite(rval)):
                raise ValueError()
            return rval
        loss = fmap(self.bandit.loss)
        loss_v = fmap(self.bandit.loss_variance)
        if self.bandit.true_loss is not Bandit.true_loss:
            true_loss = fmap(self.bandit.true_loss)
            loss3 = zip(loss, loss_v, true_loss)
        else:
            loss3 = zip(loss, loss_v, loss)
        loss3.sort()
        loss3 = numpy.asarray(loss3)
        if numpy.all(loss3[:, 1] == 0):
            best_idx = numpy.argmin(loss3[:, 0])
            return loss3[best_idx, 2]
        else:
            cutoff = 0
            sigma = numpy.sqrt(loss3[0][1])
            while (cutoff < len(loss3)
                    and loss3[cutoff][0] < loss3[0][0] + 3 * sigma):
                cutoff += 1
            pmin = utils.pmin_sampled(loss3[:cutoff, 0], loss3[:cutoff, 1])
            #print pmin
            #print loss3[:cutoff, 0]
            #print loss3[:cutoff, 1]
            #print loss3[:cutoff, 2]
            avg_true_loss = (pmin * loss3[:cutoff, 2]).sum()
            return avg_true_loss

