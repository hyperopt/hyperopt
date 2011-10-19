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
    'status' - a string describing what happened to this trial (see
                STATUS_STRINGS)
    'loss' - a scalar saying how bad this trial was.

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

STATUS_STRINGS = (
    'new',        # computations have not started
    'running',    # computations are in prog
    'suspended',  # computations have been suspended, job is not finished
    'ok',         # computations are finished, terminated normally
    'fail')       # computations are finished, terminated with error
                  #     - see result['status_fail'] for more info


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

    def dryrun_argd(self):
        """Return a point that could have been drawn from the template
        that is useful for small trial debugging.
        """
        raise NotImplementedError('override me')

    @classmethod
    def evaluate(cls, argd, ctrl):
        """Return a result document
        """
        raise NotImplementedError('override me')

    @classmethod
    def loss(cls, result, argd=None):
        """Extract the scalar-valued loss from a result document
        """
        try:
            return result['loss']
        except KeyError:
            return None

    @classmethod
    def loss_variance(cls, result, argd=None):
        """Return the variance in the estimate of the loss"""
        return 0

    @classmethod
    def true_loss(cls, result, argd=None):
        """Return a true loss, in the case that the `loss` is a surrogate"""
        return cls.loss(result, argd=argd)

    @classmethod
    def true_loss_variance(cls, result, argd=None):
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
        argd = self.dryrun_argd()
        self.evaluate(argd, ctrl)


class BanditAlgo(object):
    """
    Algorithm for solving Config-armed bandit (arms are from tree domain)

    X-armed bandit problems, and N-armed bandit problems are special cases.

    :type bandit: Bandit
    :param bandit: the bandit problem this algorithm should solve

    """
    seed = 123

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        # recursively change type in-place
        if 'template' in dct:
            ht_dist2.bless(self.template)

    def short_str(self):
        return self.__class__.__name__

    def set_bandit(self, bandit):
        self.bandit = bandit

    def suggest(self, X_list, Y_list, Y_status, N):
        raise NotImplementedError('override me')


class Experiment(object):
    """Object for conducting search experiments.
    """
    def __init__(self, bandit, bandit_algo):
        self.bandit = bandit
        self.bandit_algo = bandit_algo
        self.trials = []
        self.results = []

    def set_bandit(self, bandit=None):
        if bandit is None:
            self.bandit_algo.set_bandit(self.bandit)
        else:
            raise NotImplementedError('consider not allowing this')

    def run(self, N):
        raise NotImplementedError('override-me')

    def Ys(self):
        return map(self.bandit.loss, self.results)

    def Ys_status(self):
        return map(self.bandit.status, self.results)

    def average_best_error(self):
        """Return the average best error of the experiment

        Average best error is defined as the average of bandit.true_loss,
        weighted by the probability that the corresponding bandit.loss is best.

        For bandits with loss measurement variance of 0, this function simply
        returns the true_loss corresponding to the result with the lowest loss.
        """
        def doc(s):
            if 'TBA_id' in s:
                return s['doc']
            else:
                return s
        def fmap(f):
            rval = numpy.asarray([f(r, doc(s))
                    for (r, s) in zip(self.results, self.trials)
                    if r['status'] == 'ok' and self.bandit.loss(r, s) is not None
                    ]).astype('float')
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

