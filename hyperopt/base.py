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

The modules communicate with trials in nested dictionary form.
TheanoBanditAlgo translates nested dictionary form into idxs, vals form.

"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "James Bergstra <pylearn-dev@googlegroups.com>"

import cPickle
import logging
import sys

import ht_dist2
import utils

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
    def loss(cls, result):
        """Extract the scalar-valued loss from a result document
        """
        return result['loss']

    # TODO: loss variance
    # TODO: test set error
    # TODO: test set error variance

    @classmethod
    def status(cls, result):
        """Extract the job status from a result document
        """
        return result['status']


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


class TheanoBanditAlgo(object):
    """
    Algorithm for solving Config-armed bandit (arms are from tree domain)

    X-armed bandit problems, and N-armed bandit problems are special cases.

    :type s_prior_idxs:
        list of symbolic integer vectors

    :param s_prior_idxs:
        the i'th int vector contains the positions in the sample (n < N) where
        the i'th configuration variable is defined

    :type s_prior_vals:
        list of symbolic ndarrays

    :param s_prior_vals:
        the i'th ndarray contains the values for the i'th variables at the
        sample positions indicated in s_prior_idxs.

    :type s_N:
        symbolic int

    :param s_N:
        the number of samples drawn from the prior

    """

    def set_bandit(self, bandit):
        self.bandit = bandit
        s_idxs, s_vals, s_N = ht_dist2.theano_sampler(bandit.template, self.seed)
        self.s_idxs, self.s_vals, self.s_N = s_idxs, s_vals, s_N

    def suggest(self, X_list, Y_list, Y_status, N):
        idxs, vals = ht_dist2.dict_list_to_idxs_vals(X_list)
        return theano_suggest(idxs, vals, Y_list, Y_status, N)

    def theano_suggest(self, X_idxs, X_vals, Y, Y_status, N):
        """Return new points to try.

        :param X_idxs:
            list of int vectors that could have come from s_prior_idxs

        :param X_vals:
            list of ndarrays that could have come from s_prior_vals

        :param Y:
            vector of results for X

        :param Y_status:
            vector of status of results (elements: 'ok', 'fail', 'running')

        :param N:
            number of trials to suggest

        :rtype:
            list of int vectors, list of ndarrays

        :returns:
            suggested new X points in same idxs, vals encoding.

        """
        raise NotImplementedError('override me')



class Experiment(object):
    """Object for conducting search experiments.
    """
    def __init__(self, bandit, bandit_algo):
        self.bandit = bandit
        self.bandit_algo = bandit_algo
        self.trials = []
        self.results = []

    def run(self, N):
        raise NotImplementedError('override-me')

    def Ys(self):
        return map(self.bandit.loss, self.results)

    def Ys_status(self):
        return map(self.bandit.status, self.results)

    @classmethod
    def main_search(cls, argv):
        save_loc = argv[0]
        assert save_loc.endswith('.pkl')
        try:
            handle = open(save_loc, 'rb')
            self = cPickle.load(handle)
            handle.close()
        except IOError:
            bandit = utils.json_call(argv[1])
            bandit_algo = utils.json_call(argv[2])
            bandit_algo.set_bandit(bandit)
            self = cls(bandit, bandit_algo)
        try:
            self.run(100)
        finally:
            cPickle.dump(self, open(save_loc, 'wb'))

