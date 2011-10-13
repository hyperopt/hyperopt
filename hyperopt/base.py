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
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "James Bergstra <pylearn-dev@googlegroups.com>"

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
    def loss(cls, result):
        """Extract the scalar-valued loss from a result document
        """
        try:
            return result['loss']
        except KeyError:
            return None

    # TODO: loss variance

    # OPTIONAL BUT OFTEN MEANINGFUL
    # TODO: test set error
    # TODO: test set error variance

    @classmethod
    def status(cls, result):
        """Extract the job status from a result document
        """
        try:
            return result['status']
        except KeyError:
            return None

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


class TheanoBanditAlgo(BanditAlgo):
    """
    Base class for a BanditAlgorithm using the idxs,vals format for storing
    configurations rather than the list-of-document format.

    The idxs, vals format plays better with Theano implementations of GP models
    and PBIL and stuff.

    Instances deriving from this remember more information about suggested
    points than they return via self.suggest().
    That information is stored in the attributes self.db_idxs and self.db_vals.
    When the suggest() method receives a list documents that should be used to
    condition the suggestion, this class retrieves each document's 'TBA_id' key,
    and uses that key to look up information in self.db_idxs and self.db_vals.

    Consequently to storing this extra info in self.db_idxs and self.db_vals, it
    is essential that instances of this class be pickled in order for them to
    resume properly. It is not enough to pass a list of documents (X_list) to
    the suggest method, for the algorithm to resume optimization.

    :type s_idxs:
        list of symbolic integer vectors

    :param s_idxs:
        the i'th int vector contains the positions in the sample (n < N) where
        the i'th configuration variable is defined

    :type s_vals:
        list of symbolic ndarrays

    :param s_vals:
        the i'th ndarray contains the values for the i'th variables at the
        sample positions indicated in s_idxs.

    :type s_N:
        symbolic int

    :param s_N:
        the number of samples drawn from the prior

    :type db_idxs:
        list of integer lists

    :param db_idxs:
        positions where the corresponding element of s_idxs is
        defined.

    :type db_vals:
        list of ndarrays or lists

    :param db_vals:
        values for corresponding elements of db_idxs

    """
    def __init__(self):
        self._next_id = 0

    def next_id(self):
        rval = self._next_id
        self._next_id += 1
        return rval

    def set_bandit(self, bandit):
        seed = self.seed
        self.bandit = bandit
        all_s_idxs, all_s_vals, s_N = bandit.template.theano_sampler(seed)
        all_s_locs = [i for i, s in enumerate(all_s_idxs) if s is not None]

        self.all_s_idxs = all_s_idxs
        self.all_s_vals = all_s_vals
        self.all_s_locs = all_s_locs
        self.s_N = s_N

        self.s_idxs = list(numpy.asarray(all_s_idxs)[all_s_locs])
        self.s_vals = list(numpy.asarray(all_s_vals)[all_s_locs])
        self.db_idxs = [[] for s in self.s_idxs]
        self.db_vals = [[] for s in self.s_idxs]

    def recall(self, idlist):
        """Construct an IdxsValsList representation of the elements of idlist.

        The result will be renumberd 0,1,...len(idlist).

        Thus element 0 of the returned IdxValsList will correspond to the
        database element whose id matches the 0th element of idlist.
        """
        if idlist:
            # iddict maps idx in database -> idx in rval
            iddict = dict([(orig, new) for (new, orig) in enumerate(idlist)])
            if len(iddict) != len(idlist):
                raise NotImplementedError('dups in idlist')

            # for each variable in the bandit (each idxs, vals pair)
            # extract the database elements and put them into a new (idxs, vals)
            # pair that we can return.
            rval_idxs = []
            rval_vals = []
            for idxs, vals in zip(self.db_idxs, self.db_vals):
                assert len(idxs) == len(vals)
                ii_vv = [(iddict[ii], vv)
                        for (ii, vv) in zip(idxs, vals) if ii in iddict]
                if ii_vv:
                    idxs, vals = zip(*ii_vv)
                else:
                    idxs, vals = [], []
                rval_idxs.append(list(idxs))
                rval_vals.append(list(vals))
        else:
            rval_idxs = [[] for s in self.s_idxs]
            rval_vals = [[] for s in self.s_idxs]
        return idxs_vals_rnd.IdxsValsList.fromlists(rval_idxs, rval_vals)

    def record(self, ivl):
        """Append idxs and vals to variable database, by numbering them
        self._next_id to N, and returning the list of these ids."""
        if len(ivl) != len(self.db_idxs):
            raise ValueError('number of variables does not match db_idxs - '
                    'are you sure you are recording to the right database?')
        new_ids = []
        N = 0
        # the indexes in ivl cover integers from 0 to some number N-1
        # (inclusive)
        # these will be mapped onto the database ids
        # self._next_id to self._next_id + N
        #
        # This function does not guarantee that all the db ids in these
        # ranges are occupied.
        for i, iv in enumerate(ivl):
            for ii in iv.idxs:
                if ii < 0:
                    raise ValueError('negative index encountered')
                N = max(N, ii+1)
                new_ids.append(ii + self._next_id)
                self.db_idxs[i].append(ii + self._next_id)
            self.db_vals[i].extend(iv.vals)
        self._next_id = numpy.max(new_ids) + 1
        new_ids = list(sorted(set(new_ids)))
        return new_ids

    def suggest(self, X_list, Y_list, Y_status, N):
        assert len(X_list) == len(Y_list) == len(Y_status)

        positions = {}
        ivs = {}
        ys = {}
        for status in STATUS_STRINGS:
            positions[status] = [i
                    for i, s in enumerate(Y_status) if s == status]
            ivs[status] = self.recall([X_list[i]['TBA_id']
                for i in positions[status]])
            ys[status] = [Y_list[i]
                for i in positions[status]]

        assert sum(len(l) for l in positions.values()) == len(Y_status)

        ivl = self.theano_suggest(ivs, ys, N)
        assert isinstance(ivl, idxs_vals_rnd.IdxsValsList)

        ids = self.record(ivl)
        assert len(ids) == N

        # now call idxs_vals_to_dict_list to rebuild a nested document
        # suitable for returning
        all_r_idxs = [None] * len(self.all_s_idxs)
        all_r_vals = [None] * len(self.all_s_vals)
        for i, j in enumerate(self.all_s_locs):
            all_r_idxs[j] = ivl[i].idxs
            all_r_vals[j] = ivl[i].vals
        rval = self.bandit.template.idxs_vals_to_dict_list(
                list(all_r_idxs),
                list(all_r_vals))
        assert len(rval) == N

        # HACK!
        # tuck each suggested document into a dictionary with a TBA_id field
        rval = [SON([('TBA_id', int(rid)), ('doc', r)])
            for rid, r in zip(ids, rval)]
        return rval

    def theano_suggest(self, X_IVLs, Ys, N):
        """Return new points to try.

        :type X_IVLs:
            dictionary mapping status string -> IdxsValsList

        :param X_IVLs:
            experiment configurations at each status level

        :type Ys:
            dictionary mapping status string -> list of losses

        :param Ys:
            losses for the corresponding configurations in X_IVLs

        :param N:
            number of trials to suggest

        :rtype:
            IdxsValsList

        :returns:
            new configurations to try

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
