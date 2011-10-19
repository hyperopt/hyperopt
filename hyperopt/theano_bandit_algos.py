import sys
import logging
logger = logging.getLogger(__name__)

import numpy
import theano
from theano import tensor

import base
from idxs_vals_rnd import IdxsValsList


class TheanoBanditAlgo(base.BanditAlgo):
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
        return IdxsValsList.fromlists(rval_idxs, rval_vals)

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
        if new_ids:
            assert numpy.max(new_ids) >= self._next_id
            self._next_id = numpy.max(new_ids) + 1
        new_ids = list(sorted(set(new_ids)))
        return new_ids

    def suggest(self, X_list, Y_list, Y_status, N):
        assert len(X_list) == len(Y_list) == len(Y_status)

        for status in Y_status:
            if status not in base.STATUS_STRINGS:
                raise ValueError('un-recognized status', status)

        positions = {}
        ivs = {}
        ys = {}
        for status in base.STATUS_STRINGS:
            positions[status] = [i
                    for i, s in enumerate(Y_status) if s == status]
            ivs[status] = self.recall([X_list[i]['TBA_id']
                for i in positions[status]])
            ys[status] = [Y_list[i]
                for i in positions[status]]
            logger.info('TheanoBanditAlgo.suggest: %04i jobs with status %s'
                    % (len(ys[status]), status))
            if 'ok' == status:
                for y in ys[status]:
                    try:
                        float(y)
                    except TypeError:
                        raise TypeError('invalid loss for status "ok": %s'
                                % y)
                    if float(y) != float(y):
                        raise ValueError('invalid loss for status "ok":  %s'
                                % y)

        assert not numpy.any(
                numpy.isnan(
                    numpy.array(ys['ok'], dtype='float')))

        # this is an assert because we validated Y_status above
        assert sum(len(l) for l in positions.values()) == len(Y_status)

        ivl = self.theano_suggest(ivs, ys, N)
        assert isinstance(ivl, IdxsValsList)

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
        rval = [base.SON([('TBA_id', int(rid)), ('doc', r)])
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


class TheanoRandom(TheanoBanditAlgo):
    """Random search director, but testing the machinery that translates
    doctree configurations into sparse matrix configurations.
    """
    def set_bandit(self, bandit):
        TheanoBanditAlgo.set_bandit(self, bandit)
        self._sampler = theano.function(
                [self.s_N],
                self.s_idxs + self.s_vals)

    def theano_suggest(self, X_IVLs, Ys, N):
        """Ignore X and Y, draw from prior"""
        rvals = self._sampler(N)
        return IdxsValsList.fromlists(
                rvals[:len(rvals)/2],
                rvals[len(rvals)/2:])
