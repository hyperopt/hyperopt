"""
Hyper-parameter optimization algorithms (BanditAlgo subclasses) that depend on
Theano

"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

import numpy
import theano
from theano import tensor

import base
from idxs_vals_rnd import IdxsVals, IdxsValsList



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
    condition the suggestion, this class retrieves each document's '_config_id'
    key, and uses that key to look up information in self.db_idxs and
    self.db_vals.

    Consequently to storing this extra info in self.db_idxs and self.db_vals, it
    is essential that instances of this class be pickled in order for them to
    resume properly. It is not enough to pass a list of documents to the suggest
    method, for the algorithm to resume optimization.

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
    def __init__(self, bandit):
        base.BanditAlgo.__init__(self, bandit)
        self._next_id = 0
        seed = self.seed
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

    def next_id(self):
        rval = self._next_id
        self._next_id += 1
        return rval

    def recall(self, idlist):
        """Construct an IdxsValsList representation of the elements of idlist.

        The result will not be renumbered.
        """
        if idlist:
            idset = set(idlist)
            if len(idset) != len(idlist):
                raise NotImplementedError('dups in idlist')

            # for each variable in the bandit (each idxs, vals pair)
            # extract the database elements and put them into a new (idxs, vals)
            # pair that we can return.
            rval_idxs = []
            rval_vals = []
            for idxs, vals in zip(self.db_idxs, self.db_vals):
                assert len(idxs) == len(vals)
                ii_vv = [(ii, vv)
                        for (ii, vv) in zip(idxs, vals) if ii in idset]
                rval_idxs.append([iv[0] for iv in ii_vv])
                rval_vals.append([iv[1] for iv in ii_vv])
        else:
            rval_idxs = [[] for s in self.s_idxs]
            rval_vals = [[] for s in self.s_idxs]
        return IdxsValsList.fromlists(rval_idxs, rval_vals)

    def record(self, ivl):
        """Append idxs and vals to variable database, by numbering them
        self._next_id to N, and returning the list of these ids."""
        if len(ivl) != len(self.db_idxs):
            print('NUM', len(ivl), len(self.db_idxs))
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

    def idxs_vals_by_status(self, trials, results):
        """
        Build IdxsValsList representation of the trials, one for each status.
        Also group the losses and loss variances by status.

        returns dictionary with keys
            'positions': status -> idx into trials, results
            'x_IVLs': status -> IdxValsList of trials with that status
            'losses': status -> list of losses matching x_IVLs[status]
            'losses_variance': status -> list of matching x_IVLs[status]

        None of the returned dictionaries is aliased to the internal database
        of recorded results.

        """
        assert len(trials) == len(results)

        Y_status = map(self.bandit.status, results)

        for status in Y_status:
            if status not in STATUS_STRINGS:
                raise ValueError('un-recognized status', status)

        positions = {}
        ivs = {}
        ys = {}
        losses_variance = {}
        for status in STATUS_STRINGS:
            positions[status] = [i
                    for i, s in enumerate(Y_status) if s == status]
            ids = [trials[i]['_config_id'] for i in positions[status]]
            ivs[status] = self.recall(ids)
            ys[status] = IdxsVals(ids,
                    [self.bandit.loss(results[i], config=trials[i])
                        for i in positions[status]])
            losses_variance[status] = IdxsVals(ids,
                    [self.bandit.loss_variance(results[i], config=trials[i])
                        for i in positions[status]])
            logger.info('TheanoBanditAlgo.suggest: %04i jobs with status %s'
                    % (len(ids), status))

        # check that all ok jobs have a legitimate floating-point loss
        for y in ys['ok'].vals:
            try:
                float(y)
            except TypeError:
                raise TypeError('invalid loss for status "ok": %s'
                        % y)
            if float(y) != float(y):
                raise ValueError('invalid loss for status "ok":  %s'
                        % y)

        return dict(
                positions=positions,
                x_IVLs=ivs,
                losses=ys,
                losses_variance=losses_variance)

    def suggest_ivl(self, ivl):
        """
        ivl: IdxsValsList representation of suggestions

        This method `record`s the suggestion and rebuilds the document
        representation.
        """
        assert isinstance(ivl, IdxsValsList)

        ids = self.record(ivl)
        N = len(ids)

        # now call idxs_vals_to_dict_list to rebuild a nested document
        # suitable for returning
        all_r_idxs = [None] * len(self.all_s_idxs)
        all_r_vals = [None] * len(self.all_s_vals)
        for i, j in enumerate(self.all_s_locs):
            all_r_idxs[j] = numpy.asarray(ivl[i].idxs)
            all_r_vals[j] = numpy.asarray(ivl[i].vals)
        rval = self.bandit.template.idxs_vals_to_dict_list(
                list(all_r_idxs),
                list(all_r_vals))
        assert len(rval) == N

        # mark each trial with a _config_id that connects it to self.db_idxs
        for rid, r in zip(ids, rval):
            assert rid == int(rid)       # numpy int64 is not BSON
            r['_config_id'] = int(rid)
        return rval


class TheanoRandom(TheanoBanditAlgo):
    """Random search director, but testing the machinery that translates
    doctree configurations into sparse matrix configurations.
    """
    def __init__(self, bandit):
        TheanoBanditAlgo.__init__(self, bandit)
        self._sampler = theano.function(
                [self.s_N],
                self.s_idxs + self.s_vals)

    def suggest(self, trials, results, N):
        # normally a TheanoBanditAlto.suggest would start with this call:
        ##  ivls = self.idxs_vals_by_status(trials, results)
        rvals = self._sampler(N)

        # A TheanoBanditAlgo.suggest implementation should usually
        # return suggest_ivl(...).
        return self.suggest_ivl(
                IdxsValsList.fromlists(
                    rvals[:len(rvals)/2],
                    rvals[len(rvals)/2:]))



from hyperopt.theano_bandit_algos import TheanoBanditAlgo
from hyperopt.idxs_vals_rnd import IdxsValsList

from hyperopt import bandits

def ivl_fl(ilist, vlist):
    return IdxsValsList.fromlists(ilist, vlist)

def test_recall_and_record_1d():
    bandit = bandits.TwoArms()
    algo = TheanoBanditAlgo(bandit)

    assert algo.recall([]) == ivl_fl([[]], [[]])

    algo.record(ivl_fl([[0]], [[0]]))

    assert algo.recall([]) == ivl_fl([[]], [[]])
    assert algo.recall([0]) == ivl_fl([[0]], [[0]])

    algo.record(ivl_fl([[0, 1, 2]], [[0, 1, 0]]))

    assert algo.recall([]) == ivl_fl([[]], [[]])
    assert algo.recall([0]) == ivl_fl([[0]], [[0]])
    assert algo.recall([0, 1]) == ivl_fl([[0, 1]], [[0, 0]])
    assert algo.recall([0, 2]) == ivl_fl([[0, 2]], [[0, 1]])
    assert algo.recall([3]) == ivl_fl([[3]], [[0]])


def test_recall_record_2d():
    bandit = bandits.GaussWave2()
    algo = TheanoBanditAlgo(bandit)
    assert len(algo.all_s_idxs) == 6
    assert len(algo.s_idxs) == 3
    assert len(algo.all_s_locs) == 3
    assert algo.all_s_locs == [1, 2, 5]

    if 0:
        nodes = bandit.template.flatten()
        for i, n in enumerate(nodes):
            print i, n
        # 0 - base dict
        # 1 - x (uniform)
        # 2 - hf (one_of)
        # 3 - {'kind': 'raw'}
        # 4 - {'kind': 'negcos'}
        # 5 - amp (uniform)

    assert algo.recall([]) == ivl_fl([[], [], []], [[], [], []])

    # {x=-5, hf=raw}
    # {x=-10, hf={negcos, amp=.5}
    # {x=-15, hf={negcos, amp=.25}}
    assert algo.record(ivl_fl(
            # (x)            (hf)       (amp)  )
            [[0,   1,    2], [0, 1, 2], [1,   2  ]],
            [[-5, -10, -15], [0, 1, 1], [.5, .25 ]]))

    assert algo.recall([]) == ivl_fl([[], [], []], [[], [], []])
    assert algo.recall([0]) == ivl_fl(
            [[0], [0], []],
            [[-5], [0], []])
    assert algo.recall([1]) == ivl_fl(
            [[1], [1], [1]],
            [[-10], [1], [.5]])
    assert algo.recall([2]) == ivl_fl(
            [[2], [2], [2]],
            [[-15], [1], [.25]])

    # {x=-5, hf=raw}
    # {x=-10, hf={negcos, amp=.5}
    # {x=-15, hf={negcos, amp=.25}}
    assert algo.record(ivl_fl(
            # (x)            (hf)       (amp)  )
            [[0,  1,  2], [0, 1, 2], [0,   2  ]],
            [[5, 10, 15], [1, 0, 1], [.5, .25 ]]))

    assert algo.recall([2,3,4]) == ivl_fl(
            [[2,   3, 4 ], [2, 3, 4], [2, 3]],
            [[-15, 5, 10], [1, 1, 0], [.25, .5]])

# XXX: test suggest()
