"""
Annealing algorithm for hyperopt

Annealing is a simple but effective variant on random search that
takes some advantage of a smooth response surface.

The simple (but not overly simple) code of simulated annealing makes this file
a good starting point for implementing new search algorithms.

"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging

import numpy as np
from pyll.stochastic import (
    # -- integer
    categorical,
    # randint, -- unneeded
    # -- normal
    normal,
    lognormal,
    qnormal,
    qlognormal,
    # -- uniform
    uniform,
    loguniform,
    quniform,
    qloguniform,
    )
from .base import miscs_to_idxs_vals
from .algobase import (
    SuggestAlgo,
    ExprEvaluator,
    )

logger = logging.getLogger(__name__)


class AnnealingAlgo(SuggestAlgo):
    """
    This simple annealing algorithm begins by sampling from the prior,
    but tends over time to sample from points closer and closer to the best
    ones observed.

    In addition to the value of this algorithm as a baseline optimization
    strategy, it is a simple starting point for implementing new algorithms.

    # The Annealing Algorithm

    The annealing algorithm is to choose one of the previous trial points
    as a starting point, and then to sample each hyperparameter from a similar
    distribution to the one specified in the prior, but whose density is more
    concentrated around the trial point we selected.

    This algorithm is a simple variation on random search that leverages
    smoothness in the response surface.  The annealing rate is not adaptive.

    ## Choosing a Best Trial

    The algorithm formalizes the notion of "one of the best trials" by
    sampling a position from a geometric distribution whose mean is the
    `avg_best_idx` parameter.  The "best trial" is the trial thus selected
    from the set of all trials (`self.trials`).

    It may happen that in the process of ancestral sampling, we may find that
    the best trial at some ancestral point did not use the hyperparameter we
    need to draw.  In such a case, this algorithm will draw a new "runner up"
    best trial, and use that one as if it had been chosen as the best trial.

    The set of best trials, and runner-up best trials obtained during the
    process of choosing all hyperparameters is kept sorted by the validation
    loss, and at each point where the best trial does not define a
    required hyperparameter value, we actually go through all the list of
    runners-up too, before giving up and adding a new runner-up trial.


    ## Concentrating Prior Distributions

    To sample a hyperparameter X within a search space, we look at
    what kind of hyperparameter it is (what kind of distribution it's from)
    and the previous successful values of that hyperparameter, and make
    a new proposal for that hyperparameter independently of other
    hyperparameters (except technically any choice nodes that led us to use
    this current hyperparameter in the first place).

    For example, if X is a uniform-distributed hyperparameters drawn from
    `U(l, h)`, we look at the value `x` of the hyperparameter in the selected
    trial, and draw from a new uniform density `U(x - w/2, x + w/2)`, where w
    is related to the initial range, and the number of observations we have for
    X so far. If W is the initial range, and T is the number of observations
    we have, then w = W / (1 + T * shrink_coef).  If the resulting range would
    extend either below l or above h, we shift it to fit into the original
    bounds.

    """

    def __init__(self, domain, trials, seed,
                 avg_best_idx=2.0,
                 shrink_coef=0.1):
        SuggestAlgo.__init__(self, domain, trials, seed=seed)
        self.avg_best_idx = avg_best_idx
        self.shrink_coef = shrink_coef
        doc_by_tid = {}
        for doc in trials.trials:
            # get either this docs own tid or the one that it's from
            tid = doc['tid']
            loss = domain.loss(doc['result'], doc['spec'])
            if loss is None:
                # -- associate infinite loss to new/running/failed jobs
                loss = float('inf')
            else:
                loss = float(loss)
            doc_by_tid[tid] = (doc, loss)
        self.tid_docs_losses = sorted(doc_by_tid.items())
        self.tids = np.asarray([t for (t, (d, l)) in self.tid_docs_losses])
        self.losses = np.asarray([l for (t, (d, l)) in self.tid_docs_losses])
        self.tid_losses_dct = dict(zip(self.tids, self.losses))
        self.node_tids, self.node_vals = miscs_to_idxs_vals(
            [d['misc'] for (tid, (d, l)) in self.tid_docs_losses],
            keys=domain.params.keys())
        self.best_tids = []

    def shrinking(self, label):
        T = len(self.node_vals[label])
        return 1.0 / (1.0 + T * self.shrink_coef)

    def choose_ltv(self, label):
        """Returns (loss, tid, val) of best/runner-up trial
        """
        tids = self.node_tids[label]
        vals = self.node_vals[label]
        losses = [self.tid_losses_dct[tid] for tid in tids]

        # -- try to return the value corresponding to one of the
        #    trials that was previously chosen
        tid_set = set(tids)
        for tid in self.best_tids:
            if tid in tid_set:
                idx = tids.index(tid)
                rval = losses[idx], tid, vals[idx]
                break
        else:
            # -- choose a new best idx
            ltvs = sorted(zip(losses, tids, vals))
            best_idx = int(self.rng.geometric(1.0 / self.avg_best_idx)) - 1
            best_idx = min(best_idx, len(ltvs) - 1)
            assert best_idx >= 0
            best_loss, best_tid, best_val = ltvs[best_idx]
            self.best_tids.append(best_tid)
            rval = best_loss, best_tid, best_val
        return rval

    def on_node_hyperparameter(self, memo, node, label):
        """
        Return a new value for one hyperparameter.

        Parameters:
        -----------

        memo - a partially-filled dictionary of node -> list-of-values
               for the nodes in a vectorized representation of the
               original search space.

        node - an Apply instance in the vectorized search space,
               which corresponds to a hyperparameter

        label - a string, the name of the hyperparameter


        Returns: a list with one value in it: the suggested value for this
        hyperparameter


        Notes
        -----

        This function works by delegating to self.hp_HPTYPE functions to
        handle each of the kinds of hyperparameters in hyperopt.pyll_utils.

        Other search algorithms can implement this function without
        delegating based on the hyperparameter type, but it's a pattern
        I've used a few times so I show it here.

        """
        vals = self.node_vals[label]
        if len(vals) == 0:
            return ExprEvaluator.on_node(self, memo, node)
        else:
            loss, tid, val = self.choose_ltv(label)
            try:
                handler = getattr(self, 'hp_%s' % node.name)
            except AttributeError:
                raise NotImplementedError('Annealing', node.name)
            return handler(memo, node, label, tid, val)

    def hp_uniform(self, memo, node, label, tid, val,
                   log_scale=False,
                   pass_q=False,
                   uniform_like=uniform):
        """
        Return a new value for a uniform hyperparameter.

        Parameters:
        -----------

        memo - (see on_node_hyperparameter)

        node - (see on_node_hyperparameter)

        label - (see on_node_hyperparameter)

        tid - trial-identifier of the model trial on which to base a new sample

        val - the value of this hyperparameter on the model trial

        Returns: a list with one value in it: the suggested value for this
        hyperparameter
        """
        if log_scale:
            val = np.log(val)
        high = memo[node.arg['high']]
        low = memo[node.arg['low']]
        assert low <= val <= high
        width = (high - low) * self.shrinking(label)
        new_high = min(high, val + width / 2)
        if new_high == high:
            new_low = new_high - width
        else:
            new_low = max(low, val - width / 2)
            if new_low == low:
                new_high = new_low + width
        assert low <= new_low <= new_high <= high
        if pass_q:
            return uniform_like(
                low=new_low,
                high=new_high,
                rng=self.rng,
                q=memo[node.arg['q']],
                size=memo[node.arg['size']])
        else:
            return uniform_like(
                low=new_low,
                high=new_high,
                rng=self.rng,
                size=memo[node.arg['size']])

    def hp_quniform(self, *args, **kwargs):
        return self.hp_uniform(
            pass_q=True,
            uniform_like=quniform,
            *args,
            **kwargs)

    def hp_loguniform(self, *args, **kwargs):
        return self.hp_uniform(
            log_scale=True,
            pass_q=False,
            uniform_like=loguniform,
            *args,
            **kwargs)

    def hp_qloguniform(self, *args, **kwargs):
        return self.hp_uniform(
            log_scale=True,
            pass_q=True,
            uniform_like=qloguniform,
            *args,
            **kwargs)

    def hp_randint(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        upper = memo[node.arg['upper']]
        counts = np.zeros(upper)
        counts[val] += 1
        prior = self.shrinking(label)
        p = (1 - prior) * counts + prior * (1.0 / upper)
        rval = categorical(p=p, upper=upper, rng=self.rng,
                           size=memo[node.arg['size']])
        return rval

    def hp_categorical(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        p = p_orig = np.asarray(memo[node.arg['p']])
        if p.ndim == 2:
            assert len(p) == 1
            p = p[0]
        counts = np.zeros_like(p)
        counts[val] += 1
        prior = self.shrinking(label)
        new_p = (1 - prior) * counts + prior * p
        if p_orig.ndim == 2:
            rval = categorical(p=[new_p], rng=self.rng,
                               size=memo[node.arg['size']])
        else:
            rval = categorical(p=new_p, rng=self.rng,
                               size=memo[node.arg['size']])
        return rval

    def hp_normal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return normal(
            mu=val,
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            rng=self.rng,
            size=memo[node.arg['size']])

    def hp_lognormal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return lognormal(
            mu=np.log(val),
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            rng=self.rng,
            size=memo[node.arg['size']])

    def hp_qlognormal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return qlognormal(
            # -- prevent log(0) without messing up algo
            mu=np.log(1e-16 + val),
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            q=memo[node.arg['q']],
            rng=self.rng,
            size=memo[node.arg['size']])

    def hp_qnormal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return qnormal(
            mu=val,
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            q=memo[node.arg['q']],
            rng=self.rng,
            size=memo[node.arg['size']])


def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    new_id, = new_ids
    return AnnealingAlgo(domain, trials, seed, *args, **kwargs)(new_id)

# -- flake-8 abhors blank line EOF
