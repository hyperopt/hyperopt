"""
Graphical model (GM)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging
logger = logging.getLogger(__name__)

import numpy
import theano
from theano import tensor

import montetheano
from montetheano.for_theano import argsort
from montetheano.for_theano import where
from montetheano.for_theano import ancestors
from montetheano.for_theano import as_variable
from montetheano.for_theano import find
from montetheano.for_theano import clone_get_equiv
from montetheano.for_theano import clone_keep_replacements

import idxs_vals_rnd
from idxs_vals_rnd import IdxsVals
from idxs_vals_rnd import IdxsValsList

from theano_bandit_algos import TheanoBanditAlgo

def as_variables(*args):
    return [tensor.as_tensor_variable(a) for a in args]

class IdxsVals(object):
    """Sparse compressed vector-like representation (idxs, vals).

    N.B. This class is sometimes used to store symbolic variables, and
    sometimes used to store numeric variables.  Not all operations work in
    both cases, but many do.

    """
    def __init__(self, i, v):
        self.idxs = i  # symbolic integer vector
        self.vals = v  # symbolic ndarray with same length as self.idxs

    def __eq__(self, other):
        return self.idxs == other.idxs and self.vals == other.vals

    def __str__(self):
        return 'IdxsVals(%s, %s)' % (self.idxs, self.vals)

    def __repr__(self):
        return str(self)

    def symbolic_take(self, elements):
        """Symbolic advanced sparse vector indexing by int-list `elements`
        """
        pos = find(self.idxs, elements)
        return IdxsVals(self.idxs[pos], self.vals[pos])

    def numeric_take(self, elements):
        """Numeric advanced sparse vector indexing by int-list `elements`
        """
        d = dict(zip(self.idxs, self.vals))
        return self.__class__(
                [ii for ii in elements if ii in d],
                [d[ii] for ii in elements if ii in d])

    def copy(self):
        return self.__class__(copy.copy(self.idxs), copy.copy(self.vals))

    def reindex(self, idmap=None):
        """Replace elements of self.idxs according to `idmap`
        """
        if idmap is None:
            idmap = {}
            for idx in self.idxs:
                idmap.setdefault(idx, len(idmap))
        self.idxs = [idmap[i] for i in self.idxs]
        return idmap

    def stack(self, other):
        self.idxs.extend(other.idxs)
        self.vals.extend(other.vals)

    def as_numpy(self, vdtype=None):
        idxs = numpy.asarray(self.idxs)
        if vdtype is None:
            vals = numpy.asarray(self.vals)
        else:
            vals = numpy.asarray(self.vals, dtype=vdtype)
        return self.__class__(idxs, vals)

    def as_list(self):
        return self.__class__(list(self.idxs), list(self.vals))

    def idxset(self):
        return set(self.idxs)


class IdxsValsList(list):
    """
    List of IdxsVals instances

    N.B. This class is sometimes used to store symbolic variables, and
    sometimes used to store numeric variables.  Not all operations work in
    both cases, but many do.

    """
    def __eq__(self, other):
        return (len(self) == len(other)
                and all(s == o for (s, o) in zip(self, other)))

    def idxslist(self):
        return [e.idxs for e in self]

    def valslist(self):
        return [e.vals for e in self]

    def symbolic_take(self, subelements):
        """Return a new IdxsValsList of the same length as self, whose elements
        are restricted to contain only the given `subelements`.
        """
        # make a variable outside the loop to help theano's merge-optimizer
        subel = theano.tensor.as_tensor_variable(subelements)
        return self.__class__([e.symbolic_take(subel) for e in self])

    def numeric_take(self, subelements):
        """Numeric take, returns IdxsValsList in which elements not in
        `subelements` have been discarded
        """
        if len(set(subelements)) != len(subelements):
            raise ValueError('duplicate in subelements are ambiguous')
        return self.__class__([e.numeric_take(subelements) for e in self])

    @classmethod
    def new_like(cls, ivl):
        rval = cls([IdxsVals(e.idxs.type(), e.vals.type()) for e in ivl])
        return rval

    @classmethod
    def fromlists(cls, idxslist, valslist):
        if len(idxslist) != len(valslist):
            raise ValueError('length mismatch')
        rval = cls([IdxsVals(i, v) for i, v in zip(idxslist, valslist)])
        return rval

    def new_like_self(self):
        return self.new_like(self)

    def flatten(self):
        """Return self[0].idxs, self[0].vals, self[1].idxs, self[1].vals, ...

        This is useful for passing arguments to a Theano function.
        """
        rval = []
        for e in self:
            rval.extend([e.idxs, e.vals])
        return rval

    @classmethod
    def fromflattened(cls, args):
        """Construct an IdxsValsList from the idxs0, vals0, idxs1, vals1, ...

        This constructor re-constructs from the flattened representation.
        """
        if len(args) % 2:
            raise ValueError('expected args of form'
                    ' idxs0, vals0, idxs1, vals1, ...')
        return cls.fromlists(args[::2], args[1::2])

    def copy(self):
        return self.__class__([iv.copy() for iv in self])

    def stack(self, other):
        """
        Insert & append all the elements of other into self.

        The indexes in other are not modified, so this could introduce
        duplicate index values.
        """
        if len(self) != len(other):
            raise ValueError('other is not compatible with self')
        # append the other's elements to self
        for self_iv, other_iv in zip(self, other):
            self_iv.stack(other_iv)

    def nnz(self):
        return len(self.idxset())

    def idxset(self):
        """Return the set of active index positions"""
        rval = set()
        for idxs in self.idxslist():
            rval.update(idxs)
        return rval

    def reindex(self, idmap=None):
        if idmap is None:
            idmap = dict([(idx, i) for i, idx in
                enumerate(sorted(self.idxset()))])
        for iv in self:
            iv.reindex(idmap)
        return idmap

    def as_numpy(self):
        return self.__class__([iv.as_numpy() for iv in self])

    def as_numpy_floatX(self):
        rval = self.as_numpy()
        for iv in rval:
            if iv.vals.dtype == 'float64':
                # does nothing if floatX is float64
                iv.vals = iv.vals.astype(theano.config.floatX)
        return rval

    def as_list(self):
        return self.__class__([iv.as_list() for iv in self])


class TreeEstimator(object):
    def posterior(self, priors, observations, s_rng):
        """
        priors - an IdxsValsList of random variables
        observations - an IdxsValsList of corresponding samples

        returns - an IdxsValsList of posterior random variables
        """
        raise NotImplementedError('override-me')

    def log_likelihood(self, posterior, observations, N=None):
        """Return an IdxsVals containing the log density of observations under
        posterior

        posterior - IdxsValsList representing posterior densities
            (a return value from self.posterior())
        sample - an IdxsValsList representing draws from the posterior.
            (can be same as `posterior`)
        N - the number of observations if any idxs in observations are symbolic

        Returns IdxsVals where the vals are log densities of the examples named
        by idxs.
        """
        raise NotImplementedError('override-me')


class IndependentNodeTreeEstimator(TreeEstimator):
    def posterior(self, priors, observations, s_rng):
        """
        priors - an IdxsValsList of random variables
        observations - an IdxsValsList of corresponding samples

        returns - an IdxsValsList of posterior random variables
        """

        assert len(priors) == len(observations)

        # observation.idxs could be invalid.
        # They could be invalid, in the sense of describing samples that
        # could not have been drawn from the prior.
        # This code does not try to detect that situation.
        post_vals = [self.s_posterior_helper(p, o, s_rng)
            for p, o in zip(priors, observations)]

        # At this point, each post_vals[i] is connected to the original graph
        # formed of prior nodes.
        #
        # We want each post_vals[i] to be connected instead to the other
        # post_vals just created, corresponding to the posterior values of other
        # random vars.
        #
        # The way to get all the new_s_val nodes hooked up to one another is to
        # use the clone_keep_replacements function.

        # XXX: this clones everything. It should be possible to do a more
        #      selective clone of just the pieces that change.
        inputs = theano.gof.graph.inputs(priors.flatten() + post_vals)
        env = ienv.std_interactive_env(inputs, priors.flatten() + post_vals,
                clone_inputs_and_orphans=False)
        env.prefer_replace(
                zip(priors.valslist(), post_vals),
                reason='IndependentNodeTreeEstimator.posterior')

        # raise an exception if we created cycles
        env.toposort()

        # extract the cloned results from the env
        rval = IdxsValsList.fromlists(
                [env.newest(v) for v in priors.idxslist()],
                [env.newest(v) for v in post_vals])

        # remove all references in the variables to the env. Prepare them
        # to be inserted into another env if necessary.
        env.disown()
        return rval

    def s_posterior_helper(self, rv, obs, s_rng):
        """Return a posterior variable to replace the prior `rv.vals`
        """
        raise NotImplementedError('override-me')

    def log_likelihood(self, RVs, observations, llik):
        """
        The output from this function may be a random variable, if not all sources
        of randomness are observed.

        llik - a vector to which observation log-likelihoods will be added.
        """

        if len(RVs) != len(observations):
            raise TypeError('RVs and observations must have same length')

        for iv in RVs:
            if not montetheano.rv.is_rv(iv.vals):
                raise ValueError('non-random var in RVs element', iv)

        assignment = {}
        idxs_of = {}
        for rv, o in zip(RVs, observations):
            assignment[rv.vals] = o.vals
            idxs_of[o.vals] = o.idxs

        # All random variables that are not assigned should stay as the same
        # object so it can later be replaced
        # If this is not done this way, they get cloned
        #raw_RVs = [v for v in ancestors(RVs.flatten())
                #if montetheano.rv.is_raw_rv(v)]

        # Cast assignment elements to the right kind of thing
        assignment = montetheano.rv.typed_items(assignment)

        for rv_vals, obs_vals in assignment.items():
            lpdf = montetheano.rv.lpdf(rv_vals, obs_vals)
            llik = tensor.inc_subtensor(llik[idxs_of[obs_vals]], lpdf)

        # rewire the graph so that the posteriors depend on other
        # observations instead of each other.

        involved = [llik] + RVs.flatten() + observations.flatten()

        inputs = theano.gof.graph.inputs(involved)
        env = ienv.std_interactive_env(inputs, involved,
                clone_inputs_and_orphans=False)

        env.replace_all_sorted(
                zip(RVs.flatten(), observations.flatten()),
                reason='IndependentNodeTreeEstimator.log_likelihood')

        # raise an exception if we created cycles
        env.toposort()

        rval = env.newest(llik)
        env.disown()
        return rval


class AdaptiveParzen(theano.Op):
    """
    A heuristic estimator for the mu and sigma values of a GMM
    TODO: try to find this heuristic in the literature, and cite it - Yoshua
    mentioned the term 'elastic' I think?

    mus - matrix (N, M) of M, N-dimensional component centers
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, mus, prior_mu, prior_sigma):
        mus, prior_mu, prior_sigma = as_variables(
                mus, prior_mu, prior_sigma)
        if mus.ndim != 1:
            raise TypeError('mus must be vector', (mus, mus.type))
        if prior_mu.ndim > 0:
            raise TypeError('prior_mu', prior_mu)
        if prior_sigma.ndim > 0:
            raise TypeError('prior_sigma', prior_sigma)
        return theano.gof.Apply(self,
                [mus, prior_mu, prior_sigma],
                [mus.type(), mus.type(), mus.type()])

    def infer_shape(self, node, ishapes):
        mus_shape, prior_mu, prior_sigma = ishapes
        return [
                (tensor.maximum(1, mus_shape[0]),),
                (tensor.maximum(1, mus_shape[0]),),
                (tensor.maximum(1, mus_shape[0]),),
                ]

    def perform(self, node, inputs, outstorage):
        mus, prior_mu, prior_sigma = inputs
        mus_orig = mus.copy()
        mus = mus.copy()
        if mus.ndim != 1:
            raise TypeError('mus must be vector', mus)
        if len(mus) == 0:
            mus = numpy.asarray([prior_mu])
            sigma = numpy.asarray([prior_sigma])
        elif len(mus) == 1:
            mus = numpy.asarray([prior_mu] + [mus[0]])
            sigma = numpy.asarray([prior_sigma, prior_sigma * .5])
        elif len(mus) >= 2:
            order = numpy.argsort(mus)
            mus = mus[order]
            sigma = numpy.zeros_like(mus)
            sigma[1:-1] = numpy.maximum(
                    mus[1:-1] - mus[0:-2],
                    mus[2:] - mus[1:-1])
            if len(mus)>2:
                lsigma = mus[2] - mus[0]
                usigma = mus[-1] - mus[-3]
            else:
                lsigma = mus[1] - mus[0]
                usigma = mus[-1] - mus[-2]

            sigma[0] = lsigma
            sigma[-1] = usigma

            # XXX: is sorting them necessary anymore?
            # un-sort the mus and sigma
            mus[order] = mus.copy()
            sigma[order] = sigma.copy()

            if not numpy.all(mus_orig == mus):
                print 'orig', mus_orig
                print 'mus', mus
            assert numpy.all(mus_orig == mus)

            # put the prior back in
            mus = numpy.asarray([prior_mu] + list(mus))
            sigma = numpy.asarray([prior_sigma] + list(sigma))

        maxsigma = prior_sigma
        minsigma = prior_sigma / numpy.sqrt(len(mus))   # XXX: magic formula

        #print 'maxsigma, minsigma', maxsigma, minsigma

        sigma = numpy.clip(sigma, minsigma, maxsigma)

        weights = numpy.ones(len(mus), dtype=node.outputs[0].dtype)
        weights[0] = numpy.sqrt(1 + len(mus))

        # XXX: call asarray with dtype above to avoid re-copy here
        outstorage[0][0] = weights / weights.sum()
        outstorage[1][0] = mus.astype(node.outputs[1].dtype)
        outstorage[2][0] = sigma.astype(node.outputs[2].dtype)


class IndependentAdaptiveParzenEstimator(IndependentNodeTreeEstimator):
    """
    XXX
    """
    categorical_prior_strength = 1.0

    def s_posterior_helper(self, prior, obs, s_rng):
        """
        Return a posterior RV for rv having observed obs
        """
        # XXX: factor this out of the GM_BanditAlgo so that the density
        # modeling strategy is not coupled to the optimization strategy.
        try:
            dist_name = montetheano.rstreams.rv_dist_name(prior.vals)
        except:
            print >> sys.stderr, 'problem with', prior.vals
            raise
        if dist_name == 'normal':
            if obs.vals.ndim == 1:
                prior_mu, prior_sigma = prior.vals.owner.inputs[2:4]
                weights, mus, sigmas = AdaptiveParzen()(obs.vals,
                        prior_mu, prior_sigma)
                post_rv = s_rng.GMM1(weights, mus, sigmas,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
                return post_rv
            else:
                raise NotImplementedError()
        elif dist_name == 'uniform':
            if obs.vals.ndim == 1:
                low, high = prior.vals.owner.inputs[2:4]
                prior_mu = 0.5 * (high + low)
                prior_sigma = (high - low)
                weights, mus, sigmas = AdaptiveParzen()(obs.vals,
                        prior_mu, prior_sigma)
                post_rv = s_rng.BGMM1(weights, mus, sigmas, low, high,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
                return post_rv
            else:
                raise NotImplementedError()
        elif dist_name == 'lognormal':
            if obs.vals.ndim == 1:
                prior_mu, prior_sigma = prior.vals.owner.inputs[2:4]
                weights, mus, sigmas = AdaptiveParzen()(
                        tensor.log(tensor.maximum(obs.vals, 1.0e-8)),
                        prior_mu, prior_sigma)
                post_rv = s_rng.lognormal_mixture(weights, mus, sigmas,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
                return post_rv
            else:
                raise NotImplementedError()
        elif dist_name == 'quantized_lognormal':
            if obs.vals.ndim == 1:
                prior_mu, prior_sigma, step = prior.vals.owner.inputs[2:5]
                weights, mus, sigmas = AdaptiveParzen()(
                        tensor.log(obs.vals),
                        prior_mu, prior_sigma)
                post_rv = s_rng.quantized_lognormal_mixture(
                        weights, mus, sigmas, step,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
                return post_rv
            else:
                raise NotImplementedError()
        elif dist_name == 'categorical':
            if obs.vals.ndim == 1:
                prior_strength = self.categorical_prior_strength
                p = prior.vals.owner.inputs[1]
                if p.ndim != 1:
                    raise TypeError()
                prior_counts = p * p.shape[0] * tensor.sqrt(obs.vals.shape[0])
                pseudocounts = tensor.inc_subtensor(
                        (prior_strength * prior_counts)[obs.vals],
                        1)
                post_rv = s_rng.categorical(
                        p=pseudocounts / pseudocounts.sum(),
                        draw_shape = prior.vals.shape)
                return post_rv
            else:
                raise NotImplementedError()
        else:
            raise TypeError("unsupported distribution", dist_name)



class GM_BanditAlgo(TheanoBanditAlgo):
    """
    Graphical Model (GM) algo described in NIPS2011 paper.
    """

    mode = None

    n_startup_jobs = 30  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic

    n_EI_candidates = 256

    gamma = 0.20         # fraction of trials to consider as good
                         # this is should in theory be bandit-dependent

    def __init__(self, bandit, good_estimator, bad_estimator):
        TheanoBanditAlgo.__init__(self, bandit)
        self.good_estimator = good_estimator
        self.bad_estimator = bad_estimator
        self.build_helpers()

    def __getstate__(self):
        rval = dict(self.__dict__)
        for name in '_helper', '_prior_sampler':
            if name in rval:
                del rval[name]
        return rval

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        # this allows loading of old pickles
        # from before the current implementation
        # of __getstate__
        for name in '_helper', '_prior_sampler':
            if hasattr(self, name):
                delattr(self, name)

    def build_helpers(self):
        s_prior = IdxsValsList.fromlists(self.s_idxs, self.s_vals)
        s_obs = s_prior.new_like_self()

        # y_thresh is the boundary between 'good' and 'bad' regions of the
        # search space.
        y_thresh = tensor.scalar()

        yvals = tensor.vector()
        n_to_draw = self.s_N
        n_to_keep = tensor.iscalar()

        s_rng = montetheano.RandomStreams(self.seed + 9)

        GE = self.good_estimator
        BE = self.bad_estimator

        Gobs = s_obs.symbolic_take(where(yvals < y_thresh))
        Bobs = s_obs.symbolic_take(where(yvals >= y_thresh))

        # To "optimize" EI we just draw a pile of samples from the density
        # of good points and then just take the best of those.
        Gsamples = GE.posterior(s_prior, Gobs, s_rng)
        Bsamples = BE.posterior(s_prior, Bobs, s_rng)

        G_ll = GE.log_likelihood(Gsamples, Gsamples,
                llik = tensor.zeros((n_to_draw,)))
        B_ll = BE.log_likelihood(Bsamples, Gsamples,
                llik = tensor.zeros((n_to_draw,)))

        # subtract B_ll from G_ll
        log_EI = G_ll - B_ll
        keep_idxs = argsort(log_EI)[-n_to_keep:]

        # store all these vars for the unittests
        self.helper_locals = locals()
        del self.helper_locals['self']

    def suggest_from_prior(self, N):
        try:
            prior_sampler = self._prior_sampler
        except AttributeError:
            prior_sampler = self._prior_sampler = theano.function(
                    [self.helper_locals['n_to_draw']],
                    self.helper_locals['s_prior'].flatten(),
                    mode=self.mode)
        rvals = prior_sampler(N)
        return IdxsValsList.fromflattened(rvals)

    @property
    def _suggest_from_model_fn(self):
        try:
            helper = self._helper
        except AttributeError:
            def asdf(n_to_draw, n_to_keep, y_thresh, yvals, s_obs, Gsamples,
                    keep_idxs, Gobs, Bobs, **kwargs):
                return theano.function(
                    [n_to_draw, n_to_keep, y_thresh, yvals] + s_obs.flatten(),
                    (Gsamples.symbolic_take(keep_idxs).flatten()
                        + Gobs.flatten()
                        + Bobs.flatten()
                        ),
                    allow_input_downcast=True,
                    mode=self.mode,
                    )
            helper = self._helper = asdf(**self.helper_locals)
        return helper


    def suggest_from_model(self, ivls, N):
        helper = self._suggest_from_model_fn

        ylist = numpy.asarray(sorted(ivls['losses']['ok'].vals), dtype='float')
        y_thresh_idx = int(self.gamma * len(ylist))
        y_thresh = ylist[y_thresh_idx : y_thresh_idx + 2].mean()

        logger.info('GM_BanditAlgo splitting results at y_thresh = %f'
                % y_thresh)
        logger.info('GM_BanditAlgo keeping %i results as good'
                % y_thresh_idx)
        logger.info('GM_BanditAlgo keeping %i results as bad'
                % (len(ylist) - y_thresh_idx))
        logger.info('GM_BanditAlgo good scores: %s'
                % str(ylist[:y_thresh_idx]))

        x_all = ivls['x_IVLs']['ok'].as_list()
        y_all_iv = ivls['losses']['ok'].as_list()

        assert y_all_iv.idxset() == x_all.idxset(), (y_all_iv.idxset(),
                x_all.idxset())

        for pseudo_bad_status in 'new', 'running':
            logger.info('GM_BanditAlgo assigning bad scores to %i new jobs'
                    % len(ivls['losses'][pseudo_bad_status].idxs))
            x_all.stack(ivls['x_IVLs'][pseudo_bad_status])
            y_all_iv.stack(IdxsVals(
                ivls['losses'][pseudo_bad_status].idxs,
                [y_thresh + 1] * len(ivls['losses'][pseudo_bad_status].idxs)))
            assert y_all_iv.idxset() == x_all.idxset(), (y_all_iv.idxset(),
                    x_all.idxset())

        # renumber the configurations in x_all to be 0 .. (n_train - 1)
        idmap = y_all_iv.reindex()
        idmap = x_all.reindex(idmap)

        assert y_all_iv.idxset() == x_all.idxset(), (y_all_iv.idxset(),
                x_all.idxset())

        assert numpy.all(y_all_iv.idxs == numpy.arange(len(y_all_iv.idxs))), (
                y_all_iv.idxs)

        y_all = y_all_iv.as_numpy(vdtype=theano.config.floatX).vals
        x_all = x_all.as_numpy_floatX()

        logger.info('GM_BanditAlgo drawing %i candidates'
                % self.n_EI_candidates)

        helper_rval = helper(self.n_EI_candidates, N,
            y_thresh, y_all, *x_all.flatten())
        assert len(helper_rval) == 6 * len(x_all)

        keep_flat = helper_rval[:2 * len(x_all)]
        Gobs_flat = helper_rval[2 * len(x_all): 4 * len(x_all)]
        Bobs_flat = helper_rval[4 * len(x_all):]
        assert len(keep_flat) == len(Gobs_flat) == len(Bobs_flat)

        Gobs = IdxsValsList.fromflattened(Gobs_flat)
        Bobs = IdxsValsList.fromflattened(Bobs_flat)

        # guard against book-keeping error
        # ensure that all observations were counted as either good or bad
        gis = Gobs.idxset()
        bis = Bobs.idxset()
        xis = x_all.idxset()
        assert len(xis) == len(y_all)
        assert gis.union(bis) == xis
        assert gis.intersection(bis) == set()

        rval = IdxsValsList.fromflattened(keep_flat)
        # relabel the return values to be elements 0 ... N - 1
        rval.reindex()
        return rval

    def suggest(self, trials, results, N):
        ivls = self.idxs_vals_by_status(trials, results)
        if len(ivls['losses']['ok'].idxs) < self.n_startup_jobs:
            logger.info('GM_BanditAlgo warming up %i/%i'
                    % (len(ivls['losses']['ok'].idxs), self.n_startup_jobs))
            return self.suggest_ivl(self.suggest_from_prior(N))
        else:
            return self.suggest_ivl(self.suggest_from_model(ivls, N))


def AdaptiveParzenGM(bandit):
    GE = idxs_vals_rnd.IndependentAdaptiveParzenEstimator()
    BE = idxs_vals_rnd.IndependentAdaptiveParzenEstimator()
    rval = GM_BanditAlgo(bandit,
            good_estimator=GE,
            bad_estimator=BE)
    return rval


def ops(fn, OpCls):
    if isinstance(fn, list):
        return [v.owner for v in montetheano.for_theano.ancestors(fn)
                if v.owner and isinstance(v.owner.op, OpCls)]
    else:
        return [ap for ap in fn.maker.env.toposort()
            if isinstance(ap.op, OpCls)]


def categoricals(fn):
    return ops(fn, montetheano.distributions.Categorical)


class NestedUniform(hyperopt.bandits.Base):
    """
    Problem in which a uniform applies only to some of the examples.
    """

    loss_target = -2  # best score is +2

    def __init__(self):
        hyperopt.bandits.Base.__init__(self,
                one_of(
                    rSON2(
                        'kind', 'raw'),
                    rSON2(
                        'kind', 'negcos',
                        'amp', uniform(0, 1))))

    def score(self, pt):
        r = numpy.random.uniform()
        if pt['kind'] == 'negcos':
            r += pt['amp']
        return r


class IndependentNullEstimator(idxs_vals_rnd.IndependentNodeTreeEstimator):
    """Dummy node estimator that keeps graphs small.
    Useful for debugging IndependentNodeTreeEstimator
    """

    def s_posterior_helper(self, prior, obs, s_rng):
        try:
            dist_name = montetheano.rstreams.rv_dist_name(prior.vals)
        except:
            print >> sys.stderr, 'problem with', prior.vals
            raise

        if dist_name == 'normal':
            if obs.vals.ndim == 1:
                return s_rng.normal(
                        mu=obs.vals.mean(),
                        sigma=0.75,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
            else:
                raise NotImplementedError()
        elif dist_name == 'uniform':
            if obs.vals.ndim == 1:
                low, high = prior.vals.owner.inputs[2:4]
                return s_rng.uniform(
                        low=obs.vals.min(),
                        high=high,
                        draw_shape=prior.vals.shape,
                        ndim=prior.vals.ndim,
                        dtype=prior.vals.dtype)
            else:
                raise NotImplementedError()
        elif dist_name == 'lognormal':
            raise NotImplementedError()
        elif dist_name == 'categorical':
            if obs.vals.ndim == 1:
                pseudocounts = prior.vals.owner.inputs[1] + obs.vals.sum()
                post_rv = s_rng.categorical(
                        p=pseudocounts / pseudocounts.sum(),
                        draw_shape = prior.vals.shape)
                return post_rv
            else:
                raise NotImplementedError()
        else:
            raise TypeError("unsupported distribution", dist_name)


class TestIndependentNodeTreeEstimator(unittest.TestCase):
    def setUp(self):
        self.TE = IndependentNullEstimator()
        self.bandit = NestedUniform()
        self.experiment = SerialExperiment(
            GM_BanditAlgo(self.bandit,
                    good_estimator=IndependentNullEstimator(),
                    bad_estimator=IndependentNullEstimator()))

        self.s_rng = montetheano.RandomStreams(123)
        prior_idxs, prior_vals, s_N = self.bandit.template.theano_sampler(self.s_rng)
        #print prior_idxs
        #print prior_vals
        self.prior = IdxsValsList.fromlists(
                [i for i in prior_idxs if i is not None],
                [v for v in prior_vals if v is not None])
        self.s_N = s_N
        self.observations = self.prior.new_like_self()
        for i, o in enumerate(self.observations):
            o.idxs.name = 'Obs_idxs{%i}' % i
            o.vals.name = 'Obs_vals{%i}' % i

    def test_posterior_op_count(self):
        posterior = self.TE.posterior(self.prior, self.observations, self.s_rng)
        if 0:
            for i, p in enumerate(posterior):
                print ''
                print 'POSTERIOR', i
                print '============'
                theano.printing.debugprint([p.idxs, p.vals])

        assert len(categoricals(posterior.flatten())) == 2
        # one for the shape
        # one for the posterior of the choice variable

    def test_posterior_runs(self):
        posterior = self.TE.posterior(self.prior, self.observations, self.s_rng)
        f = theano.function([self.s_N] + self.observations.flatten(),
                posterior.flatten(),
                allow_input_downcast=True)
        assert len(categoricals(posterior.flatten())) == 2

        if 0:
            # we should be able to optimize out the first categorical
            # because it is used only for shape.  This doesn't currently work.
            theano.printing.debugprint(f)
            assert len(categoricals(f)) == 1

        obs_vals = [0,1,2,3], [0,0,0,1], [3], [.95]
        idxs0, vals0, idxs1, vals1  = f(100, *obs_vals)

        assert numpy.all(idxs0 == range(100))
        assert (vals0 == 0).sum() > 10
        assert (vals0 == 1).sum() > 10
        assert (vals0 == 1).sum()  + (vals0 == 0).sum() == 100

        assert len(idxs1) == (vals0 == 1).sum()
        assert vals1.min() >= .95
