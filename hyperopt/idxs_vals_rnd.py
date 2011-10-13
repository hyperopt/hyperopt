"""
Idxs-Vals reprepresentation of random samples

XXX: What is Idxs-Vals representation?
"""
import sys

import numpy

import theano
from theano import tensor

import montetheano
from montetheano.for_theano import ancestors
from montetheano.for_theano import as_variable
from montetheano.for_theano import find
from montetheano.for_theano import clone_get_equiv
from montetheano.for_theano import clone_keep_replacements

import ienv

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

    def take(self, elements):
        """Advanced sparse vector indexing by int-list `elements`
        """
        pos = find(self.idxs, elements)
        return IdxsVals(self.idxs[pos], self.vals[pos])


class IdxsValsList(list):
    """
    List of IdxsVals instances

    N.B. This class is sometimes used to store symbolic variables, and
    sometimes used to store numeric variables.  Not all operations work in
    both cases, but many do.

    """
    def idxslist(self):
        return [e.idxs for e in self]

    def valslist(self):
        return [e.vals for e in self]

    def take(self, subelements):
        """Return a new IdxsValsList of the same length as self, whose elements
        are restricted to contain only the given `subelements`.
        """
        # make a variable outside the loop to help theano's merge-optimizer
        subel = theano.tensor.as_tensor_variable(subelements)
        return self.__class__([e.take(subel) for e in self])

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
            mus = numpy.asarray([prior_mu])
            sigma = numpy.asarray([prior_sigma])
        elif len(mus) >= 2:
            low = prior_mu - 2 * prior_sigma
            high = prior_mu + 2 * prior_sigma
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

            sigma[0] = max(mus[0]-low, lsigma)
            sigma[-1] = max(high - mus[-1], usigma)

            # un-sort the mus and sigma
            mus[order] = mus.copy()
            sigma[order] = sigma.copy()

            #print mus, sigma
            if not numpy.all(mus_orig == mus):
                print 'orig', mus_orig
                print 'mus', mus


            assert numpy.all(mus_orig == mus)

        maxsigma = prior_sigma
        minsigma = 3.0 * prior_sigma / len(mus)   # XXX: magic formula

        sigma = numpy.clip(sigma, minsigma, maxsigma)

        outstorage[0][0] = numpy.ones(len(mus), dtype=node.outputs[0].dtype) / len(mus)
        outstorage[1][0] = mus.astype(node.outputs[1].dtype)
        outstorage[2][0] = sigma.astype(node.outputs[2].dtype)


class IndependentAdaptiveParzenEstimator(IndependentNodeTreeEstimator):
    """
    XXX
    """
    # pseudocount is prior_strength / #choices
    categorical_prior_strength = 2.0

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
                prior_counts = prior.vals.owner.inputs[1]  #XXX: name this?
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

