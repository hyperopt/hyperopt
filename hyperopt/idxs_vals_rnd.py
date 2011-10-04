"""
Idxs-Vals reprepresentation of random samples

XXX: What is Idxs-Vals representation?
"""
import sys
import theano
from theano import tensor

import montetheano
from montetheano.for_theano import ancestors
from montetheano.for_theano import as_variable
from montetheano.for_theano import restrict
from montetheano.for_theano import clone_keep_replacements

def as_variables(*args):
    return [tensor.as_tensor_variable(a) for a in args]


class IdxsVals(object):
    def __init__(self, i, v):
        self.idxs = i  # symbolic integer vector
        self.vals = v  # symbolic ndarray with same length as self.idxs

    def take(self, elements):
        """Advanced sparse vector indexing by int-list `elements`
        """
        pos = restrict(elements, self.idxs)
        return IdxsVals(self.idxs[pos], self.vals[pos])


class IdxsValsList(list):
    """
    List of IdxsVals instances
    """
    def idxslist(self):
        return [e.idxs for e in self]

    def valslist(self):
        return [e.vals for e in self]

    def take(self, subelements):
        """Return a new IdxsValsList of the same length as self, whose elements
        are restricted to containing only the given `subelements`.
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
        """
        rval = []
        for e in self:
            rval.extend([e.idxs, e.vals])
        return rval


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

        blockers = observations.flatten()
        rvs_anc = ancestors(priors.flatten(), blockers=blockers)
        frontier = [r for r in rvs_anc if r.owner is None or r in blockers]
        _frontier, cloned_post_ivs = clone_keep_replacements(
                i=frontier,
                o=priors.idxslist() + post_vals,
                replacements=dict(zip(priors.valslist(), post_vals)))
        assert len(cloned_post_ivs) == 2 * len(priors)
        return IdxsValsList.fromlists(
                cloned_post_ivs[:len(priors)],
                cloned_post_ivs[len(priors):])

    def s_posterior_helper(self, rv, obs, s_rng):
        """Return a posterior variable to replace the prior `rv.vals`
        """
        raise NotImplementedError('override-me')

    def log_likelihood(self, posterior, observations, N=None):
        """
        The output from this function may be a random variable, if not all sources
        of randomness are observed.
        """

        if len(posterior) != len(observations):
            raise TypeError('posterior and observations must have same length')

        for iv in posterior:
            if not montetheano.rv.is_rv(iv.vals):
                raise ValueError('non-random var in posterior element', iv)

        if N is None:
            raise NotImplementedError('Need N for now')

        assignment = {}
        idxs_of = {}
        for p, o in zip(posterior, observations):
            assignment[p.vals] = o.vals
            idxs_of[o.vals] = o.idxs
            idxs_of[p.vals] = p.idxs

        # All random variables that are not assigned should stay as the same
        # object so it can later be replaced
        # If this is not done this way, they get cloned
        RVs = [v for v in ancestors(posterior.flatten())
                if montetheano.rv.is_raw_rv(v)]
        for rv in RVs:
            if rv not in assignment:
                #assignment[rv] = rv
                # TODO: Consider a protocol of adding a .idxs attribute
                # to the .vals RV itself?  This might be better than
                # the IdxsVals system currently implemented.
                raise NotImplementedError('dont know idxs for RV', rv)

        # Cast assignment elements to the right kind of thing
        assignment = montetheano.rv.typed_items(assignment)

        llik = tensor.zeros((N,))
        for rv, sample in assignment.items():
            lpdf = montetheano.rv.lpdf(rv, sample)
            llik = tensor.inc_subtensor(llik[idxs_of[sample]], lpdf)

        dfs_variables = ancestors([llik], blockers=assignment.keys())
        frontier = [r for r in dfs_variables
                if r.owner is None or r in assignment.keys()]
        cloned_inputs, cloned_outputs = clone_keep_replacements(
                frontier,
                [llik],
                replacements=assignment)
        cloned_llik, = cloned_outputs
        return IdxsVals(tensor.arange(N), cloned_llik)


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

    def make_node(self, mus, low, high, minsigma):
        mus, low, high, minsigma = as_variables(mus, low, high, minsigma)
        if mus.ndim != 1:
            raise TypeError('mus must be vector', (mus, mus.type))
        if low.ndim > 0:
            raise TypeError('low', low)
        if high.ndim > 0:
            raise TypeError('high', high)
        return theano.gof.Apply(self,
                [mus, low, high, minsigma],
                [tensor.vector(), mus.type(), mus.type()])

    def perform(self, node, inputs, outstorage):
        mus, low, high, minsigma = inputs
        mu_orig = mu.copy()
        mu = mu.copy()
        if len(mu) == 0:
            mu = numpy.asarray([0.5 * (low + high)])
            sigma = numpy.asarray([0.5 * (high - low)])
        elif len(mu) == 1:
            sigma = numpy.maximum(abs(mu-high), abs(mu-low))
        elif len(mu) >= 2:
            order = numpy.argsort(mu)
            mu = mu[order]
            sigma = numpy.zeros_like(mu)
            sigma[1:-1] = numpy.maximum(
                    mu[1:-1] - mu[0:-2],
                    mu[2:] - mu[1:-1])
            if len(mu)>2:
                lsigma = mu[2] - mu[0]
                usigma = mu[-1] - mu[-3]
            else:
                lsigma = mu[1] - mu[0]
                usigma = mu[-1] - mu[-2]

            sigma[0] = max(mu[0]-low, lsigma)
            sigma[-1] = max(high - mu[-1], usigma)

            # un-sort the mu and sigma
            mu[order] = mu.copy()
            sigma[order] = sigma.copy()

            print mu, sigma

            assert numpy.all(mu_orig == mu)


        sigma = numpy.maximum(sigma, minsigma)

        outstorage[0][0] = numpy.ones(len(mu)) / len(mu)
        outstorage[1][0] = mu[numpy.newaxis, :]
        outstorage[2][0] = sigma[numpy.newaxis, :]


class IndependentAdaptiveParzenEstimator(IndependentNodeTreeEstimator):
    """
    """

    def s_posterior_helper(self, prior, obs, s_rng):
        """
        Return a posterior RV for rv having observed obs
        """
        # XXX: factor this out of the GM_BanditAlgo so that the density
        # modeling strategy is not coupled to the optimization strategy.
        try:
            dist_name = montetheano.rstreams.rv_dist_name(prior.vals)
        except:
            print >> sys.stderr, 'problem with', rv
            raise
        if dist_name == 'normal':
            raise NotImplementedError()
        elif dist_name == 'uniform':
            # XXX: move this logic to the 'normal' case and
            #      and replace it with bounded_gmm
            weights, mus, sigmas = AdaptiveParzen()(obs.vals, -5, 5, 1)
            post_rv = s_rng.GMM1(weights, mus, sigmas,
                    draw_shape=prior.vals.shape,
                    ndim=prior.vals.ndim)
            return post_rv
        elif dist_name == 'lognormal':
            raise NotImplementedError()
        elif dist_name == 'categorical':
            # weighted categorical
            raise NotImplementedError()
        else:
            raise TypeError("unsupported distribution", dist_name)

