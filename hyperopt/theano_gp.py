"""
Gaussian-process (GP)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import copy
import logging
import sys
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

import numpy
from scipy.optimize import fmin_l_bfgs_b
import theano
from theano.printing import Print
from theano import tensor
from theano.tensor import as_tensor_variable
from theano.sandbox.linalg import (diag, matrix_inverse, det, psd, trace)
import montetheano
import montetheano.distributions as mt_dist

from idxs_vals_rnd import IdxsVals, IdxsValsList
from theano_bandit_algos import TheanoBanditAlgo
from theano_gm import AdaptiveParzenGM

from gdist import set_difference

class picklable_instancemethod(object):
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name
    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.name)(*args, **kwargs)


def dots(*args):
    """Computes matrix product of N matrices"""
    rval = args[0]
    for a in args[1:]:
        rval = theano.tensor.dot(rval, a)
    return rval


def value(x):
    try:
        return x.get_value()
    except AttributeError:
        return x


class SparseGramGet(theano.gof.Op):
    """
    Particular kind of advanced indexing for reading an irregularly sliced
    subarray.

    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash((type(self)))

    def make_node(self, base, i0, i1):
        base, i0, i1 = map(tensor.as_tensor_variable, (base, i0, i1))
        if base.ndim != 2:
            raise TypeError('base not matrix', base)
        if i0.ndim != 1:
            raise TypeError('i0 not lvector', i0)
        if 'int' not in str(i0.dtype):
            raise TypeError('i0 not lvector', i0)
        if i1.ndim != 1:
            raise TypeError('i1 not lvector', i1)
        if 'int' not in str(i1.dtype):
            raise TypeError('i1 not lvector', i1)
        return theano.gof.Apply(self,
                [base, i0, i1],
                [base.type()])

    def perform(self, node, inputs, storage):
        base, i0, i1 = inputs
        #N.B. adv indexing copies, so no view/destroymap necessary
        storage[0][0] = base[i0[:, None], i1]

    def grad(self, inputs, g_outputs):
        base, i0, i1 = inputs
        base0 = tensor.zeros_like(base)
        gbase = sparse_gram_inc(base0, g_outputs[0], i0, i1)
        return [gbase, None, None]


sparse_gram_get = SparseGramGet()


class SparseGramSet(theano.gof.Op):
    """
    Particular kind of advanced indexing for modifying an irregularly sliced
    subarray.

    """
    def __init__(self, operation, destructive=False):
        if operation not in ('set', 'inc', 'mul'):
            raise ValueError('invalid operation', operation)
        self.operation = operation
        self.destructive = destructive
        if self.destructive:
            self.destroy_map = {0: [0]}
        else:
            self.destroy_map = {}

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.operation == other.operation
                and self.destructive == other.destructive)

    def __hash__(self):
        return hash((type(self),
            self.operation,
            self.destructive))

    def make_node(self, base, amt, i0, i1):
        base, amt, i0, i1 = map(tensor.as_tensor_variable,
                (base, amt, i0, i1))
        if base.ndim != 2:
            raise TypeError('base not matrix', base)
        if amt.ndim not in (0, 2):
            raise TypeError('amt not matrix or scalar', amt.type)
        if i0.ndim != 1:
            raise TypeError('i0 not lvector', i0)
        if 'int' not in str(i0.dtype):
            raise TypeError('i0 not lvector', i0)
        if i1.ndim != 1:
            raise TypeError('i1 not lvector', i1)
        if 'int' not in str(i1.dtype):
            raise TypeError('i1 not lvector', i1)
        return theano.gof.Apply(self,
                [base, amt, i0, i1],
                [base.type()])

    def perform(self, node, inputs, storage):
        base, amt, i0, i1 = inputs
        if self.destructive:
            rval = base
        else:
            rval = base.copy()

        storage[0][0] = rval
        if (len(i0) * len(i1) == 0):
            if amt.size == 0 or amt.shape == ():
                return
            # if one of the index vecs is empty, but amt is not a scalar
            # then the code below should generate a shape-based error

        if 0:
            print 'SparseGramSet operation', self.operation,
            print [id(n) for n in node.inputs]
            print 'SparseGramSet base', base.shape
            print 'SparseGramSet amt', amt.shape
            print 'SparseGramSet i0', i0
            print 'SparseGramSet i1', i1
            print 'SparseGramSet amtvals', amt

        if len(set(i0)) != len(i0):
            raise NotImplementedError('dups illegal in numpy adv. indexing')

        if len(set(i1)) != len(i1):
            raise NotImplementedError('dups illegal in numpy adv. indexing')

        if 'set' == self.operation:
            rval[i0[:, None], i1] = amt
        elif 'inc' == self.operation:
            rval[i0[:, None], i1] += amt
        elif 'mul' == self.operation:
            rval[i0[:, None], i1] *= amt
        else:
            assert 0, self.operation

    def grad(self, inputs, g_outputs):
        base, amt, i0, i1 = inputs
        z = self(*inputs)
        gz, = g_outputs
        if 'set' == self.operation:
            gbase = sparse_gram_set(gz, tensor.zeros_like(amt), i0, i1)
            gamt = sparse_gram_get(gz, i0, i1)
        elif 'inc' == self.operation:
            gbase = gz
            gamt = sparse_gram_get(gz, i0, i1)
        elif 'mul' == self.operation:
            gbase = sparse_gram_mul(gz, amt, i0, i1)
            gamt = (sparse_gram_get(gz, i0, i1)
                    * sparse_gram_get(base, i0, i1))
        if amt.ndim == 0:
            return [gbase, gamt.sum(), None, None]
        elif amt.ndim == 2:
            return [gbase, gamt, None, None]
        else:
            raise NotImplementedError()


sparse_gram_set = SparseGramSet('set')
sparse_gram_inc = SparseGramSet('inc')
sparse_gram_mul = SparseGramSet('mul')


class SquaredExponentialKernel(object):
    """

    K(x,y) = exp(-0.5 ||x-y||^2 / l^2)

    Attributes:

        log_lenscale - log(2 l^2)

    """

    def __init__(self, l=1, l_min=1e-4, l_max=1000):
        log_l = numpy.log(2 * (l ** 2))
        log_lenscale = theano.shared(log_l)
        if l_min is None:
            log_lenscale_min = None
        else:
            log_lenscale_min = numpy.log(2 * (l_min ** 2))
        if l_max is None:
            log_lenscale_max = None
        else:
            log_lenscale_max = numpy.log(2 * (l_max ** 2))
        self.log_lenscale = log_lenscale
        self.log_lenscale_min = log_lenscale_min
        self.log_lenscale_max = log_lenscale_max
        if self.log_lenscale.ndim != 0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)

    def lenscale(self, thing=None):
        if thing is None:
            thing = self.log_lenscale
        return numpy.sqrt(numpy.exp(value(thing)) / 2.0)

    def set_lenscale(self, new_l):
        candidate = numpy.log(2 * (value(new_l) ** 2))
        if candidate < self.log_lenscale_min:
            raise ValueError('lenscale too small')
            candidate = self.log_lenscale_min
        if candidate > self.log_lenscale_max:
            raise ValueError('lenscale too large')
            candidate = self.log_lenscale_max
        self.log_lenscale.set_value(candidate)

    def __str__(self):
        l = self.lenscale()
        (low, high), = self.param_bounds()
        if low is not None:
            low = self.lenscale(low)
        if high is not None:
            high = self.lenscale(high)
        return "%s{l=%s,bounds=(%s,%s)}" % (
                    self.__class__.__name__,
                    str(l), str(low), str(high))

    def random_reset(self, rng):
        self.log_lenscale.set_value(rng.randn())

    def params(self):
        return [self.log_lenscale]

    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        if x.ndim == y.ndim == 1:
            x = x.dimshuffle(0, 'x')
            y = y.dimshuffle(0, 'x')
        ll2 = tensor.exp(self.log_lenscale)  # 2l^2
        d = ((x ** 2).sum(axis=1).dimshuffle(0, 'x')
                + (y ** 2).sum(axis=1)
                - 2 * tensor.dot(x, y.T))
        K = tensor.exp(-d / ll2)
        return K


class LogSquaredExponentialKernel(SquaredExponentialKernel):
    def K(self, x, y):
        if x.ndim == y.ndim == 1:
            x = x.dimshuffle(0, 'x')
            y = y.dimshuffle(0, 'x')
        ll2 = tensor.exp(self.log_lenscale)  # 2l^2
        log = tensor.log
        lx = log(x)
        ly = log(y)
        d = ((lx ** 2).sum(axis=1).dimshuffle(0, 'x')
                + (ly ** 2).sum(axis=1)
                - 2 * tensor.dot(lx, ly.T))
        K = tensor.exp(-d / ll2)
        return K


class ExponentialKernel(SquaredExponentialKernel):
    """
    K(x,y) = exp(- ||x-y|| / l)

    Attributes:

        log_lenscale - log(l)

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.log_lenscale.ndim != 0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)

    def __str__(self):
        l = numpy.exp(self.log_lenscale.value)
        return "ExponentialKernel{l=%s}" % str(l)

    @classmethod
    def alloc(cls, l=1):
        log_l = numpy.log(l)
        log_lenscale = theano.shared(log_l)
        return cls(log_lenscale=log_lenscale)

    def K(self, x, y):
        l = tensor.exp(self.log_lenscale)
        d = ((x ** 2).sum(axis=1).dimshuffle(0, 'x')
                + (y ** 2).sum(axis=1)
                - 2 * tensor.dot(x, y.T))
        K = tensor.exp(-tensor.sqrt(d) / l)
        return K


class CategoryKernel(SquaredExponentialKernel):
    """
    K(x,y) is 1 if x==y else exp(-1/l)

    The idea is that it's like a SquaredExponentialKernel
    where every point is a distance of 1 from every other one,
    except itself.

    Attributes:

        log_lenscale -

    """
    def K(self, x, y):
        xx = x.reshape((x.shape[0],))
        yy = y.reshape((y.shape[0],))
        xx = xx.dimshuffle(0, 'x')  # drop cols because there should be 1
        yy = yy.dimshuffle(0)       # drop cols because there should be 1

        ll2 = tensor.exp(self.log_lenscale)  # 2l^2
        d = tensor.neq(xx, yy)
        K = tensor.exp(-d / ll2)
        return K


class GPR_math(object):
    """
    Formulae for Gaussian Process Regression

    # K - the gram matrix of training data
    # y - the vector of training data values
    # var_y - the vector of training data variances

    """

    def __init__(self, x, y, var_y, K_fn, N=None, min_variance=1e-6,
            dtype='float64'):
        self.x = x
        self.y = tensor.as_tensor_variable(y)
        self.var_y = tensor.as_tensor_variable(var_y)
        if self.y.dtype != dtype:
            raise TypeError('y has dtype', self.y.dtype)
        if self.var_y.dtype != dtype:
            raise TypeError('y has dtype', self.var_y.dtype)
        self.K_fn = K_fn
        self.min_variance = min_variance
        if N is None:
            self.N = self.y.shape[0]
        else:
            self.N = N
        self.dtype = dtype

    def kyn(self, x=None):
        """Return tuple: K, y, var_y, N"""
        if x is None:
            return self.K_fn(self.x, self.x), self.y, self.var_y, self.N
        else:
            return self.K_fn(self.x, x), self.y, self.var_y, self.N

    def s_nll(self):
        """ Marginal negative log likelihood of model

        :note: See RW.pdf page 37, Eq. 2.30.
        """
        K, y, var_y, N = self.kyn()
        rK = psd(K + var_y * tensor.eye(N))
        nll = (0.5 * dots(y, matrix_inverse(rK), y)
                + 0.5 * tensor.log(det(rK))
                + N / 2.0 * tensor.log(2 * numpy.pi))
        if nll.dtype != self.dtype:
            raise TypeError('nll dtype', nll.dtype)
        return nll

    def s_mean(self, x):
        """Gaussian Process mean at points x"""
        K, y, var_y, N = self.kyn()
        rK = psd(K + var_y * tensor.eye(N))
        alpha = tensor.dot(matrix_inverse(rK), y)

        K_x = self.K_fn(self.x, x)
        y_x = tensor.dot(alpha, K_x)
        if y_x.dtype != self.dtype:
            raise TypeError('y_x dtype', y_x.dtype)
        return y_x

    def s_variance(self, x):
        """Gaussian Process variance at points x"""
        K, y, var_y, N = self.kyn()
        rK = psd(K + var_y * tensor.eye(N))
        K_x = self.K_fn(self.x, x)
        var_x = 1 - diag(dots(K_x.T, matrix_inverse(rK), K_x))
        if var_x.dtype != self.dtype:
            raise TypeError('var_x dtype', var_x.dtype)
        return var_x

    def s_deg_of_freedom(self):
        """
        Degrees of freedom aka "effective number of parameters"
        of kernel smoother.

        Defined pg. 25 of Rasmussen & Williams.
        """
        K, y, var_y, N = self.kyn()
        rK = psd(K + var_y * tensor.eye(N))
        dof = trace(tensor.dot(K, matrix_inverse(rK)))
        if dof.dtype != self.dtype:
            raise TypeError('dof dtype', dof.dtype)
        return dof

    def s_expectation_lt_thresh(self, x, thresh):
        """
        return \int_{-inf}^{thresh} y p(y|x) dy
        """

        mu = self.s_mean(x)
        sigma = tensor.sqrt(
                tensor.maximum(self.s_variance(x),
                    self.min_variance))
        rval = 0.5 + 0.5 * tensor.erf((thresh - mu) / sigma)
        if rval.dtype != self.dtype:
            raise TypeError('rval dtype', rval.dtype)
        return rval

    def softplus_hack(self, x, thresh):
        """
        return approximation of \log(\int_{-inf}^{thresh} y p(y|x) dy)
        """

        mu = self.s_mean(x)
        sigma = tensor.sqrt(
                tensor.maximum(self.s_variance(x),
                    self.min_variance))
        rval = tensor.log1p(tensor.exp((thresh - mu) / sigma))
        if rval.dtype != self.dtype:
            raise TypeError('rval dtype', rval.dtype)
        return rval


def get_refinability(v, dist_name):
    v = v.vals
    if dist_name == 'uniform':
        params = [mt_dist.uniform_get_low(v), mt_dist.uniform_get_high(v)]
    elif dist_name == 'normal':
        params = [mt_dist.normal_get_mu(v), mt_dist.normal_get_sigma(v)]
    elif dist_name == 'lognormal':
        params = [mt_dist.lognormal_get_mu(v), mt_dist.lognormal_get_sigma(v)]
    elif dist_name == 'quantized_lognormal':
        params = [mt_dist.quantized_lognormal_get_mu(v),
                  mt_dist.quantized_lognormal_get_sigma(v),
                  mt_dist.quantized_lognormal_get_round(v)]
    for p in params:
        try:
            tensor.get_constant_value(p)
        except TypeError:
            return False
    return True


def categorical_parent(v):
    """
    Return the categorical variable c in the case that v = a[where(b==c)]
    """
    #theano.printing.debugprint(v)
    if not v.owner:
        raise ValueError(v)
    if not isinstance(v.owner.op, tensor.AdvancedSubtensor1):
        raise ValueError('expecting AdvancedSubtensor1', v)
    base, locs = v.owner.inputs
    if not (locs.owner and locs.owner.op == montetheano.for_theano.where):
        raise ValueError('expecting where', locs)
    value_eq_cat, = locs.owner.inputs
    if not (value_eq_cat.owner and value_eq_cat.owner.op == tensor.eq):
        raise ValueError('expecting equals', value_eq_cat)
    value, catvar = value_eq_cat.owner.inputs
    if not (catvar.owner and isinstance(catvar.owner.op, mt_dist.Categorical)):
        raise ValueError('expecting categorical', catvar)
    return catvar


class GP_BanditAlgo(TheanoBanditAlgo):
    """
    Gaussian proces - based BanditAlgo
    """
    params_l2_penalty = 0
    # fitting penalty on the lengthscales of kernels
    # might make sense to make this negative to blur out the ML solution.

    mode = None          # None to use theano's default compilation mode

    n_startup_jobs = 30  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic

    y_minvar = 1e-6      # minimum variance to permit for observations

    # EI_criterion can be
    #   'EI' for expected improvement
    #       - numerically terrible
    #       - unusable in practice because gradient is mostly near 0
    #   'log_EI' for the logarithm of EI
    #       - mathematically equivalent to optimizing EI
    #       - numerically good (in theory)
    #       - not implemented
    #   'softplus_hack' log(1 + sigmoid((mu-thresh)/sigma))
    #       - mathematically approximate (logistic sigmoid \approx erf)
    #       - numerically good
    #       - analytic form, fast computation
    EI_criterion = 'softplus_hack'

    EI_ambition = 0.75

    n_candidates_to_draw = 50
    # number of candidates returned by GM, and refined with gradient EI

    n_candidates_to_draw_in_GM = 200
    # number of candidates drawn within GM

    trace_on = False

    local_improvement_patience = 20
    # For this many iterations after the suggestion of a new best point, this
    # algorithm will use the GP (and not the GM).
    # N.B. than in parallel search, this number must be overestimated because
    # several time-steps will have elapsed by the time the best point switches
    # to status 'ok'.

    p_GP_during_exploration = .5
    # probability of using the GP when more than `local_improvement_patience`
    # iterations have elapsed since the last winning point was found.

    liar_percentile = .2
    # Attribute to jobs in progress the mean and variance of this quantile of
    # finished jobs.  0 would be most optimistic, 1 would be least.

    def trace(self, msg, obj):
        """Keep a trace of actions and results, useful for debugging"""
        if self.trace_on:
            try:
                _trace = self._trace
            except AttributeError:
                _trace = self._trace = []
            _trace.append((msg, copy.deepcopy(obj)))

    def theano_trace_mode(self):
        print >> sys.stderr, "WARNING: theano_trace_mode breaks pickling"
        class PrintEverythingMode(theano.Mode):
            def __init__(sss):
                def print_eval(i, node, fn):
                    for j, ij in enumerate(fn.inputs):
                        self.trace(('linker in', j, node.op), ij[0])
                    fn()
                    for j, ij in enumerate(fn.outputs):
                        self.trace(('linker out', j, node.op), ij[0])
                wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [print_eval])
                super(PrintEverythingMode, sss).__init__(wrap_linker, optimizer='fast_run')
        return PrintEverythingMode()

    def qln_cleanup(self, prior_vals, kern, candidate_vals):
        """
        Undo the smooth relaxation applied to quantized log-normal variables
        """
        round = tensor.get_constant_value(
                mt_dist.quantized_lognormal_get_round(
                    prior_vals))
        intlike = numpy.ceil(candidate_vals / float(round))
        assert intlike.ndim >= 1
        # in test problems, it seems possible to get stuck in a mode
        # where the EI optimum always gets rounded up to 3
        # and so 2 is never tried, even though it is actually the best point.
        intlike = numpy.maximum(1,
                intlike - self.numpy_rng.randint(2, size=len(intlike)))
        assert intlike.ndim >= 1
        rval = intlike * float(round)
        rval = rval.astype(prior_vals.dtype)
        return rval

    def post_refinement(self, candidates):
        # Coercing candidates from the form that was good for optimizing
        # to the form that is required by the configuration grammar
        for i, (iv, k, c) in enumerate(
                zip(self.s_prior, self.kernels, candidates)):
            if k in self.post_refinement_cleanup:
                f = self.post_refinement_cleanup[k]
                cvals = f(iv.vals, k, c.vals)
                assert cvals.shape == c.vals.shape
                assert str(cvals.dtype) == iv.vals.dtype
                assert cvals.ndim == iv.vals.ndim
                c.vals = cvals

    def __init__(self, bandit):
        TheanoBanditAlgo.__init__(self, bandit)
        self.numpy_rng = numpy.random.RandomState(234)
        self.s_prior = IdxsValsList.fromlists(self.s_idxs, self.s_vals)
        self.s_n_train = tensor.lscalar('n_train')
        self.s_n_test = tensor.lscalar('n_test')
        self.y_obs = tensor.vector('y_obs')
        self.y_obs_var = tensor.vector('y_obs_var')
        self.x_obs_IVL = self.s_prior.new_like_self()

        self.cand_x = self.s_prior.new_like_self()
        self.cand_EI_thresh = tensor.scalar()

        self.init_kernels()
        self.init_gram_weights()
        self.params.extend(self.convex_coefficient_params)
        self.param_bounds.extend(self.convex_coefficient_params_bounds)

        self.s_big_param_vec = tensor.vector()
        ### assumes all variables are refinable
        ### assumes all variables are vectors
        n_elements_used = 0
        for k, iv in zip(self.kernels, self.cand_x):
            if self.is_refinable[k]:
                n_elements_in_v = iv.idxs.shape[0]
                start = n_elements_used
                stop = n_elements_used + n_elements_in_v
                iv.vals = self.s_big_param_vec[start:stop]
                n_elements_used += n_elements_in_v

        self.gprmath = GPR_math(self.x_obs_IVL,
                self.y_obs,
                self.y_obs_var,
                picklable_instancemethod(self, 'K_fn'),
                N=self.s_n_train,
                min_variance=self.y_minvar)

        self.nll_obs = self.gprmath.s_nll()

        if self.EI_criterion == 'EI':
            self.cand_EI = self.gprmath.s_expectation_lt_thresh(
                    self.cand_x,
                    self.cand_EI_thresh)
        elif self.EI_criterion == 'softplus_hack':
            self.cand_EI = self.gprmath.softplus_hack(
                    self.cand_x,
                    self.cand_EI_thresh)
        else:
            raise ValueError('EI_criterion', self.EI_criterion)

        # self.gm_algo is used to draw candidates for subsequent refinement
        # It is also entirely responsible for choosing categorical variables.
        self.gm_algo = AdaptiveParzenGM(self.bandit)
        self.gm_algo.n_EI_candidates = self.n_candidates_to_draw_in_GM

    def __getstate__(self):
        rval = dict(self.__dict__)
        todel = [k for k, v in rval.items()
                if isinstance(v, theano.compile.Function)]
        for name in todel:
            del rval[name]
        return rval

    def init_kernels(self):
        self.kernels = []
        self.is_refinable = {}
        self.bounds = {}
        self.params = []
        self.param_bounds = []
        self.idxs_mulsets = {}
        self.post_refinement_cleanup = {}

        for iv in self.s_prior:
            dist_name = montetheano.rstreams.rv_dist_name(iv.vals)
            if dist_name == 'normal':
                k = SquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv, dist_name)
                self.bounds[k] = (None, None)
            elif dist_name == 'uniform':
                k = SquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv, dist_name)
                if self.is_refinable[k]:
                    low = tensor.get_constant_value(
                            mt_dist.uniform_get_low(iv.vals))
                    high = tensor.get_constant_value(
                            mt_dist.uniform_get_high(iv.vals))
                    self.bounds[k] = (low, high)
            elif dist_name == 'lognormal':
                k = LogSquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv, dist_name)
                self.bounds[k] = (1e-8, None)
            elif dist_name == 'quantized_lognormal':
                k = LogSquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv, dist_name)
                if self.is_refinable:
                    lbound = tensor.get_constant_value(
                            mt_dist.quantized_lognormal_get_round(
                                iv.vals))
                    self.bounds[k] = (lbound, None)
                    ff = picklable_instancemethod(self, 'qln_cleanup')
                    self.post_refinement_cleanup[k] = ff
            elif dist_name == 'categorical':
                # XXX: a better CategoryKernel would have different
                # similarities for different choices
                k = CategoryKernel()
                self.is_refinable[k] = False
                # refinable is false, so not setting bounds
            else:
                raise TypeError("unsupported distribution", dist_name)

            self.kernels.append(k)
            self.params.extend(k.params())
            self.param_bounds.extend(k.param_bounds())
            # XXX : to be more robust, it would be nice to build an Env with
            # the idxs as outputs, and then run the MergeOptimizer on it.
            self.idxs_mulsets.setdefault(iv.idxs, []).append(k)

    def init_gram_weights_helper(self, idxs, parent_weight, cparent):
        if parent_weight.ndim != 0:
            raise TypeError(parent_weight.type)
        kerns = self.idxs_mulsets[idxs]
        cat_kerns = [k for k, iv in zip(self.kernels, self.s_prior) if (
            isinstance(k, CategoryKernel)
            and k in kerns
            and iv.vals in cparent.values())]
        if len(cat_kerns) == 0:
            self.gram_weights[idxs] = parent_weight
        elif len(cat_kerns) == 1:
            # We have a mulset with one categorical variable in it.
            param = theano.shared(numpy.asarray(0.0))
            self.convex_coefficient_params.append(param)
            self.convex_coefficient_params_bounds.append((-5, 5))
            weight = tensor.nnet.sigmoid(param)
            # call recursively for each mulset
            # that corresponds to a slice out of idxs
            cat_vals = self.s_prior[self.kernels.index(cat_kerns[0])].vals
            self.weight_to_children[cat_vals] = parent_weight * weight
            sub_idxs_list = [sub_idxs for sub_idxs in self.idxs_mulsets
                    if cparent[sub_idxs] == cat_vals]
            assert all(si.owner.inputs[0] == idxs for si in sub_idxs_list)
            for sub_idxs in sub_idxs_list:
                    self.init_gram_weights_helper(
                            sub_idxs,
                            parent_weight=self.weight_to_children[cat_vals],
                            cparent=cparent)
            #print 'adding gram_weight', idxs
            #theano.printing.debugprint(parent_weight * (1 - weight))
            self.gram_weights[idxs] = parent_weight * (1 - weight)
        else:
            # We have a mulset with multiple categorical variables in it.
            # in this case the parent_weight must be divided among
            # this mulset itself, and each of the contained mulsets
            # (corresponding to the choices within each categorical variable)
            n_terms = len(cat_kerns) + 1
            params = theano.shared(numpy.zeros(n_terms))
            self.convex_coefficient_params.append(params)
            self.convex_coefficient_params_bounds.extend([(-5, 5)] * n_terms)
            weights = tensor.nnet.softmax(params) * parent_weight
            if weights.ndim == 2:
                # dimshuffle gets rid of the extra dimension inserted by the
                # stupid softmax implementation.  Get rid of this once
                # Theano's softmax vector branch is merged to master.
                weights = weights.dimshuffle(1)
            for i, k in enumerate(cat_kerns):
                # we're looking for sub_idxs that are formed by
                # advanced-indexing into `idxs` at positions determined
                # by the random choices of the variable corresponding to
                # kernel k
                weights_i = weights[i]
                cat_vals = self.s_prior[self.kernels.index(k)].vals
                self.weight_to_children[cat_vals] = weights_i
                sub_idxs_list = [sub_idxs for sub_idxs in self.idxs_mulsets
                        if cparent[sub_idxs] == cat_vals]
                assert all(si.owner.inputs[0] == idxs for si in sub_idxs_list)
                for sub_idxs in sub_idxs_list:
                    self.init_gram_weights_helper(
                            sub_idxs,
                            parent_weight=weights_i,
                            cparent=cparent)
            self.gram_weights[idxs] = weights[len(cat_kerns)]

    def init_gram_weights(self):
        """ Initialize mixture component weights of the hierarchical kernel.
        """
        try:
            self.gram_weights
            raise Exception('already initialized weights')
        except AttributeError:
            self.convex_coefficient_params = []
            self.convex_coefficient_params_bounds = []
            self.gram_weights = {}
            self.weight_to_children = {}

        # XXX : to be more robust, it would be better to build an Env
        # with the idxs as outputs, and then run the MergeOptimizer on
        # it.

        # Precondition: all idxs are either the root ARange or an
        # AdvancedSubtensor1 of some other idxs variable
        root_idxs = None
        cparent = {}
        for ii in self.idxs_mulsets:
            assert ii.owner
            if isinstance(ii.owner.op, tensor.ARange):
                assert root_idxs in (ii, None)
                root_idxs = ii
                cparent[ii] = None
            else:
                if isinstance(ii.owner.op, tensor.AdvancedSubtensor1):
                    assert ii.owner.inputs[0] in self.idxs_mulsets
                    cparent[ii] = categorical_parent(ii)
                else:
                    raise Exception('WHAT IS', ii)

        self.categorical_parent_of_idxs = cparent
        self.init_gram_weights_helper(root_idxs, as_tensor_variable(1.0), cparent)

    def K_fn(self, x0, x1):
        """
        :param x0: an IdxsValsList of symbolic variables
        :param x1: an IdxsValsList of symbolic variables

        :returns: symbolic gram matrix
        """

        gram_matrices = {}
        gram_matrices_idxs = {}
        for k, iv_prior, iv0, iv1 in zip(self.kernels, self.s_prior, x0, x1):
            gram = k.K(iv0.vals, iv1.vals)
            gram_matrices.setdefault(iv_prior.idxs, []).append(gram)
            gram_matrices_idxs.setdefault(iv_prior.idxs, [iv0.idxs, iv1.idxs])

        nx1 = self.s_n_train if x1 is x0 else self.s_n_test
        # N.B. the asarray works around mysterious Theano casting rules...
        base = tensor.alloc(numpy.asarray(0.0), self.s_n_train, nx1)
        for idxs, grams in gram_matrices.items():
            prod = self.gram_weights[idxs] * tensor.mul(*grams)
            base = sparse_gram_inc(base, prod, *gram_matrices_idxs[idxs])

        # we need to top up the gram matrix with weighted blocks of 1s
        # every time a categorical variable
        # sliced categoricals
        if 1:
            sliced_vals = set(self.categorical_parent_of_idxs.values())
            sliced_vals.remove(None)
            if 0:
                print sliced_vals
                for v in sliced_vals:
                    print v, [iv for iv in self.s_prior if iv.vals is v]
            # assert there are no dups
            assert len(sliced_vals) == len(set(sliced_vals))

            cparent = self.categorical_parent_of_idxs

            for prior_vals in sliced_vals:
                weight = self.weight_to_children[prior_vals]
                pos_of_child_idxs = [i for i, iv in enumerate(self.s_prior)
                        if cparent[iv.idxs] == prior_vals]
                child_idxs0 = [x0[i].idxs for i in pos_of_child_idxs]
                child_idxs1 = [x1[i].idxs for i in pos_of_child_idxs]
                iii = self.s_prior.valslist().index(prior_vals)
                assert iii >= 0
                base = sparse_gram_inc(base, weight,
                        set_difference(x0[iii].idxs, *child_idxs0),
                        set_difference(x1[iii].idxs, *child_idxs1))
        assert base.dtype == 'float64'
        return base

    def prepare_GP_training_data(self, ivls):
        # The mean and std should be estimated only from
        # the initial jobs that were sampled randomly.
        ok_idxs = ivls['losses']['ok'].idxs
        ok_vals = ivls['losses']['ok'].vals
        if (max(ok_idxs[:self.n_startup_jobs])
                < min([sys.maxint] + ok_idxs[self.n_startup_jobs:])):
            y_mean = numpy.mean(ok_vals[:self.n_startup_jobs])
            y_std = numpy.std(ok_vals[:self.n_startup_jobs])
        else:
            # TODO: extract the elements of losses['ok'] corresponding to
            # initial random jobs, and use them to estimate y_mean, y_std
            raise NotImplementedError()
        y_std = numpy.maximum(y_std, numpy.sqrt(self.y_minvar))
        del ok_idxs, ok_vals

        x_all = ivls['x_IVLs']['ok'].as_list()
        y_all_iv = ivls['losses']['ok'].as_list()
        y_var_iv = ivls['losses_variance']['ok'].as_list()

        # -- HEURISTIC: assign running jobs the same performance as the
        #    some percentile of the observed losses.
        liar_y_pos = numpy.argsort(ivls['losses']['ok'].vals)[
                int(self.liar_percentile * len(ivls['losses']['ok'].vals))]
        liar_y_mean = ivls['losses']['ok'].vals[liar_y_pos]
        liar_y_var = ivls['losses_variance']['ok'].vals[liar_y_pos]

        for pseudo_bad_status in 'new', 'running':
            logger.info('GM_BanditAlgo assigning bad scores to %i new jobs'
                    % len(ivls['losses'][pseudo_bad_status].idxs))
            x_all.stack(ivls['x_IVLs'][pseudo_bad_status])
            y_all_iv.stack(IdxsVals(
                ivls['losses'][pseudo_bad_status].idxs,
                [liar_y_mean] * len(ivls['losses'][pseudo_bad_status].idxs)))
            y_var_iv.stack(IdxsVals(
                ivls['losses_variance'][pseudo_bad_status].idxs,
                [liar_y_var] * len(ivls['losses'][pseudo_bad_status].idxs)))

        # renumber the configurations in x_all to be 0 .. (n_train - 1)
        idmap = y_all_iv.reindex()
        idmap = y_var_iv.reindex(idmap)
        idmap = x_all.reindex(idmap)

        assert y_all_iv.idxset() == y_var_iv.idxset() == x_all.idxset()

        assert numpy.all(y_all_iv.idxs == numpy.arange(len(y_all_iv.idxs)))
        assert numpy.all(y_var_iv.idxs == numpy.arange(len(y_all_iv.idxs)))

        y_all = y_all_iv.as_numpy(vdtype=theano.config.floatX).vals
        y_var = y_var_iv.as_numpy(vdtype=theano.config.floatX).vals
        x_all = x_all.as_numpy_floatX()

        y_all = (y_all - y_mean) / (1e-8 + y_std)
        y_var /= (1e-8 + y_std) ** 2

        assert y_all.shape == y_var.shape
        if y_var.min() < -1e-6:
            raise ValueError('negative variance encountered in results')
        y_var = numpy.maximum(y_var, self.y_minvar)
        return x_all, y_all, y_mean, y_var, y_std

    def fit_GP(self, x_all, y_all, y_mean, y_var, y_std, maxiter=1000):
        """
        Fit GPR kernel parameters by minimizing magininal nll.

        Returns: None

        Side effect: chooses optimal kernel parameters.
        """
        if y_std <= 0:
            raise ValueError('y_std must be postiive', y_std)

        if list(sorted(x_all.idxset())) != range(len(x_all.idxset())):
            raise NotImplementedError('need contiguous 0-based indexes on x')
        n_train = len(y_all)


        #TODO: optimize this function by making theano include the get_pt and
        #      set_pt, and theano function returns gradient and function value
        #      at once.

        self._GP_n_train = n_train
        self._GP_x_all = x_all
        self._GP_y_all = y_all
        self._GP_y_var = y_var
        self._GP_y_mean = y_mean
        self._GP_y_std = y_std

        if hasattr(self, 'nll_fn'):
            nll_fn = self.nll_fn
            dnll_dparams = self.dnll_dparams
        else:
            cost = (self.nll_obs
                + self.params_l2_penalty * sum(
                    [(p ** 2).sum() for p in self.params]))
            nll_fn = self.nll_fn = theano.function(
                    [self.s_n_train, self.s_n_test, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten(),
                    cost,
                    allow_input_downcast=True,
                    mode=self.mode,
                    )
            dnll_dparams = self.dnll_dparams = theano.function(
                    [self.s_n_train, self.s_n_test, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten(),
                    tensor.grad(cost, self.params),
                    allow_input_downcast=True,
                    mode=self.mode)
            print('Compiled nll_fn with %i thunks' %
                    len(nll_fn.maker.env.toposort()))
            print('Compiled dnll_fn with %i thunks' %
                    len(dnll_dparams.maker.env.toposort()))

        lbounds = []
        ubounds = []
        for lb, ub in self.param_bounds:
            lbounds.extend(numpy.asarray(value(lb)).flatten())
            ubounds.extend(numpy.asarray(value(ub)).flatten())
        bounds = numpy.asarray([lbounds, ubounds]).T

        # re-initialize params to eliminate warm-start bias
        for k in self.kernels:
            k.random_reset(self.numpy_rng)

        # re-initialize coefficients to even weights
        for p in self.convex_coefficient_params:
            p.set_value(0 * p.get_value())

        def get_pt():
            rval = []
            for p in self.params:
                v = p.get_value().flatten()
                rval.extend(v)
            return numpy.asarray(rval)

        def set_pt(pt):
            i = 0
            self.trace('fit_GP set_pt', pt)
            for p in self.params:
                assert p.dtype == 'float64'
                shape = p.get_value(borrow=True).shape
                size = int(numpy.prod(shape))
                p.set_value(pt[i:i + size].reshape(shape))
                i += size
            assert i == len(pt)

        n_calls = [0]
        def f(pt):
            n_calls[0] += 1
            set_pt(pt)
            rval = nll_fn(self._GP_n_train,
                    self._GP_n_train,
                    self._GP_y_all,
                    self._GP_y_var,
                    *self._GP_x_all.flatten())
            self.trace('fit_GP f', rval)
            return rval

        def df(pt):
            n_calls[0] += 1
            set_pt(pt)
            dparams = dnll_dparams(self._GP_n_train,
                    self._GP_n_train,
                    self._GP_y_all,
                    self._GP_y_var,
                    *self._GP_x_all.flatten())
            rval = []
            for dp in dparams:
                rval.extend(dp.flatten())

            rval = numpy.asarray(rval)
            self.trace('fit_GP df', rval)
            return rval

        self.trace('fit_GP start_pt', get_pt())

        best_pt, best_value, best_d = fmin_l_bfgs_b(f,
                get_pt(),
                df,
                maxfun=maxiter,
                bounds=bounds,
                iprint=-1)
        logger.info('fit_GP best value: %f' % best_value)
        set_pt(best_pt)
        self.trace('fit_GP best_pt', best_pt)
        return best_value

    def GP_mean(self, x):
        """
        Compute mean at points in x
        """
        return self.GP_mean_variance(x)[0]

    def GP_variance(self, x):
        """
        Compute variance at points in x
        """
        return self.GP_mean_variance(x)[1]

    def GP_mean_variance(self, x, ret_K=False):
        """
        Compute mean and variance at points in x
        """
        try:
            self._mean_variance
        except AttributeError:
            s_x = self.s_prior.new_like_self()
            self._mean_variance = theano.function(
                    [self.s_n_train, self.s_n_test, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten()
                        + s_x.flatten(),
                    [self.gprmath.s_mean(s_x),
                        self.gprmath.s_variance(s_x),
                        #self.K_fn(self.x_obs_IVL, self.x_obs_IVL),
                        self.K_fn(self.x_obs_IVL, s_x),
                        ],
                    allow_input_downcast=True)
            #theano.printing.debugprint(self._mean_variance)
        if len(x) != len(self._GP_x_all):
            raise ValueError('x has wrong len',
                    (len(x), len(self._GP_x_all)))
        x_idxset = x.idxset()
        if list(sorted(x_idxset)) != range(len(x_idxset)):
            raise ValueError('x needs re-indexing')
        rval_mean, rval_var, rval_K = self._mean_variance(
                self._GP_n_train,
                len(x_idxset),
                self._GP_y_all,
                self._GP_y_var,
                *(self._GP_x_all.flatten() + x.flatten()))

        if ret_K:
            return rval_K

        rval_var_min = rval_var.min()
        assert rval_var_min > -1e-4, rval_var_min
        rval_var = numpy.maximum(rval_var, 0)
        return (rval_mean * self._GP_y_std + self._GP_y_mean,
                rval_var * self._GP_y_std ** 2)

    def GP_train_K(self):
        return self.GP_mean_variance(self._GP_x_all, ret_K=True)

    def GP_EI(self, x):
        x_idxset = x.idxset()
        if list(sorted(x_idxset)) != range(len(x_idxset)):
            raise ValueError('x needs re-indexing')

        try:
            self._EI_fn
        except AttributeError:
            self._EI_fn = theano.function(
                    [self.s_n_train, self.s_n_test, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten()
                        + [self.cand_EI_thresh]
                        + self.cand_x.flatten(),
                    self.cand_EI,
                    allow_input_downcast=True)

        thresh = (self._GP_y_all
                - self.EI_ambition * numpy.sqrt(self._GP_y_var)).min()

        rval = self._EI_fn(self._GP_n_train,
                len(x_idxset),
                self._GP_y_all,
                self._GP_y_var,
                *(self._GP_x_all.flatten()
                    + [thresh]
                    + x.flatten()))
        assert rval.shape == (len(x_idxset),)
        return rval

    def GP_EI_optimize(self, x, maxiter=1000):
        x_idxset = x.idxset()
        if list(sorted(x_idxset)) != range(len(x_idxset)):
            raise ValueError('x needs re-indexing')

        if len(x) != len(self.kernels):
            raise ValueError('len(x) == %i but len(self.kernels)==%i' % (
                len(x), len(self.kernels)))

        n_refinable = len([k for k in self.kernels if self.is_refinable[k]])
        if n_refinable == 0:
            return x

        try:
            EI_fn_g = self._EI_fn_g
        except AttributeError:
            EI_fn_g = self._EI_fn_g = theano.function(
                    [self.s_big_param_vec] +
                    [self.s_n_train, self.s_n_test,
                        self.y_obs,
                        self.y_obs_var]
                        + self.x_obs_IVL.flatten()
                        + [self.cand_EI_thresh]
                        + self.cand_x.idxslist()
                        + [v for (k, v) in zip(self.kernels,
                            self.cand_x.valslist())
                            if not self.is_refinable[k]],
                    [-self.cand_EI.sum(),
                        -tensor.grad(self.cand_EI.sum(),
                            self.s_big_param_vec)],
                    allow_input_downcast=True,
                    mode=self.mode)
            print('Compiled EI_fn_g with %i thunks' %
                    len(EI_fn_g.maker.env.toposort()))

        thresh = (self._GP_y_all - numpy.sqrt(self._GP_y_var + 1e-8)).min()

        start_pt = numpy.asarray(
                numpy.concatenate(
                    [xk for k, xk in zip(self.kernels, x.valslist())
                        if self.is_refinable[k]]),
                dtype='float64')

        args = ((self._GP_n_train,
            len(x_idxset),
            self._GP_y_all,
            self._GP_y_var)
            + tuple(self._GP_x_all.flatten())
            + (thresh,)
            + tuple(x.idxslist())
            + tuple([v
                for (k, v) in zip(self.kernels, x.valslist())
                if not self.is_refinable[k]]))

        bounds = []
        for (k, xk) in zip(self.kernels, x.valslist()):
            if self.is_refinable[k]:
                bounds.extend([self.bounds[k]] * len(xk))

        if self.trace_on:
            def fff(*vvv):
                for i, v in enumerate(vvv):
                    self.trace(('vvv', i), numpy.asarray(v))
                f, df = EI_fn_g(*vvv)
                self.trace('f', f)
                self.trace('df', df)
                return f, df
        else:
            fff = EI_fn_g

        self.trace('start_pt', start_pt)
        for i, v in enumerate(args):
            self.trace(('args', i), numpy.asarray(v))
        self.trace('bounds', numpy.asarray(bounds))
        self.trace('maxiter', numpy.asarray(maxiter))
        best_pt, best_value, best_d = fmin_l_bfgs_b(fff,
                start_pt,
                None,
                args=args,
                maxfun=maxiter,
                bounds=bounds,
                iprint=-1)
        self.trace('best_pt', best_pt)

        # print 'BEST_PT', best_pt
        rval = x.copy()
        initial = 0
        for (_ind, iv) in enumerate(x):
            if self.is_refinable[self.kernels[_ind]]:
                diff = len(iv.vals)
                # XXX: assumes vector-valued vals (scalar elements)
                rval[_ind].vals = best_pt[initial:initial + diff]
                initial += diff
        # -- assert that all elements of best_pt have been used
        assert initial == len(best_pt)

        # -- apply any quantization required by the distributions
        self.post_refinement(rval)
        return rval

    def suggest_from_gp(self, trials, results, N):
        logger.info('suggest_from_gp')
        if N != 1:
            raise NotImplementedError('only N==1 is supported')
        ivls = self.idxs_vals_by_status(trials, results)

        prepared_data = self.prepare_GP_training_data(ivls)
        self.fit_GP(*prepared_data)

        # -- add the best previous trials as candidates
        n_trials_to_opt = self.n_candidates_to_draw // 2
        best_idxs = numpy.asarray(ivls['losses']['ok'].idxs)[
                numpy.argsort(ivls['losses']['ok'].vals)[:n_trials_to_opt]]
        best_IVLs = ivls['x_IVLs']['ok'].numeric_take(best_idxs)
        best_idxset = best_IVLs.idxset()

        # -- draw the remainder as random candidates
        candidates = self.gm_algo.suggest_from_model(ivls,
                self.n_candidates_to_draw - len(best_idxset))

        # -- re-index the best_IVLs to ensure no collision during stack
        cand_idxset = candidates.idxset()
        assert (len(cand_idxset) + len(best_idxset)
                == self.n_candidates_to_draw)
        idmap = {}
        for i in best_idxset:
            if i in cand_idxset:
                idmap[i] = (max(cand_idxset) + max(best_idxset) +
                        len(idmap) + 1)
            else:
                idmap[i] = i
            assert idmap[i] not in cand_idxset
        assert (len(cand_idxset.union(idmap.values()))
                == self.n_candidates_to_draw)
        best_IVLs.reindex(idmap)
        candidates = candidates.as_list()
        candidates.stack(best_IVLs)
        assert len(candidates.idxset()) == self.n_candidates_to_draw
        # XXX: rather than reindex here, take advantage of fact that random
        #      candidates were already contiguously indexed and stack
        #      appropriately reindexed trials on top of them.
        candidates.reindex()
        candidates = candidates.as_numpy()

        candidates_opt = self.GP_EI_optimize(candidates)

        EI_opt = self.GP_EI(candidates_opt)
        best_idx = numpy.argmax(EI_opt)
        if 1:
            # for DEBUGGING
            EI = self.GP_EI(candidates)
            if EI.max() > EI_opt.max():
                logger.warn(
                    'Optimization actually *decreased* EI!? %.3f -> %.3f' % (
                        EI.max(), EI_opt.max()))
        rval = candidates_opt.numeric_take([best_idx])
        return rval

    def suggest_from_gm(self, trials, results, N):
        logger.info('suggest_from_gm')
        ivls = self.idxs_vals_by_status(trials, results)
        rval = self.gm_algo.suggest_from_model(ivls, N)
        return rval

    def suggest_from_prior(self, trials, results, N):
        logger.info('suggest_from_prior')
        if not hasattr(self, '_prior_sampler'):
            self._prior_sampler = theano.function(
                    [self.s_N],
                    self.s_prior.flatten(),
                    mode=self.mode)
        rvals = self._prior_sampler(N)
        return IdxsValsList.fromflattened(rvals)

    def suggest(self, trials, results, N):
        ivls = self.idxs_vals_by_status(trials, results)
        t0 = time.time()
        n_ok = len(ivls['losses']['ok'].idxs)

        # -- choose the suggestion strategy (heuristic)
        if n_ok < self.n_startup_jobs:
            fn = self.suggest_from_prior
        else:
            # -- figure out how long (in iterations) it has been since picking a
            #    winner: `winner_age`
            assert (list(ivls['losses']['ok'].idxs)
                    == list(sorted(ivls['losses']['ok'].idxs)))
            t_winner = numpy.asarray(ivls['losses']['ok'].vals).argmin()
            winner_age = n_ok - t_winner
            if winner_age < self.local_improvement_patience:
                fn = self.suggest_from_gp
            else:
                if self.numpy_rng.rand() < self.p_GP_during_exploration:
                    fn = self.suggest_from_gp
                else:
                    fn = self.suggest_from_gm
        try:
            rval = self.suggest_ivl(fn(trials, results, N))
        finally:
            logger.info('suggest %i took %.2f seconds' % (
                    len(ivls['losses']['ok'].idxs),
                    time.time() - t0))
        return rval


def HGP(bandit):
    return GP_BanditAlgo(bandit)
