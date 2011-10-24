"""
Gaussian-process (GP)-based optimization algorithm using Theano
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import logging
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

import numpy
import scipy.optimize
import theano
from theano import tensor
from theano.sandbox.linalg import (diag, matrix_inverse, det, PSD_hint, trace)
import montetheano
import montetheano.distributions as mt_dist

from idxs_vals_rnd import IdxsVals, IdxsValsList
from theano_bandit_algos import TheanoBanditAlgo
from theano_gm import AdaptiveParzenGM

if 0:
    from scipy.optimize import fmin_l_bfgs_b
else:
    # in theory this is about twice as fast as scipy's because
    # it permits passing a combination f and df implementation
    # that theano has compiled
    from lbfgsb import fmin_l_bfgs_b

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
        #print 'SparseGramGet base', base.shape
        #print 'SparseGramGet i0', i0
        #print 'SparseGramGet i1', i1
        storage[0][0] = base[i0[:,None], i1]

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
        if amt.ndim != 2:
            raise TypeError('amt not matrix', base)
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
        if amt.size == 0:
            return

        if 0:
            print 'SparseGramSet operation', self.operation, [id(n) for n in node.inputs]
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
            rval[i0[:,None], i1] = amt
        elif 'inc' == self.operation:
            rval[i0[:,None], i1] += amt
        elif 'mul' == self.operation:
            rval[i0[:,None], i1] *= amt
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
        return [gbase, gamt, None, None]

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
        (low,high), = self.param_bounds()
        if low is not None:
            low = self.lenscale(low)
        if high is not None:
            high = self.lenscale(high)
        return "%s{l=%s,bounds=(%s,%s)}"%(
                    self.__class__.__name__,
                    str(l), str(low), str(high))

    def params(self):
        return [self.log_lenscale]

    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        if x.ndim == y.ndim == 1:
            x = x.dimshuffle(0, 'x')
            y = y.dimshuffle(0, 'x')
        ll2 = tensor.exp(self.log_lenscale) #2l^2
        d = ((x**2).sum(axis=1).dimshuffle(0, 'x')
                + (y ** 2).sum(axis=1)
                - 2 * tensor.dot(x, y.T))
        K = tensor.exp(-d / ll2)
        return K


class LogSquaredExponentialKernel(SquaredExponentialKernel):

    def K(self, x, y):
        if x.ndim == y.ndim == 1:
            x = x.dimshuffle(0, 'x')
            y = y.dimshuffle(0, 'x')
        ll2 = tensor.exp(self.log_lenscale) #2l^2
        log = tensor.log
        lx = log(x)
        ly = log(y)
        d = ((lx**2).sum(axis=1).dimshuffle(0, 'x')
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
        if self.log_lenscale.ndim!=0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)
    def __str__(self):
        l = numpy.exp(self.log_lenscale.value)
        return "ExponentialKernel{l=%s}"%str(l)

    @classmethod
    def alloc(cls, l=1):
        log_l = numpy.log(l)
        log_lenscale = theano.shared(log_l)
        return cls(log_lenscale=log_lenscale)

    def K(self, x, y):
        l = tensor.exp(self.log_lenscale)
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * tensor.dot(x, y.T))
        K = tensor.exp(-tensor.sqrt(d)/l)
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
        xx = xx.dimshuffle(0,'x') # drop cols because there should only be 1
        yy = yy.dimshuffle(0)     # drop cols because there should only be 1

        ll2 = tensor.exp(self.log_lenscale) #2l^2
        d = tensor.neq(xx,yy)
        K = tensor.exp(-d/ll2)
        return K


class GPR_math(object):
    """
    Formulae for Gaussian Process Regression

    # K - the gram matrix of training data
    # y - the vector of training data values
    # var_y - the vector of training data variances

    """

    def __init__(self, x, y, var_y, K_fn, N=None, min_variance=1e-6):
        self.x = x
        self.y = tensor.as_tensor_variable(y)
        self.var_y = tensor.as_tensor_variable(var_y)
        self.K_fn = K_fn
        self.min_variance = min_variance
        if N is None:
            self.N = self.y.shape[0]
        else:
            self.N = N

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
        rK = PSD_hint(K + var_y * tensor.eye(N))
        nll = (0.5 * dots(y, matrix_inverse(rK), y)
                + 0.5 * tensor.log(det(rK))
                + N / 2.0 * tensor.log(2 * numpy.pi))
        return nll

    def s_mean(self, x):
        """Gaussian Process mean at points x"""
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        alpha = tensor.dot(matrix_inverse(rK), y)

        K_x = self.K_fn(self.x, x)
        y_x = tensor.dot(alpha, K_x)
        return y_x

    def s_variance(self, x):
        """Gaussian Process variance at points x"""
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        K_x = self.K_fn(self.x, x)
        var_x = 1 - diag(dots(K_x.T, matrix_inverse(rK), K_x))
        return var_x

    def s_deg_of_freedom(self):
        """
        Degrees of freedom aka "effective number of parameters"
        of kernel smoother.

        Defined pg. 25 of Rasmussen & Williams.
        """
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        dof = trace(tensor.dot(K, matrix_inverse(rK)))
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
        return rval

    def softmax_hack(self, x, thresh):
        """
        return approximation of \log(\int_{-inf}^{thresh} y p(y|x) dy)
        """

        mu = self.s_mean(x)
        sigma = tensor.sqrt(
                tensor.maximum(self.s_variance(x),
                    self.min_variance))
        rval = tensor.log1p(tensor.exp((thresh - mu) / sigma))
        return rval

def get_refinability(v,dist_name):
    v = v.vals
    print(v.owner.op, dir(v.owner.op))
    if dist_name == 'uniform':
        params = [mt_dist.uniform_get_low(v), mt_dist.uniform_get_high(v)]
    elif dist_name == 'normal':
        params = [mt_dist.normal_get_mu(v), mt_dist.normal_get_sigma(v)]
    elif dist_name == 'lognormal':
        params = [mt_dist.lognormal_get_mu(v), mt_dist.lognormal_get_sigma(v)]
    elif dist_name == 'quantized_lognormal':
        params = [mt_dist.qlognormal_get_mu(v),
                  mt_dist.qlognormal_get_sigma(v),
                  mt_dist.qlognormal_get_round(v)]
    for p in params:
        try:
            tensor.get_constant_value(p)
        except TypeError:
            return False
    return True        
    
class GP_BanditAlgo(TheanoBanditAlgo):

    constant_liar_global_mean = True
    #XXX: True/False is bad interface for this
    # one scheme is to use running mean
    # other scheme estimates mean from startup jobs

    use_cg = False
    # what optimizer to use for fitting GP, optimizing candidates?

    params_l2_penalty = 0
    # fitting penalty on the lengthscales of kernels
    # might make sense to make this negative to blur out the ML solution.

    mode = None  # to use theano's default

    n_startup_jobs = 30  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic
    y_minvar = 1e-6

    # EI_criterion can be
    #   EI - expected improvement (numerically bad)
    #   log_EI - log(EI)
    #          (mathematically correct,
    #           numerically good,
    #           is there an analytic form?)
    #           Not currently implemented
    #   softmax_hack - log(1 + sigmoid((mu-thresh)/sigma))
    #       - mathematically incorrect
    #       - numerically good
    #       - analytic form, fast computation
    EI_criterion = 'softmax_hack'

    n_candidates_to_draw = 5  # XXX: make this bigger for non-debugging

    n_candidates_to_draw_in_GM = 5 # XXX: make this bigger and only refine best

    def init_kernels(self):
        self.kernels = []
        self.is_refinable = {}
        self.bounds = {}
        self.params = []
        self.param_bounds = []
        self.idxs_mulsets = {}


        for iv in self.s_prior:
            dist_name = montetheano.rstreams.rv_dist_name(iv.vals)
            if dist_name == 'normal':
                k = SquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv,dist_name)                                                        
                self.bounds[k] = (None,None)
            elif dist_name == 'uniform':
                k = SquaredExponentialKernel()
                low = tensor.get_constant_value(mt_dist.uniform_get_low(iv.vals))
                high = tensor.get_constant_value(mt_dist.uniform_get_high(iv.vals))
                self.is_refinable[k] = get_refinability(iv,dist_name) 
                self.bounds[k] = (low,high)
            elif dist_name == 'lognormal':
                k = LogSquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv,dist_name) 
                self.bounds[k] = (None,None)
            elif dist_name == 'quantized_lognormal':
                k = LogSquaredExponentialKernel()
                self.is_refinable[k] = get_refinability(iv,dist_name) 
                self.bounds[k] = (None,None)
            elif dist_name == 'categorical':
                # XXX: a better CategoryKernel would have different similarities
                #      for different choices
                k = CategoryKernel()
                self.is_refinable[k] = False
                self.bounds[k] = (None,None)
            else:
                raise TypeError("unsupported distribution", dist_name)

            self.kernels.append(k)
            self.params.extend(k.params())
            self.param_bounds.extend(k.param_bounds())
            # XXX : to be more robust, it would be nice to build an Env with the
            #       idxs as outputs, and then run the MergeOptimizer on it.
            self.idxs_mulsets.setdefault(iv.idxs, []).append(k)


    def init_gram_weights(self, idxs=None, parent_weight=1):
        """ Recursively
        parent_k: current, None for root
        """
        try:
            gram_weights = self.gram_weights
        except AttributeError:
            self.convex_coefficient_params = []
            gram_weights = self.gram_weights = {}

        if idxs is None:
            root_idxs = [idxs for idxs in self.idxs_mulsets
                    if isinstance(idxs.owner.op, tensor.ARange)]
            if len(root_idxs) > 1:
                # XXX : to be more robust, it would be nice to build an Env with
                # the idxs as outputs, and then run the MergeOptimizer on it.
                raise NotImplementedError('not all idxs were derived from'
                    ' single ARange object')
            idxs = root_idxs[0]

        kerns = self.idxs_mulsets[idxs]
        cat_kerns = [k for k in kerns if isinstance(k, CategoryKernel)]
        if len(cat_kerns) == 0:
            self.gram_weights[idxs] = parent_weight
        elif len(cat_kerns) == 1:
            param = theano.shared(numpy.asarray(0.0))
            self.convex_coefficient_params.append(param)
            weight = tensor.nnet.sigmoid(param)
            for i, k in enumerate(cat_kerns):
                # call recursively for each mulset
                # that corresponds to a slice out of idxs
                for sub_idxs in self.idxs_mulsets:
                    if idxs == sub_idxs.owner.inputs[0]:
                        # choice nodes in genson generate
                        # advanced indexing on idxs.
                        if not isinstance(sub_idxs.owner.op,
                                tensor.AdvancedSubtensor1):
                            raise NotImplementedError(sub_idxs)
                        self.init_gram_weights(sub_idxs,
                                parent_weight=parent_weight * weight)
            #print 'adding gram_weight', idxs
            #theano.printing.debugprint(parent_weight * (1 - weight))
            self.gram_weights[idxs] = parent_weight * (1 - weight)
        else:
            # in this case the parent_weight must be shared between
            # this mulset and the child mulsets
            raise NotImplementedError()
            n_terms = len(cat_kerns) + 1
            params = theano.shared(numpy.zeros(n_terms))
            self.convex_coefficient_params.append(params)
            weights = tensor.nnet.softmax(params) * parent_weight
            for i, k in enumerate(cat_kerns):
                build_softmax_coefs(k, weights[i])
            self.gram_weights[idxs] = weights[len(cat_kerns)]

    def __init__(self, bandit):
        TheanoBanditAlgo.__init__(self, bandit)
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
                self.K_fn,
                N=self.s_n_train,
                min_variance = self.y_minvar)

        self.nll_obs = self.gprmath.s_nll()

        if self.EI_criterion == 'EI':
            self.cand_EI = self.gprmath.s_expectation_lt_thresh(
                    self.cand_x,
                    self.cand_EI_thresh)
        elif self.EI_criterion == 'softmax_hack':
            self.cand_EI = self.gprmath.softmax_hack(
                    self.cand_x,
                    self.cand_EI_thresh)
        else:
            raise ValueError('EI_criterion', self.EI_criterion)

        # self.gm_algo is used to draw candidates for subsequent refinement
        # It is also entirely responsible for choosing categorical variables.
        self.gm_algo = AdaptiveParzenGM(self.bandit)

    def K_fn(self, x0, x1):
        """
        :param x0: an IdxsValsList of symbolic variables
        :param x1: an IdxsValsList of symbolic variables

        :returns: symbolic gram matrix
        """

        gram_matrices = {}
        gram_matrices_idxs = {}
        for k, iv_prior, iv0, iv1 in zip(self.kernels, self.s_prior, x0, x1):
            gram_matrices.setdefault(iv_prior.idxs, []).append(
                    k.K(iv0.vals, iv1.vals))
            gram_matrices_idxs.setdefault(iv_prior.idxs, [iv0.idxs, iv1.idxs])

        nx1 = self.s_n_train if x1 is x0 else self.s_n_test
        base = tensor.alloc(0.0, self.s_n_train, nx1)
        for idxs, grams in gram_matrices.items():
            prod = self.gram_weights[idxs] * tensor.mul(*grams)
            base = sparse_gram_inc(base, prod, *gram_matrices_idxs[idxs])

        return base

    def prepare_GP_training_data(self, ivls):
        y_mean = numpy.mean(ivls['losses']['ok'].vals)
        y_std = numpy.std(ivls['losses']['ok'].vals)

        x_all = ivls['x_IVLs']['ok'].as_list()
        y_all_iv = ivls['losses']['ok'].as_list()
        y_var_iv = ivls['losses_variance']['ok'].as_list()

        if self.constant_liar_global_mean:
            liar_y_mean = y_mean
            liar_y_var = numpy.mean(ivls['losses_variance']['ok'].vals)
        else:
            raise NotImplementedError()

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
        y_var /= (1e-8 + y_std)**2

        assert y_all.shape == y_var.shape
        if y_var.min() < -1e-6:
            raise ValueError('negative variance encountered in results')
        y_var = numpy.maximum(y_var, self.y_minvar)
        return x_all, y_all, y_mean, y_var, y_std

    def fit_GP(self, x_all, y_all, y_mean, y_var, y_std, maxiter=100):
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
                    allow_input_downcast=True)
            dnll_dparams = self.dnll_dparams = theano.function(
                    [self.s_n_train, self.s_n_test, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten(),
                    tensor.grad(cost, self.params),
                    allow_input_downcast=True)

        lbounds = []
        ubounds = []
        for lb, ub in self.param_bounds:
            lbounds.extend(numpy.asarray(value(lb)).flatten())
            ubounds.extend(numpy.asarray(value(ub)).flatten())
        bounds = numpy.asarray([lbounds, ubounds]).T

        def get_pt():
            rval = []
            for p in self.params:
                v = p.get_value().flatten()
                rval.extend(v)
            return numpy.asarray(rval)
        def set_pt(pt):
            i = 0
            for p in self.params:
                shape = p.get_value(borrow=True).shape
                size = numpy.prod(shape)
                p.set_value(pt[i:i + size].reshape(shape))
                i += size
            assert i == len(pt)
            #print self.kernel.summary()

        def f(pt):
            set_pt(pt)
            return nll_fn(self._GP_n_train,
                    self._GP_n_train,
                    self._GP_y_all,
                    self._GP_y_var,
                    *self._GP_x_all.flatten())
        def df(pt):
            set_pt(pt)
            dparams = dnll_dparams(self._GP_n_train,
                    self._GP_n_train,
                    self._GP_y_all,
                    self._GP_y_var,
                    *self._GP_x_all.flatten())
            rval = []
            for dp in dparams:
                rval.extend(dp.flatten())
            rval =  numpy.asarray(rval)
            #print numpy.sqrt((rval**2).sum())
            return rval

        best_pt, best_value, best_d = fmin_l_bfgs_b(f,
                get_pt(),
                df,
                maxfun=maxiter,
                bounds=bounds,
                iprint=-1)
        logger.info('fit_GP best value: %f' % best_value)
        set_pt(best_pt)
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

    def GP_mean_variance(self, x):
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

        for K_row in rval_K:
            print K_row
        rval_var_min = rval_var.min()
        assert rval_var_min > -1e-4, rval_var_min
        rval_var = numpy.maximum(rval_var, 0)
        return (rval_mean * self._GP_y_std + self._GP_y_mean,
                rval_var * self._GP_y_std**2)

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

        thresh = (self._GP_y_all - numpy.sqrt(self._GP_y_var)).min()

        rval = self._EI_fn(self._GP_n_train,
                len(x_idxset),
                self._GP_y_all,
                self._GP_y_var,
                *(self._GP_x_all.flatten()
                    + [thresh]
                    + x.flatten()))
        assert rval.shape == (len(x_idxset),)
        return rval

    def GP_EI_optimize(self, x, maxiter=100):
        x_idxset = x.idxset()
        if list(sorted(x_idxset)) != range(len(x_idxset)):
            raise ValueError('x needs re-indexing')

        
        if len(x) != len(self.kernels):
            raise ValueError('x has wrong length,',len(x) ,'!=',len(self.kernels))

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
                    allow_input_downcast=True)

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
                
        print('THING',[len(xk) for k, xk in zip(self.kernels, x.valslist())
                        if self.is_refinable[k]])
        print(len(start_pt))
                        
        bounds = []
        for (k,xk) in zip(self.kernels, x.valslist()):
            if self.is_refinable[k]:
                bounds.extend([self.bounds[k][:] for _ind in range(len(xk))])
        print(bounds)
        
        best_pt, best_value, best_d = fmin_l_bfgs_b(EI_fn_g,
                start_pt,
                None,
                args = args,
                maxfun=maxiter,
                #TODO: bounds from distributions
                bounds=bounds,
                iprint=-1)

        #print 'BEST_PT', best_pt
        rval = x.copy()
        initial = 0
        for (_ind, iv) in enumerate(x):
            if self.is_refinable[self.kernels[_ind]]:
                diff = len(iv.vals)
                rval[_ind].vals = best_pt[initial:initial+diff]
                initial += diff
        assert initial == len(best_pt)
        return rval

    def suggest_from_model(self, trials, results, N):
        logger.debug('suggest_from_model')
        ivls = self.idxs_vals_by_status(trials, results)
        prepared_data = self.prepare_GP_training_data(ivls)
        self.fit_GP(*prepared_data)

        self.gm_algo.n_EI_candidates = self.n_candidates_to_draw_in_GM
        candidates = self.gm_algo.suggest_from_model(ivls,
                self.n_candidates_to_draw)

        candidates_opt = self.GP_EI_optimize(candidates)

        EI_opt = self.GP_EI(candidates_opt)
        best_idx = numpy.argmax(EI_opt)

        if 1: # for DEBUGGING
            EI = self.GP_EI(candidates)
            if EI.max() > EI_opt.max():
                logger.warn(
                    'Optimization actually *decreased* EI!? %.3f -> %.3f' % (
                        EI.max(), EI_opt.max()))

        rval = candidates_opt.numeric_take([best_idx])
        return rval

    def suggest_from_prior(self, N):
        logger.debug('suggest_from_prior')
        if not hasattr(self, '_prior_sampler'):
            self._prior_sampler = theano.function(
                    [self.s_N],
                    self.s_prior.flatten(),
                    mode=self.mode)
        rvals = self._prior_sampler(N)
        return IdxsValsList.fromflattened(rvals)

    def suggest(self, trials, results, N):
        ivls = self.idxs_vals_by_status(trials, results)
        t = time.time()
        try:
            if len(ivls['losses']['ok'].idxs) < self.n_startup_jobs:
                return self.suggest_ivl(
                        self.suggest_from_prior(N))
            else:
                return self.suggest_ivl(
                        self.suggest_from_model(trials, results, N))

        finally:
            print 'suggest %i took %.2f seconds' % (
                    len(ivls['losses']['ok'].idxs),
                    time.time() - t)

